"""
chunker.py
----------
Semantic chunker for Markdown files produced by Azure Document Intelligence.

Improvements over v1:
  - Process table detection: DR0/DR1.0 etc. tables get process_number + process_name
    extracted as structured metadata and prepended to content for better retrieval
  - Figure/flowchart blocks: extracted with their AI descriptions instead of dropped
  - Document version metadata: extracted from PageFooter comments (e.g. VERSION 2.3)
  - DR process number extracted from headings (## DR 1.1 ...) as metadata field
  - Process rule bullet lists preserved with their parent process context
  - Smarter table-to-text conversion: HTML tables rendered as readable key:value pairs
    for process definition tables, kept as HTML for data/code tables
  - Page header repetition suppressed (same header seen twice = skip second)
"""

import re
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Configuration ─────────────────────────────────────────────────────────────

CHUNK_MAX_TOKENS     = 512
CHUNK_OVERLAP_TOKENS = 64
CHARS_PER_TOKEN      = 4

# Process table field labels — tables whose first column matches these are
# "process definition" tables and get special handling
PROCESS_TABLE_LABELS = {
    "PROCESS NUMBER", "PROCESS NAME", "PROCESS DEFINITION",
    "TRIGGER(S)", "PROCESS INPUTS", "PROCESS OUTPUTS",
    "SUB OR PRECEDING PROCESSES", "PROCESS RULES", "COMMENTS", "COMMENT",
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:       str
    source:         str
    page:           Optional[int]
    section:        str           # heading breadcrumb
    content:        str           # text sent to embedding + LLM
    is_table:       bool = False
    metadata:       dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_id(source: str, index: int, content: str) -> str:
    h = hashlib.md5(f"{source}:{index}:{content[:80]}".encode()).hexdigest()[:12]
    return f"{Path(source).stem}_{index:04d}_{h}"


def _split_text_with_overlap(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    max_chars     = max_tokens     * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("\n", start, end),
            )
            if boundary > start + overlap_chars:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - overlap_chars
    return [c for c in chunks if c.strip()]


# ── HTML table parser ─────────────────────────────────────────────────────────

def _extract_table_cells(html: str) -> list[list[str]]:
    """Parse HTML table into list of rows, each row a list of cell texts."""
    rows = []
    for row_match in re.finditer(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE):
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_match.group(1), re.DOTALL | re.IGNORECASE)
        cleaned = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        cleaned = [re.sub(r'\s+', ' ', c) for c in cleaned]
        if any(cleaned):
            rows.append(cleaned)
    return rows


def _is_process_table(rows: list[list[str]]) -> bool:
    """Return True if this table looks like a DR process definition table."""
    if not rows:
        return False
    first_col_values = {r[0].upper().rstrip(':') for r in rows if r}
    return bool(first_col_values & PROCESS_TABLE_LABELS)


def _process_table_to_text(rows: list[list[str]]) -> tuple[str, dict]:
    """
    Convert a process definition table to readable text + structured metadata.
    Returns (text_for_embedding, metadata_dict)
    """
    fields = {}
    for row in rows:
        if len(row) >= 2:
            key = row[0].rstrip(':').strip()
            val = " ".join(row[1:]).strip()
            if key and val:
                fields[key] = val
        elif len(row) == 1 and row[0]:
            # continuation cell (rowspan) — append to last key
            if fields:
                last_key = list(fields.keys())[-1]
                fields[last_key] += " " + row[0]

    # Build readable text block
    lines = []
    priority_keys = [
        "PROCESS NUMBER", "PROCESS NAME", "PROCESS DEFINITION",
        "TRIGGER(S)", "PROCESS RULES", "PROCESS INPUTS", "PROCESS OUTPUTS",
        "SUB OR PRECEDING PROCESSES", "COMMENTS", "COMMENT",
    ]
    seen = set()
    for key in priority_keys:
        if key in fields:
            lines.append(f"{key}: {fields[key]}")
            seen.add(key)
    for key, val in fields.items():
        if key not in seen:
            lines.append(f"{key}: {val}")

    text = "\n".join(lines)

    metadata = {
        "process_number": fields.get("PROCESS NUMBER", ""),
        "process_name":   fields.get("PROCESS NAME", ""),
        "table_type":     "process_definition",
    }
    return text, metadata


def _data_table_to_text(rows: list[list[str]]) -> str:
    """Convert a generic data/code table to pipe-delimited markdown text."""
    if not rows:
        return ""
    lines = []
    for row in rows:
        lines.append(" | ".join(row))
    return "\n".join(lines)


# ── Figure/flowchart handler ──────────────────────────────────────────────────

def _extract_figure_text(figure_html: str) -> str:
    """
    Extract the AI description from a <figure> block and any visible text.
    Returns a clean text summary suitable for embedding.
    """
    # Extract AI description from comment
    desc_match = re.search(r'<!--\s*AI Description:(.*?)-->', figure_html, re.DOTALL)
    ai_desc = ""
    if desc_match:
        ai_desc = re.sub(r'\s+', ' ', desc_match.group(1)).strip()

    # Extract any plain text outside the comment (step labels etc.)
    plain = re.sub(r'<!--.*?-->', '', figure_html, flags=re.DOTALL)
    plain = re.sub(r'<[^>]+>', '', plain)
    plain = re.sub(r'[☒✓□]', '', plain)   # remove checkbox chars
    plain = re.sub(r'\s+', ' ', plain).strip()

    parts = []
    if ai_desc:
        parts.append(f"[Flowchart Description] {ai_desc}")
    if plain and len(plain) > 20:
        parts.append(f"[Process Steps] {plain}")
    return "\n".join(parts)


# ── Document metadata extractor ───────────────────────────────────────────────

def _extract_doc_metadata(text: str) -> dict:
    """Extract document-level metadata from the full text."""
    meta = {}

    # Version from PageFooter e.g. VERSION 2.3 - DROP REQUEST & RESPONSE
    ver = re.search(r'VERSION\s+([\d.]+)', text, re.IGNORECASE)
    if ver:
        meta["doc_version"] = ver.group(1)

    # Case number
    case = re.search(r'Case\s+([\d-]+[A-Z]-[\d]+)', text, re.IGNORECASE)
    if case:
        meta["case_number"] = case.group(1)

    # Supplement
    supp = re.search(r'Supplement\s+([A-Z0-9]+)', text, re.IGNORECASE)
    if supp:
        meta["supplement"] = supp.group(1)

    return meta


# ── DR process number extractor ───────────────────────────────────────────────

def _extract_dr_number(heading: str) -> Optional[str]:
    """Extract DR process number from heading text. e.g. 'DR 1.1' from '## DR 1.1 Customer...'"""
    m = re.match(r'^(DR\s*[\d.]+)', heading.strip(), re.IGNORECASE)
    return m.group(1).strip() if m else None


# ── Core parser ───────────────────────────────────────────────────────────────

def parse_markdown_to_chunks(filepath: str | Path) -> list[Chunk]:
    filepath = Path(filepath)
    text     = filepath.read_text(encoding="utf-8")
    source   = filepath.name

    # Extract document-level metadata once
    doc_meta = _extract_doc_metadata(text)

    chunks: list[Chunk]      = []
    heading_stack: list[str] = []
    current_page: Optional[int] = None
    current_dr_number: Optional[str] = None
    chunk_index = 0
    seen_headers: set[str]   = set()   # suppress repeated page headers

    lines = text.splitlines(keepends=True)
    buffer: list[str] = []

    def flush_buffer():
        nonlocal chunk_index
        raw = "".join(buffer).strip()
        if not raw:
            return
        section = " > ".join(heading_stack) if heading_stack else "Document"
        for piece in _split_text_with_overlap(raw, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS):
            if not piece.strip():
                continue
            meta = {**doc_meta, "char_count": len(piece)}
            if current_dr_number:
                meta["dr_process"] = current_dr_number
            chunks.append(Chunk(
                chunk_id = _make_id(source, chunk_index, piece),
                source   = source,
                page     = current_page,
                section  = section,
                content  = piece,
                is_table = False,
                metadata = meta,
            ))
            chunk_index += 1
        buffer.clear()

    # ── State machine ──────────────────────────────────────────────────────
    in_table   = False
    in_figure  = False
    table_buf: list[str] = []
    figure_buf: list[str] = []

    for line in lines:
        stripped = line.strip()

        # ── Page number ───────────────────────────────────────────────────
        page_match = re.search(r'PageNumber.*?(\d+)', stripped)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        # ── PageHeader — suppress duplicates ─────────────────────────────
        if re.match(r'<!--\s*PageHeader', stripped):
            header_val = re.search(r'PageHeader="([^"]+)"', stripped)
            if header_val:
                hv = header_val.group(1)
                if hv in seen_headers:
                    continue
                seen_headers.add(hv)
            continue

        # ── PageFooter — extract version, skip the line ───────────────────
        if re.match(r'<!--\s*PageFooter', stripped):
            continue

        # ── Figure block ──────────────────────────────────────────────────
        if re.match(r'<figure', stripped, re.IGNORECASE):
            flush_buffer()
            in_figure = True
            figure_buf = [line]
            continue

        if in_figure:
            figure_buf.append(line)
            if re.match(r'</figure', stripped, re.IGNORECASE):
                in_figure = False
                fig_html  = "".join(figure_buf)
                fig_text  = _extract_figure_text(fig_html)
                if fig_text.strip():
                    section = " > ".join(heading_stack) if heading_stack else "Document"
                    meta = {**doc_meta, "chunk_type": "flowchart"}
                    if current_dr_number:
                        meta["dr_process"] = current_dr_number
                    chunks.append(Chunk(
                        chunk_id = _make_id(source, chunk_index, fig_text),
                        source   = source,
                        page     = current_page,
                        section  = section,
                        content  = fig_text,
                        is_table = False,
                        metadata = meta,
                    ))
                    chunk_index += 1
                figure_buf = []
            continue

        # ── HTML table ────────────────────────────────────────────────────
        if re.match(r'<table', stripped, re.IGNORECASE):
            flush_buffer()
            in_table = True
            table_buf = [line]
            continue

        if in_table:
            table_buf.append(line)
            if re.match(r'</table', stripped, re.IGNORECASE):
                in_table  = False
                table_html = "".join(table_buf)
                rows       = _extract_table_cells(table_html)
                section    = " > ".join(heading_stack) if heading_stack else "Document"

                if _is_process_table(rows):
                    # Convert to readable key:value text, extract process metadata
                    content, tbl_meta = _process_table_to_text(rows)
                    meta = {**doc_meta, **tbl_meta}
                    if current_dr_number:
                        meta["dr_process"] = current_dr_number
                    # Update dr_number if this table has a PROCESS NUMBER field
                    if tbl_meta.get("process_number"):
                        current_dr_number = tbl_meta["process_number"]
                    # Prepend section heading so the chunk is self-contained
                    full_content = f"[Process: {tbl_meta.get('process_number','')} - {tbl_meta.get('process_name','')}]\n{content}"
                    for piece in _split_text_with_overlap(full_content, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS):
                        chunks.append(Chunk(
                            chunk_id = _make_id(source, chunk_index, piece),
                            source   = source,
                            page     = current_page,
                            section  = section,
                            content  = piece,
                            is_table = True,
                            metadata = meta,
                        ))
                        chunk_index += 1
                else:
                    # Generic data table — pipe-delimited text
                    content = _data_table_to_text(rows)
                    if content.strip():
                        meta = {**doc_meta, "table_type": "data", "row_count": len(rows)}
                        if current_dr_number:
                            meta["dr_process"] = current_dr_number
                        chunks.append(Chunk(
                            chunk_id = _make_id(source, chunk_index, content),
                            source   = source,
                            page     = current_page,
                            section  = section,
                            content  = content,
                            is_table = True,
                            metadata = meta,
                        ))
                        chunk_index += 1
                table_buf = []
            continue

        # ── Markdown pipe tables ──────────────────────────────────────────
        if stripped.startswith("|"):
            flush_buffer()
            table_buf.append(line)
            continue
        elif table_buf and not stripped.startswith("|"):
            table_text = "".join(table_buf).strip()
            section    = " > ".join(heading_stack) if heading_stack else "Document"
            chunks.append(Chunk(
                chunk_id = _make_id(source, chunk_index, table_text),
                source   = source,
                page     = current_page,
                section  = section,
                content  = table_text,
                is_table = True,
                metadata = {**doc_meta},
            ))
            chunk_index += 1
            table_buf = []

        # ── Headings ──────────────────────────────────────────────────────
        heading_match = re.match(r'^(#{1,3})\s+(.*)', stripped)
        if heading_match:
            flush_buffer()
            level   = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            heading_stack = heading_stack[:level - 1]
            heading_stack.append(heading)
            # Extract DR process number from heading
            dr = _extract_dr_number(heading)
            if dr:
                current_dr_number = dr
            continue

        # ── Horizontal rule = section break ───────────────────────────────
        if re.match(r'^-{3,}$', stripped):
            flush_buffer()
            continue

        # ── Regular text ──────────────────────────────────────────────────
        buffer.append(line)

    flush_buffer()
    return chunks


# ── Batch processor ───────────────────────────────────────────────────────────

def chunk_directory(folder: str | Path, glob: str = "**/*.md") -> list[Chunk]:
    folder = Path(folder)
    all_chunks: list[Chunk] = []
    files = list(folder.glob(glob))
    print(f"Found {len(files)} markdown files in '{folder}'")
    for f in files:
        file_chunks = parse_markdown_to_chunks(f)
        print(f"  {f.name}: {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "./documents"
    chunks = chunk_directory(folder)

    out_path = Path("chunks_preview.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks[:30]], f, indent=2, ensure_ascii=False)
    print(f"\nFirst 30 chunks written to {out_path}")