"""
Semantic Chunker for EDI 814 documents.

Strategy:
  1. Parse the document into logical sections (pages, tables, paragraphs)
  2. Respect EDI structural boundaries (segments, elements, version blocks)
  3. Embed metadata into each chunk (page, section type, segment codes found)
  4. Merge small chunks, split oversized ones — always on sentence boundaries
"""

import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from bs4 import BeautifulSoup
import tiktoken

from config.settings import chunking_cfg

logger = logging.getLogger(__name__)

# Tokenizer for accurate token counting (matches ada-002)
TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    content: str
    content_type: str           # "paragraph" | "table" | "header" | "mixed"
    page_number: int
    section_title: str
    segment_codes: list[str]    # EDI codes found, e.g. ["REF*1P", "DTM*151"]
    version_refs: list[str]     # Version strings found, e.g. ["1.3", "1.6"]
    token_count: int
    source_file: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Helpers ───────────────────────────────────────────────────────────────────

EDI_CODE_PATTERN = re.compile(
    r'\b(REF|DTM|N1|N3|N4|LIN|ASI|PER|BGN|ST|SE|GS|GE|ISA|IEA)'
    r'[*\^]?([A-Z0-9]{1,3})?\b'
)
VERSION_PATTERN = re.compile(r'\bVersion\s+(\d+\.\d+)\b', re.IGNORECASE)
PAGE_PATTERN    = re.compile(r'<!--\s*Page(?:Number|Header|Footer)?="?([^"]*)"?\s*-->')
PAGE_BREAK      = re.compile(r'<!--\s*Page\s+\d+\s*-->')


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def extract_edi_codes(text: str) -> list[str]:
    return list({m.group(0).strip() for m in EDI_CODE_PATTERN.finditer(text)})


def extract_versions(text: str) -> list[str]:
    return list({m.group(1) for m in VERSION_PATTERN.finditer(text)})


def clean_html_table(html: str) -> str:
    """Convert HTML table to plain pipe-delimited text."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def is_semantic_boundary(line: str) -> bool:
    for pattern in chunking_cfg.semantic_boundaries:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


# ── Core chunker ──────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Splits an EDI markdown document into semantically coherent chunks.

    Pipeline:
        raw markdown → split into sections → split oversized sections
                     → merge undersized sections → annotate metadata
    """

    def __init__(self, source_file: str = "edi814.md"):
        self.source_file = source_file
        self.cfg = chunking_cfg

    # ── Public entry point ────────────────────────────────────────────────────

    def chunk(self, markdown_text: str) -> list[Chunk]:
        pages = self._split_into_pages(markdown_text)
        raw_sections = []
        for page_num, page_text in pages:
            raw_sections.extend(self._split_page_into_sections(page_text, page_num))

        merged   = self._merge_small_sections(raw_sections)
        final    = self._split_large_sections(merged)
        chunks   = [self._build_chunk(i, s) for i, s in enumerate(final)]

        logger.info(f"Produced {len(chunks)} chunks from {self.source_file}")
        return chunks

    # ── Stage 1: page splitting ───────────────────────────────────────────────

    def _split_into_pages(self, text: str) -> list[tuple[int, str]]:
        """Split on <!-- Page N --> markers, return (page_number, text) pairs."""
        parts = PAGE_BREAK.split(text)
        pages = []
        for i, part in enumerate(parts):
            page_num = i + 1
            pages.append((page_num, part.strip()))
        return [(n, t) for n, t in pages if t]

    # ── Stage 2: section splitting ────────────────────────────────────────────

    def _split_page_into_sections(
        self, text: str, page_num: int
    ) -> list[dict]:
        """
        Within a page, split on semantic boundaries.
        Tables are extracted as atomic units first.
        """
        sections = []

        # Extract tables as atomic blocks
        table_pattern = re.compile(r'(<table[\s\S]*?</table>)', re.IGNORECASE)
        last_end = 0
        for match in table_pattern.finditer(text):
            # Text before this table
            before = text[last_end:match.start()].strip()
            if before:
                sections.extend(
                    self._split_text_on_boundaries(before, page_num, "paragraph")
                )
            # Table itself
            table_text = clean_html_table(match.group(1))
            sections.append({
                "text": table_text,
                "page": page_num,
                "type": "table"
            })
            last_end = match.end()

        # Remaining text after last table
        tail = text[last_end:].strip()
        if tail:
            sections.extend(
                self._split_text_on_boundaries(tail, page_num, "paragraph")
            )

        return sections

    def _split_text_on_boundaries(
        self, text: str, page_num: int, content_type: str
    ) -> list[dict]:
        """Split plain text on EDI semantic boundaries."""
        sections = []
        buffer_lines = []

        for line in text.splitlines():
            if is_semantic_boundary(line) and buffer_lines:
                sections.append({
                    "text": "\n".join(buffer_lines).strip(),
                    "page": page_num,
                    "type": content_type
                })
                buffer_lines = []
            buffer_lines.append(line)

        if buffer_lines:
            sections.append({
                "text": "\n".join(buffer_lines).strip(),
                "page": page_num,
                "type": content_type
            })

        return [s for s in sections if s["text"]]

    # ── Stage 3: merge small sections ─────────────────────────────────────────

    def _merge_small_sections(self, sections: list[dict]) -> list[dict]:
        """
        Merge adjacent sections that are under min_chunk_size tokens,
        as long as they share the same page and type.
        """
        if not sections:
            return []

        merged = [sections[0].copy()]

        for current in sections[1:]:
            prev = merged[-1]
            prev_tokens = count_tokens(prev["text"])
            curr_tokens = count_tokens(current["text"])

            can_merge = (
                prev["page"] == current["page"]
                and prev["type"] == current["type"]
                and prev_tokens < self.cfg.min_chunk_size
            )

            if can_merge and (prev_tokens + curr_tokens) <= self.cfg.chunk_size:
                merged[-1] = {
                    "text": prev["text"] + "\n\n" + current["text"],
                    "page": prev["page"],
                    "type": "mixed" if prev["type"] != current["type"] else prev["type"]
                }
            else:
                merged.append(current.copy())

        return merged

    # ── Stage 4: split large sections ─────────────────────────────────────────

    def _split_large_sections(self, sections: list[dict]) -> list[dict]:
        """
        Split any section exceeding chunk_size tokens into overlapping sub-chunks.
        Splits on sentence boundaries to preserve coherence.
        """
        result = []
        for section in sections:
            tokens = count_tokens(section["text"])
            if tokens <= self.cfg.chunk_size:
                result.append(section)
            else:
                result.extend(self._sliding_window_split(section))
        return result

    def _sliding_window_split(self, section: dict) -> list[dict]:
        """Sliding window split that respects sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', section["text"])
        sub_chunks = []
        buffer = []
        buffer_tokens = 0

        for sentence in sentences:
            sent_tokens = count_tokens(sentence)

            if buffer_tokens + sent_tokens > self.cfg.chunk_size and buffer:
                sub_chunks.append({
                    "text": " ".join(buffer),
                    "page": section["page"],
                    "type": section["type"]
                })
                # Keep overlap
                overlap_buffer = []
                overlap_tokens = 0
                for s in reversed(buffer):
                    t = count_tokens(s)
                    if overlap_tokens + t <= self.cfg.chunk_overlap:
                        overlap_buffer.insert(0, s)
                        overlap_tokens += t
                    else:
                        break
                buffer = overlap_buffer
                buffer_tokens = overlap_tokens

            buffer.append(sentence)
            buffer_tokens += sent_tokens

        if buffer:
            sub_chunks.append({
                "text": " ".join(buffer),
                "page": section["page"],
                "type": section["type"]
            })

        return sub_chunks

    # ── Stage 5: build Chunk objects ──────────────────────────────────────────

    def _build_chunk(self, index: int, section: dict) -> Chunk:
        text = section["text"]
        return Chunk(
            chunk_id=f"{self.source_file}__chunk_{index:04d}",
            content=text,
            content_type=section["type"],
            page_number=section["page"],
            section_title=self._infer_section_title(text),
            segment_codes=extract_edi_codes(text),
            version_refs=extract_versions(text),
            token_count=count_tokens(text),
            source_file=self.source_file,
            chunk_index=index,
        )

    def _infer_section_title(self, text: str) -> str:
        """Use first header line or first 80 chars as section title."""
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#"):
                return line.lstrip("#").strip()
            if line:
                return line[:80]
        return "Untitled"


# ── CLI helper ────────────────────────────────────────────────────────────────

def chunk_file(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    chunker = SemanticChunker(source_file=filepath)
    chunks = chunker.chunk(text)
    return [c.to_dict() for c in chunks]


if __name__ == "__main__":
    import sys, pprint
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.md"
    results = chunk_file(path)
    print(f"\n✅ Total chunks: {len(results)}")
    for r in results[:3]:
        pprint.pprint(r)
        print("---")
