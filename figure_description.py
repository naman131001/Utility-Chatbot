import fitz  # PyMuPDF
import base64, os, re, json
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01"
)

VISION_MODEL = "gpt-4o-mini"


# ─── HTML → Markdown table converter ─────────────────────────────────────────

def html_table_to_markdown(html: str) -> str:
    """
    Convert a single HTML <table>...</table> block to a markdown pipe table.

    Handles:
      - <th> header cells  → header row + separator line
      - <td> body cells    → body rows
      - Nested tags stripped from cell content
      - Empty cells preserved
      - Pipe characters in cells escaped
      - Tables with no <th> (row 0 treated as header)
    """
    rows = []
    for row_match in re.finditer(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE):
        row_html = row_match.group(1)
        has_th   = bool(re.search(r'<th', row_html, re.IGNORECASE))
        cells    = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL | re.IGNORECASE)

        cleaned = []
        for cell in cells:
            text = re.sub(r'<[^>]+>', ' ', cell)
            text = re.sub(r'\s+', ' ', text).strip()
            text = text.replace('|', '\\|')
            cleaned.append(text)

        if cleaned:
            rows.append({"cells": cleaned, "is_header": has_th})

    if not rows:
        return ""

    col_count = max(len(r["cells"]) for r in rows)

    for row in rows:
        while len(row["cells"]) < col_count:
            row["cells"].append("")

    header_idx = next((i for i, r in enumerate(rows) if r["is_header"]), 0)

    col_widths = [3] * col_count
    for row in rows:
        for j, cell in enumerate(row["cells"]):
            col_widths[j] = max(col_widths[j], len(cell))

    def format_row(cells):
        padded = [cells[j].ljust(col_widths[j]) for j in range(len(cells))]
        return "| " + " | ".join(padded) + " |"

    def separator_row():
        return "| " + " | ".join("-" * col_widths[j] for j in range(col_count)) + " |"

    lines = []
    for i, row in enumerate(rows):
        lines.append(format_row(row["cells"]))
        if i == header_idx:
            lines.append(separator_row())

    return "\n".join(lines)


def convert_html_tables_to_markdown(content: str) -> str:
    """
    Replace every <table>...</table> block in a string with a markdown pipe table.
    Falls back to original HTML if conversion produces an empty result.
    """
    def replacer(match):
        md = html_table_to_markdown(match.group(0))
        return md if md.strip() else match.group(0)

    return re.sub(
        r'<table[^>]*>.*?</table>',
        replacer,
        content,
        flags=re.DOTALL | re.IGNORECASE
    )


# ─── JSON Adapter ─────────────────────────────────────────────────────────────

def normalize_di_json(di_result_json: dict) -> dict:
    """
    Normalize per-page DI JSON into the flat internal format.

    Also returns a parallel list of per-page character offsets so that
    _split_enriched_content_by_page() can do precise offset-based splitting
    rather than fragile text-landmark searching.

    Per-page (your format):
      { "source_file": ..., "pages": [ { "page_number": N, "content": ..., "pages": [...] } ] }

    Returns flat format PLUS injects "_char_offset" and "_char_length" into
    each page_entry of the original di_result_json so main() can use them.
    """
    all_content_parts = []
    all_di_pages      = []
    cumulative_offset = 0

    raw_entries = []
    for page_entry in di_result_json.get("pages", []):
        raw_entries.append((
            page_entry.get("content", ""),
            page_entry.get("page_number", 1),
            page_entry.get("pages", [])
        ))

    spans_are_absolute = _detect_absolute_spans(raw_entries)
    print(f"  Span mode: {'ABSOLUTE' if spans_are_absolute else 'RELATIVE (shifting)'}")

    # Stamp char offsets onto the original page entries so main() can use them
    for idx, page_entry in enumerate(di_result_json.get("pages", [])):
        page_content = page_entry.get("content", "")
        page_entry["_char_offset"] = cumulative_offset
        page_entry["_char_length"] = len(page_content)
        cumulative_offset += len(page_content) + 1   # +1 for "\n" join separator

    # Reset and build flat structure
    cumulative_offset = 0
    for page_content, page_number, di_pages_raw in raw_entries:
        import copy
        di_pages = copy.deepcopy(di_pages_raw)
        for di_page in di_pages:
            di_page["pageNumber"] = page_number
        if not spans_are_absolute and cumulative_offset > 0:
            di_pages = _shift_spans(di_pages, cumulative_offset)
        all_content_parts.append(page_content)
        all_di_pages.extend(di_pages)
        cumulative_offset += len(page_content) + 1

    return {"content": "\n".join(all_content_parts), "pages": all_di_pages}


def _detect_absolute_spans(raw_entries: list) -> bool:
    if len(raw_entries) < 2:
        return True
    page1_len = len(raw_entries[0][0])
    _, _, di_pages_p2 = raw_entries[1]
    for di_page in di_pages_p2:
        for word in di_page.get("words", []):
            return word["span"]["offset"] > page1_len
    return True


def _shift_spans(di_pages: list, offset: int) -> list:
    for page in di_pages:
        for word in page.get("words", []):
            if "span" in word:
                word["span"]["offset"] += offset
        for line in page.get("lines", []):
            for span in line.get("spans", []):
                span["offset"] += offset
        for mark in page.get("selectionMarks", []):
            if "span" in mark:
                mark["span"]["offset"] += offset
    return di_pages


# ─── Step 1: Find all <figure> blocks ────────────────────────────────────────

def extract_figure_spans(content: str) -> list[dict]:
    figures = []
    pattern = re.compile(r'<figure>(.*?)</figure>', re.DOTALL | re.IGNORECASE)
    for i, match in enumerate(pattern.finditer(content)):
        figures.append({
            "figure_id":  f"fig_{i + 1}",
            "offset":     match.start(),
            "length":     len(match.group(0)),
            "inner_text": match.group(1).strip()
        })
    return figures


# ─── Step 2: Find words inside a figure span ─────────────────────────────────

def get_words_in_span(pages: list, span_start: int, span_end: int) -> list[dict]:
    matched = []
    for page in pages:
        page_number = page.get("pageNumber", page.get("page_number", 1))
        for word in page.get("words", []):
            w_start = word["span"]["offset"]
            w_end   = w_start + word["span"]["length"]
            if w_start >= span_start and w_end <= span_end:
                matched.append({"pageNumber": page_number, "polygon": word["polygon"]})
    return matched


# ─── Step 3: Line-level fallback ─────────────────────────────────────────────

def get_lines_in_span(pages: list, span_start: int, span_end: int) -> list[dict]:
    matched = []
    for page in pages:
        page_number = page.get("pageNumber", page.get("page_number", 1))
        for line in page.get("lines", []):
            for s in line.get("spans", []):
                l_start = s["offset"]
                l_end   = l_start + s["length"]
                if l_start >= span_start and l_end <= span_end:
                    matched.append({"pageNumber": page_number, "polygon": line["polygon"]})
                    break
    return matched


# ─── Step 4: Bounding box from polygons ──────────────────────────────────────

def bounding_box_from_words(words: list[dict]) -> tuple[int, list[float]]:
    if not words:
        return None, None
    page_number = words[0]["pageNumber"]
    all_x, all_y = [], []
    for w in words:
        poly   = w["polygon"]
        all_x += poly[0::2]
        all_y += poly[1::2]
    return page_number, [min(all_x), min(all_y), max(all_x), max(all_y)]


# ─── Step 5: Crop figure from PDF ────────────────────────────────────────────

def crop_figure_from_pdf(pdf_path: str, page_number: int,
                          bbox_inches: list[float], figure_id: str,
                          output_dir: str = "figures",
                          padding_pts: float = 10.0) -> str:
    os.makedirs(output_dir, exist_ok=True)
    doc  = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    x0, y0, x1, y1 = [c * 72 for c in bbox_inches]
    rect = fitz.Rect(x0, y0, x1, y1)
    rect = rect + fitz.Rect(-padding_pts, -padding_pts, padding_pts, padding_pts)
    rect = rect.intersect(page.rect)
    pix  = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=rect)
    safe_id     = re.sub(r'[^\w\-]', '_', figure_id)
    output_path = os.path.join(output_dir, f"figure_{safe_id}.png")
    pix.save(output_path)
    doc.close()
    return output_path


# ─── Step 6: Describe figure with GPT-4o Vision ──────────────────────────────

def describe_figure(image_path: str, inner_text: str = "",
                     surrounding_context: str = "") -> str:
    with open(image_path, "rb") as f:
        b64_image = base64.standard_b64encode(f.read()).decode("utf-8")

    text_hint = ""
    if inner_text:
        text_hint += f"\n\nThe OCR text extracted from inside this figure is:\n{inner_text[:600]}"
    if surrounding_context:
        text_hint += f"\n\nThe surrounding document text (for context) is:\n{surrounding_context[:400]}"

    prompt = (
        "You are analyzing a figure extracted from a utility industry document "
        "(EDI guides, tariffs, operating manuals, regulatory documents).\n\n"
        "First, determine what type of figure this is. It could be:\n"
        "- A table (rows and columns of data)\n"
        "- A chart or graph (axes, trends, values)\n"
        "- A diagram or flowchart (process steps, boxes, arrows)\n"
        "- A scanned form or template (fillable fields, labels)\n"
        "- A logo or purely decorative image (company branding, watermark, divider)\n\n"
        "IMPORTANT: If the figure is ONLY a logo or purely decorative image with no "
        "informational content, respond with exactly the single word: LOGO\n\n"
        "Otherwise, provide a thorough structured description using these markdown sections:\n\n"
        "### Structure:\n"
        "- Describe the visual layout and all entities/components present\n"
        "  (shapes, boxes, ovals, columns, rows, sections, etc.)\n"
        "- Use bullet points for each distinct component\n\n"
        "### Flow of Actions / Data:\n"
        "- For flowcharts: number each step, name the actor, and describe the action "
        "and direction of flow including any arrows or connections\n"
        "- For tables: describe each column header and representative data values row by row\n"
        "- For forms: list every field label, its type (text box, checkbox, signature), "
        "and its purpose\n\n"
        "### Additional Context:\n"
        "- Summarize what this figure communicates overall and how it relates to the "
        "surrounding document content\n\n"
        "Be exhaustive — the description must fully convey the figure's content and "
        f"meaning to someone who cannot see it.{text_hint}"
    )

    response = openai_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64_image}",
                    "detail": "high"
                }}
            ]
        }],
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()


# ─── Step 7: Extract surrounding context ─────────────────────────────────────

def get_surrounding_context(content: str, offset: int, length: int,
                             window: int = 300) -> str:
    start = max(0, offset - window)
    end   = min(len(content), offset + length + window)
    return re.sub(r'<[^>]+>', '', content[start:end]).strip()


# ─── Step 8: Inject AI descriptions ──────────────────────────────────────────

def inject_figure_descriptions(content: str, figures_meta: list) -> str:
    for fig in sorted(figures_meta, key=lambda x: x["offset"], reverse=True):
        # If the AI identified this as a logo/decorative image, leave the
        # original <figure> block completely unchanged — no ID, no description.
        if fig["description"].strip().upper() == "LOGO":
            continue

        replacement = (
            f'<figure id="{fig["figure_id"]}">\n'
            f'<!-- AI Description: {fig["description"]} -->\n'
            f'{fig["inner_text"]}\n'
            f'</figure>'
        )
        content = content[:fig["offset"]] + replacement + content[fig["offset"] + fig["length"]:]
    return content


# ─── Master orchestrator ─────────────────────────────────────────────────────

def process_figures(pdf_path: str, di_result_json: dict,
                    output_dir: str = "figures") -> tuple[str, list]:
    first_page = di_result_json.get("pages", [{}])[0] if di_result_json.get("pages") else {}
    is_per_page_format = (
        "content" not in di_result_json and "source_file" in di_result_json
        or (isinstance(first_page, dict) and "content" in first_page)
    )

    if is_per_page_format:
        print("Detected per-page JSON format. Normalizing...")
        di_result_json = normalize_di_json(di_result_json)
        print(f"Normalized: {len(di_result_json['content'])} chars, "
              f"{len(di_result_json['pages'])} DI page(s).\n")

    content = di_result_json.get("content", "")
    pages   = di_result_json.get("pages", [])

    if not content:
        print("ERROR: 'content' key is empty or missing.")
        return "", []

    figure_spans = []
    # extract_figure_spans(content)
    if not figure_spans:
        print("No <figure> tags found in document content.")
        return content, []

    print(f"Found {len(figure_spans)} <figure> block(s). Processing...\n")
    figures_meta = []

    for fig_span in figure_spans:
        figure_id  = fig_span["figure_id"]
        offset     = fig_span["offset"]
        length     = fig_span["length"]
        inner_text = fig_span["inner_text"]
        span_end   = offset + length

        print(f"  → {figure_id} | offsets [{offset} : {span_end}]")
        print(f"     Preview: {inner_text[:80].replace(chr(10), ' ')}...")

        matched_words = get_words_in_span(pages, offset, span_end)
        if not matched_words:
            print("     No word matches — trying line-level fallback...")
            matched_words = get_lines_in_span(pages, offset, span_end)

        if not matched_words:
            print("     Attempting full-page crop as last resort...")
            page_number = _guess_page_from_nearby_span(pages, offset)
            if page_number and pages:
                page_info = next(
                    (p for p in pages if p.get("pageNumber", p.get("page_number")) == page_number),
                    pages[0]
                )
                bbox = [0, 0, page_info.get("width", 8.5), page_info.get("height", 11.0)]
                print(f"     Full-page fallback on page {page_number}.")
            else:
                print(f"     Cannot determine page. Skipping {figure_id}.")
                continue
        else:
            page_number, bbox = bounding_box_from_words(matched_words)

        print(f"     Page {page_number} | bbox {[round(c, 3) for c in bbox]} in.")

        try:
            image_path = crop_figure_from_pdf(pdf_path, page_number, bbox, figure_id, output_dir)
            print(f"     Saved  → {image_path}")
        except Exception as e:
            print(f"     Crop FAILED: {e}")
            continue

        context = get_surrounding_context(content, offset, length)

        try:
            description = describe_figure(image_path, inner_text=inner_text,
                                           surrounding_context=context)
            print(f"     AI:    {description[:120]}...\n")
        except Exception as e:
            print(f"     Vision FAILED: {e}\n")
            description = "Figure could not be described automatically."

        figures_meta.append({
            "figure_id":   figure_id,
            "page_number": page_number,
            "offset":      offset,
            "length":      length,
            "inner_text":  inner_text,
            "bbox_inches": bbox,
            "image_path":  image_path,
            "description": description
        })

    enriched_content = inject_figure_descriptions(content, figures_meta)
    print(f"Done. {len(figures_meta)} figure(s) described and injected.")
    return enriched_content, figures_meta


def _guess_page_from_nearby_span(pages: list, target_offset: int) -> int | None:
    best_page, best_offset = None, -1
    for page in pages:
        page_number = page.get("pageNumber", page.get("page_number", 1))
        for word in page.get("words", []):
            w_off = word["span"]["offset"]
            if w_off < target_offset and w_off > best_offset:
                best_offset = w_off
                best_page   = page_number
    return best_page


# ─── Page splitter ────────────────────────────────────────────────────────────

def _split_enriched_content_by_page(enriched_content: str, page_entries: list) -> dict:
    """
    Re-map flat enriched_content back to per-page chunks.

    PRIMARY STRATEGY — offset tracking:
      normalize_di_json() stamped "_char_offset" and "_char_length" onto each
      page_entry. These are the exact byte positions of each page's original
      content inside the flat "\n".join(all_pages) string.

      inject_figure_descriptions() only GROWS content (it replaces
      "<figure>...</figure>" with a longer "<figure id=...>...</figure>").
      We track how much each injection grew the string (delta) and shift
      all subsequent page boundaries accordingly.

    FALLBACK — landmark anchoring:
      Used only when _char_offset is missing (e.g. flat-format JSON that
      skipped normalize_di_json).
    """
    # ── Build a list of figure injection deltas ───────────────────────────────
    # Each figure grew the string by: len(replacement) - len(original_tag)
    # The figures_meta offsets are in the PRE-injection flat string, so we
    # apply them in sorted order to compute cumulative growth at any position.
    #
    # We don't have figures_meta here, but we can reconstruct growth by
    # comparing the enriched string to original page boundaries directly
    # using the stamped offsets.

    result  = {}
    n_pages = len(page_entries)

    # Check if normalize_di_json stamped offsets onto entries
    has_offsets = all("_char_offset" in p for p in page_entries)

    if has_offsets:
        # ── Build a map of: original_offset → how much extra chars exist
        #    in enriched_content up to that point due to figure injections.
        #
        #    Strategy: scan all <figure id="..."> blocks in enriched_content,
        #    find the corresponding <figure> block in the original content,
        #    compute delta, accumulate.
        #
        #    Simpler equivalent: for each page, we know its original start
        #    offset and original length. We find the enriched content for
        #    that page by walking through the enriched string and keeping
        #    a running "extra_chars" counter that grows each time we pass
        #    a figure injection point.

        # Reconstruct original flat content to find figure positions
        original_parts   = [p.get("content", "") for p in page_entries]
        original_flat    = "\n".join(original_parts)
        original_figures = list(re.finditer(r'<figure>(.*?)</figure>',
                                            original_flat, re.DOTALL | re.IGNORECASE))
        enriched_figures = list(re.finditer(r'<figure\s+id="[^"]*">(.*?)</figure>',
                                            enriched_content, re.DOTALL | re.IGNORECASE))

        # Build list of (original_offset, delta) for each figure injection
        injection_deltas = []
        for orig_fig, enr_fig in zip(original_figures, enriched_figures):
            delta = len(enr_fig.group(0)) - len(orig_fig.group(0))
            injection_deltas.append((orig_fig.start(), delta))

        # For each page, compute its enriched start offset by adding up
        # all injection deltas that occurred BEFORE the page's original start
        for page_entry in page_entries:
            page_number    = page_entry.get("page_number")
            orig_start     = page_entry["_char_offset"]
            orig_end       = orig_start + page_entry["_char_length"]

            # Sum of all injection deltas with original_offset < orig_start
            extra_before_start = sum(
                d for pos, d in injection_deltas if pos < orig_start
            )
            # Sum of all injection deltas with original_offset inside this page
            extra_inside = sum(
                d for pos, d in injection_deltas if orig_start <= pos < orig_end
            )

            enr_start = orig_start + extra_before_start
            enr_end   = orig_end   + extra_before_start + extra_inside

            result[page_number] = enriched_content[enr_start:enr_end].strip()

    else:
        # ── Fallback: landmark anchoring ──────────────────────────────────────
        # Used when page entries don't have _char_offset (flat format JSON).
        cursor = 0
        for i, page_entry in enumerate(page_entries):
            page_number      = page_entry.get("page_number")
            original_content = page_entry.get("content", "")

            if i == n_pages - 1:
                result[page_number] = enriched_content[cursor:].strip()
            else:
                # Try to anchor on the first non-empty, non-tag text from
                # the next page (skip past any HTML/figure tags at the start)
                next_content = page_entries[i + 1].get("content", "")
                # Strip leading tags/whitespace to find real text to anchor on
                stripped     = re.sub(r'^(\s*<[^>]+>\s*)+', '', next_content).strip()
                landmark     = stripped[:80].strip() if stripped else next_content[:60].strip()

                matched = False
                if landmark:
                    match = re.search(re.escape(landmark), enriched_content[cursor:])
                    if match:
                        split_pos           = cursor + match.start()
                        result[page_number] = enriched_content[cursor:split_pos].strip()
                        cursor              = split_pos
                        matched             = True

                if not matched:
                    fallback_end        = cursor + len(original_content)
                    result[page_number] = enriched_content[cursor:fallback_end].strip()
                    cursor              = fallback_end + 1

    return result


# ─── Main entry point ─────────────────────────────────────────────────────────

def main(pdf_path: str, json_path: str, output_dir: str = "output/document_intelligence/figures",blob_path: str | None = None):
    """
    Full pipeline:
      1. Load DI JSON
      2. process_figures()  → enriched_content with AI figure descriptions
         (normalize_di_json stamps _char_offset/_char_length on each page entry)
      3. _split_enriched_content_by_page() → per-page chunks using precise offsets
      4. convert_html_tables_to_markdown() → <table> tags → pipe tables
      5. Build paginated markdown with header
      6. Write .md file
    """

    with open(json_path, "r", encoding="utf-8") as f:
        di_json = json.load(f)

    enriched_content, figures_data = process_figures(
        pdf_path=pdf_path,
        di_result_json=di_json,
        output_dir=output_dir
    )

    base_name        = os.path.splitext(os.path.basename(json_path))[0]
    out_dir          = os.path.dirname(json_path) or "."
    enriched_md_path = os.path.join(out_dir, f"{base_name}.md")

    AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
    AZURE_CONTAINER       = os.getenv("AZURE_CONTAINER")
    pdf_name              = os.path.basename(pdf_path)
    effective_blob_path = blob_path or pdf_name

    blob_url = (
        f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
        f"/{AZURE_CONTAINER}/{effective_blob_path}"
    )

    total_pages      = di_json.get("total_pages", len(di_json.get("pages", [])))
    total_tables     = sum(len(p.get("tables",     [])) for p in di_json.get("pages", []))
    total_paragraphs = sum(len(p.get("paragraphs", [])) for p in di_json.get("pages", []))

    header = (
        f"# {os.path.splitext(pdf_name)[0]}\n\n"
        f"**Source PDF:** [{pdf_name}]({blob_url})\n\n"
        f"**Total Pages:** {total_pages}  \n"
        f"**Total Tables:** {total_tables}  \n"
        f"**Total Paragraphs:** {total_paragraphs}  \n\n"
        f"---\n\n"
    )

    page_entries   = di_json.get("pages", [])
    enriched_pages = _split_enriched_content_by_page(enriched_content, page_entries)

    paginated_md = ""
    for page_entry in page_entries:
        page_number  = page_entry.get("page_number")
        page_content = enriched_pages.get(page_number, page_entry.get("content", ""))

        # Convert any HTML <table> blocks to markdown pipe tables
        page_content = convert_html_tables_to_markdown(page_content)

        paginated_md += (
            f"---\n\n"
            f"<!-- Page {page_number} -->\n\n"
            f"{page_content}\n\n"
            f"<!-- PageNumber=\"{page_number}\" -->\n\n"
        )

    final_md = header + paginated_md

    with open(enriched_md_path, "w", encoding="utf-8") as f:
        f.write(final_md)

    print(f"\n✓ Enriched markdown   → {enriched_md_path}")
    print(f"✓ HTML tables         → converted to markdown pipe tables")
    print(f"✓ Source PDF link     → {blob_url}")
    print(f"✓ Pages               → {total_pages}")
    print(f"✓ Figures described   → {len(figures_data)}")

    return final_md, figures_data


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main(
        pdf_path="C:/Users/NamanMalik/Desktop/US Ocean Passport/utility-chatbot/NY Info/Con Edison/Con Edison CUBS.pdf",
        json_path="output/document_intelligence/Con Edison CUBS.json"
    )

