"""仅用标准库生成最小可用的 PDF / DOCX 样本（无需 pip）。"""
import zipfile
from pathlib import Path

KNOWLEDGE = Path(__file__).resolve().parent.parent / "data" / "knowledge"

DOCX_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{title}</w:t></w:r></w:p>
    {paragraphs}
  </w:body>
</w:document>"""

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def write_docx(path: Path, title: str, lines: list[str]) -> None:
    paras = "".join(
        f"<w:p><w:r><w:t>{line}</w:t></w:r></w:p>" for line in lines
    )
    doc = DOCX_XML.format(title=title, paragraphs=paras)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", RELS)
        zf.writestr("word/document.xml", doc.encode("utf-8"))


def write_pdf(path: Path, title: str, lines: list[str]) -> None:
    """最小 PDF 1.4（仅 ASCII 行，避免字体依赖）。"""
    content_lines = [f"({title}) Tj T*"]
    y = 750
    for line in lines:
        safe = line.encode("ascii", errors="replace").decode("ascii")[:80]
        content_lines.append(f"50 {y} Td ({safe}) Tj")
        y -= 14
    stream = "\n".join(["BT", "/F1 12 Tf", "50 750 Td"] + content_lines + ["ET"])
    stream_bytes = stream.encode("latin-1", errors="replace")
    objects = []
    objects.append(b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n")
    objects.append(b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n")
    objects.append(
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj\n"
    )
    objects.append(
        f"4 0 obj<< /Length {len(stream_bytes)} >>stream\n".encode()
        + stream_bytes
        + b"\nendstream\nendobj\n"
    )
    objects.append(b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n")
    header = b"%PDF-1.4\n"
    body = b""
    xref_positions = [0]
    offset = len(header)
    for obj in objects:
        xref_positions.append(offset)
        body += obj
        offset += len(obj)
    xref_start = offset
    xref = b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for pos in xref_positions[1:]:
        xref += f"{pos:010d} 00000 n \n".encode()
    trailer = (
        f"trailer<< /Size {len(objects)+1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF"
    ).encode()
    path.write_bytes(header + body + xref + trailer)


def main():
    KNOWLEDGE.mkdir(parents=True, exist_ok=True)
    write_docx(
        KNOWLEDGE / "biology_cell_structure.docx",
        "Cell Structure",
        [
            "Cells are basic units of life.",
            "Mitochondria produce ATP via respiration.",
            "Chloroplasts perform photosynthesis in plants.",
        ],
    )
    write_docx(
        KNOWLEDGE / "english_grammar_tenses.docx",
        "English Tenses",
        [
            "Simple present: habits and facts.",
            "Present perfect: past linked to present.",
            "Passive voice: be + past participle.",
        ],
    )
    write_pdf(
        KNOWLEDGE / "math_quadratic_equations.pdf",
        "Quadratic Equations",
        [
            "ax^2 + bx + c = 0",
            "Discriminant D = b^2 - 4ac",
            "Roots: x = (-b +/- sqrt(D)) / 2a",
        ],
    )
    write_pdf(
        KNOWLEDGE / "physics_newton_laws.pdf",
        "Newton Laws",
        [
            "First law: inertia",
            "Second law: F = ma",
            "Third law: action and reaction",
        ],
    )
    print(f"Created static samples in {KNOWLEDGE}")


if __name__ == "__main__":
    main()
