from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def summary_as_text(result: dict) -> str:
    summary = result.get("summary", {})
    metrics = result.get("metrics", {})
    lines = [
        "灵影速览 · 视频摘要",
        "=" * 24,
        "",
        "视频摘要",
        str(summary.get("overview", "")).strip(),
        "",
        "关键词",
        "、".join(summary.get("keywords", [])),
        "",
        "章节",
    ]
    for chapter in summary.get("chapters", []):
        lines.extend(
            [
                f"{chapter.get('index', '')}. {chapter.get('title', '')} [{chapter.get('time', '')}]",
                str(chapter.get("summary", "")).strip(),
                "",
            ]
        )
    lines.extend(
        [
            "处理指标",
            f"原视频时长：{metrics.get('duration', 0)} 秒",
            f"摘要时长：{metrics.get('selected_duration', 0)} 秒",
            f"压缩比例：{float(metrics.get('compression_ratio', 0)):.1%}",
            f"处理耗时：{metrics.get('processing_seconds', 0)} 秒",
            "",
        ]
    )
    return "\n".join(lines)


def _paragraph(text: str, style: str | None = None) -> str:
    props = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return (
        "<w:p>"
        + props
        + '<w:r><w:t xml:space="preserve">'
        + escape(text)
        + "</w:t></w:r></w:p>"
    )


def write_docx_summary(result: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    summary = result.get("summary", {})
    metrics = result.get("metrics", {})
    body = [
        _paragraph("灵影速览 · 视频摘要", "Title"),
        _paragraph("视频摘要", "Heading1"),
        _paragraph(str(summary.get("overview", "")).strip()),
        _paragraph("关键词", "Heading1"),
        _paragraph("、".join(summary.get("keywords", []))),
        _paragraph("章节", "Heading1"),
    ]
    for chapter in summary.get("chapters", []):
        body.append(
            _paragraph(
                f"{chapter.get('index', '')}. {chapter.get('title', '')} "
                f"[{chapter.get('time', '')}]",
                "Heading2",
            )
        )
        body.append(_paragraph(str(chapter.get("summary", "")).strip()))
    body.extend(
        [
            _paragraph("处理指标", "Heading1"),
            _paragraph(
                f"原视频 {metrics.get('duration', 0)} 秒；"
                f"摘要 {metrics.get('selected_duration', 0)} 秒；"
                f"处理耗时 {metrics.get('processing_seconds', 0)} 秒。"
            ),
        ]
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(body)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>'
        "</w:sectPr></w:body></w:document>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>'
        '<w:rPr><w:rFonts w:ascii="Aptos" w:eastAsia="Microsoft YaHei"/>'
        '<w:sz w:val="22"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
        '<w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
        '<w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>'
        '<w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>'
        "</w:styles>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    document_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        "</Relationships>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", root_rels)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles_xml)
        docx.writestr("word/_rels/document.xml.rels", document_rels)
    return out_path


def write_summary_exports(result: dict, out_dir: str | Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    txt_path = out_dir / "summary.txt"
    txt_path.write_text(summary_as_text(result), encoding="utf-8")
    docx_path = write_docx_summary(result, out_dir / "summary.docx")
    return {"text": txt_path.name, "docx": docx_path.name}
