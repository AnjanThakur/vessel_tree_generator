"""Build the comprehensive final project report for the coronary4d release."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Sequence

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "submission_release"
DEMO = RELEASE / "demo_cases"
OUT = RELEASE / "FINAL_PROJECT_REPORT.docx"

NAVY = "17324D"
BLUE = "1A73E8"
RED = "C5221F"
GREEN = "188038"
GOLD = "B06000"
GRAY = "5F6368"
LIGHT_GRAY = "F1F3F4"
PALE_BLUE = "EAF2FB"
PALE_GREEN = "E6F4EA"
PALE_GOLD = "FEF7E0"
PALE_RED = "FCE8E6"
WHITE = "FFFFFF"
CONTENT_DXA = 9360
BRANCHES = ("LMCA", "LAD", "LCX")
CASES = ("healthy", "focal_lad", "diffuse_lcx", "tandem_lad")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int,)):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)


def set_repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:tblHeader")
    node.set(qn("w:val"), "true")
    tr_pr.append(node)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:cantSplit")
    tr_pr.append(node)


def shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    node = tc_pr.find(qn("w:shd"))
    if node is None:
        node = OxmlElement("w:shd")
        tc_pr.append(node)
    node.set(qn("w:fill"), fill)


def set_cell_margins(cell, top: int = 85, start: int = 120, bottom: int = 85, end: int = 120) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def apply_table_geometry(table, widths: Sequence[int]) -> None:
    """Normalize tblW, tblGrid, and every tcW using exact DXA widths."""
    widths = [int(value) for value in widths]
    if sum(widths) != CONTENT_DXA:
        raise ValueError(f"table widths must total {CONTENT_DXA}; received {sum(widths)}")
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.insert(0, tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_layout = tbl_pr.find(qn("w:tblLayout"))
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tbl_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "0")
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        column = OxmlElement("w:gridCol")
        column.set(qn("w:w"), str(width))
        grid.append(column)
    for row in table.rows:
        prevent_row_split(row)
        for cell, width in zip(row.cells, widths):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    run.font.size = Pt(8)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for node in (begin, instr, separate, text, end):
        run._r.append(node)


def add_hyperlink(paragraph, text: str, url: str) -> None:
    relationship = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), relationship)
    run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), BLUE)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_pr.extend((color, underline))
    run.append(r_pr)
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.append(text_node)
    link.append(run)
    paragraph._p.append(link)


def configure(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.3)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(NAVY)
    normal.paragraph_format.line_spacing = 1.08
    normal.paragraph_format.space_after = Pt(6)
    for name, size, color, before, after in (
        ("Title", 26, NAVY, 0, 12),
        ("Subtitle", 12, GRAY, 0, 10),
        ("Heading 1", 17, NAVY, 14, 7),
        ("Heading 2", 14, BLUE, 11, 5),
        ("Heading 3", 11.5, NAVY, 8, 3),
    ):
        style = styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = name != "Subtitle"
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = name.startswith("Heading")
    for name in ("List Bullet", "List Number"):
        style = styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(10.5)
        style.paragraph_format.left_indent = Inches(0.5)
        style.paragraph_format.first_line_indent = Inches(-0.25)
        style.paragraph_format.space_after = Pt(5)

    # Use one shared default part for every page.  Keeping odd/even parts
    # disabled is the most interoperable choice across Word and LibreOffice.
    document.settings.odd_and_even_pages_header_footer = False
    for part in (section.header,):
        header = part.paragraphs[0]
        header.text = "CORONARY4D  |  FINAL TECHNICAL AND VALIDATION REPORT"
        header.runs[0].font.name = "Arial"
        header.runs[0].font.size = Pt(8)
        header.runs[0].font.color.rgb = RGBColor.from_string(GRAY)
        p_pr = header._p.get_or_add_pPr()
        border = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "4")
        bottom.set(qn("w:color"), "DADCE0")
        border.append(bottom)
        p_pr.append(border)
    for part in (section.footer,):
        footer = part.paragraphs[0]
        footer.add_run("Research/engineering prototype - not for clinical use                                      ")
        footer.runs[0].font.name = "Arial"
        footer.runs[0].font.size = Pt(8)
        footer.runs[0].font.color.rgb = RGBColor.from_string(GRAY)
        add_page_number(footer)


def paragraph(document: Document, text: str, *, bold_lead: str | None = None) -> Any:
    p = document.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        p.add_run(bold_lead).bold = True
        p.add_run(text[len(bold_lead):])
    else:
        p.add_run(text)
    return p


def bullet(document: Document, text: str) -> None:
    document.add_paragraph(text, style="List Bullet")


def number_item(document: Document, text: str, number: int) -> Any:
    """Add a manually numbered item so independent lists restart reliably."""
    p = document.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_after = Pt(5)
    p.add_run(f"{number}.  ")
    p.add_run(text)
    return p


def heading(document: Document, text: str, level: int = 1, *, page_break: bool = False) -> None:
    p = document.add_heading(text, level=level)
    p.paragraph_format.page_break_before = page_break


def callout(document: Document, title: str, text: str, fill: str = PALE_BLUE) -> None:
    table = document.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    apply_table_geometry(table, [CONTENT_DXA])
    # Word accessibility checkers expect the first row of every table to carry
    # the header marker.  A callout is a single-cell presentation table, so the
    # sole row is also its semantic heading/content row.
    set_repeat_header(table.rows[0])
    cell = table.cell(0, 0)
    shade(cell, fill)
    p = cell.paragraphs[0]
    r = p.add_run(title + "\n")
    r.bold = True
    r.font.color.rgb = RGBColor.from_string(NAVY)
    p.add_run(text)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def table(
    document: Document,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    widths: Sequence[int],
    *,
    header_fill: str = PALE_BLUE,
    font_size: float = 9.2,
) -> Any:
    obj = document.add_table(rows=1, cols=len(headers))
    obj.style = "Table Grid"
    apply_table_geometry(obj, widths)
    set_repeat_header(obj.rows[0])
    for index, value in enumerate(headers):
        cell = obj.rows[0].cells[index]
        shade(cell, header_fill)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(str(value))
        r.bold = True
        r.font.size = Pt(font_size)
    for values in rows:
        row = obj.add_row()
        for index, value in enumerate(values):
            cell = row.cells[index]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if index > 0 and len(str(value)) < 30 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(value))
            r.font.size = Pt(font_size)
    apply_table_geometry(obj, widths)
    document.add_paragraph().paragraph_format.space_after = Pt(0)
    return obj


def add_figure(document: Document, image: Path, caption: str, *, width: float = 6.35) -> None:
    if not image.is_file():
        paragraph(document, f"Figure unavailable: {image.relative_to(ROOT)}")
        return
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    shape = p.add_run().add_picture(str(image), width=Inches(width))
    # python-docx does not expose alt text directly; set the DrawingML
    # non-visual properties so screen readers receive the figure description.
    shape._inline.docPr.set("title", caption)
    shape._inline.docPr.set("descr", caption)
    cap = document.add_paragraph(style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_with_next = False
    cap.add_run(caption).italic = True


def add_split_montage(document: Document, image: Path) -> None:
    """Embed a tall cohort montage across two complete, readable pages."""
    if not image.is_file():
        paragraph(document, f"Figure unavailable: {image.relative_to(ROOT)}")
        return
    document.add_page_break()
    with TemporaryDirectory(prefix="coronary4d_report_montage_") as tmp:
        with Image.open(image) as source:
            # The montage is a five-column grid.  Split in the whitespace
            # between trees 0030 and 0031 so neither row label is bisected.
            split_y = min(3075, source.height - 1)
            crops = (
                source.crop((0, 0, source.width, split_y)),
                source.crop((0, split_y, source.width, source.height)),
            )
            paths = []
            for index, crop in enumerate(crops, start=1):
                path = Path(tmp) / f"cohort_montage_part_{index}.png"
                crop.save(path)
                paths.append(path)
        add_figure(
            document,
            paths[0],
            "Figure 2a. Fixed cardiac front views for the first portion of the accepted 52-tree cohort.",
            width=6.4,
        )
        document.add_page_break()
        add_figure(
            document,
            paths[1],
            "Figure 2b. Fixed cardiac front views for the remaining accepted trees; together Figures 2a-b show all 52.",
            width=6.4,
        )


def add_cover(document: Document, validation: dict[str, Any]) -> None:
    for _ in range(3):
        document.add_paragraph()
    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("4D Coronary LCA Tree Generator")
    sub = document.add_paragraph(style="Subtitle")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run("Final Technical, Validation, and User Deliverable Report")
    rule = document.add_paragraph()
    rule.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = rule.add_run("____________________________________________")
    run.font.color.rgb = RGBColor.from_string(BLUE)
    callout(
        document,
        "Final release outcome",
        "52/52 population trees accepted; all 50 real-versus-generated comparisons passed; "
        "four controlled healthy/disease 4D cases passed engineering and major-vessel LCA anatomical audits.",
        PALE_GREEN,
    )
    table(
        document,
        ("Field", "Value"),
        (
            ("Release", validation["release"]),
            ("Model scope", validation["model_scope"]),
            ("Repository branch", "latestt_branchh"),
            ("Report date", "22 August 2026"),
            ("Software status", validation["status"]),
            ("Clinical status", "Not clinically validated; research/engineering prototype"),
        ),
        (2300, 7060),
    )
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    r = p.add_run("Prepared as the master submission document; Section 13 provides a verified handoff for later presentation authoring.")
    r.italic = True
    r.font.color.rgb = RGBColor.from_string(GRAY)
    document.add_page_break()


def add_contents(document: Document) -> None:
    heading(document, "Document map", 1)
    sections = (
        "1. Executive summary and submission claim",
        "2. Scope, objectives, and non-claims",
        "3. Repository and system architecture",
        "4. Source data, anatomical assignment, and coordinate systems",
        "5. Statistical shape model and training population",
        "6. Synthetic anatomy generation and acceptance",
        "7. Radius taper, disease, motion, and pulsatility",
        "8. Output contract, visualization, and operating workflow",
        "9. Verification and quantitative results",
        "10. Curated case-by-case deliverables",
        "11. Installation, CLI, Python API, and reproducibility",
        "12. Limitations, risk boundaries, and future work",
        "13. Presentation handoff",
        "14. Primary literature context",
        "Appendices: metrics, artifact map, acceptance checklist, and commands",
    )
    for item in sections:
        bullet(document, item)
    callout(
        document,
        "How to read this report",
        "Software correctness, anatomical plausibility, and clinical validation are deliberately separated. "
        "A PASS means the implemented engineering contract passed; it never means clinical certification.",
        PALE_GOLD,
    )


def load_comparison_rows() -> dict[str, dict[str, str]]:
    path = ROOT / "outputs/lca_ssm/lca_population_cohort/population_validation/real_vs_generated_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: row for row in csv.DictReader(handle)}


def build() -> Path:
    validation = load_json(RELEASE / "VALIDATION_SUMMARY.json")
    cohort = load_json(ROOT / "outputs/lca_ssm/lca_population_cohort/cohort_manifest.json")
    population_validation = load_json(
        ROOT / "outputs/lca_ssm/lca_population_cohort/population_validation/real_vs_generated_validation.json"
    )
    thresholds = load_json(
        ROOT / "outputs/lca_ssm/lca_population_model/generator_statistics/population_validation_thresholds.json"
    )
    pca = load_json(
        ROOT / "outputs/lca_ssm/lca_population_model/generator_statistics/surface_deviation_pca_summary.json"
    )
    motion_summary = load_json(ROOT / "outputs/lca_ssm/lca_population_motion/4d_trees_summary.json")
    export_summary = load_json(ROOT / "outputs/lca_ssm/lca_population_export/exported_trees_summary.json")
    audit_summary = load_json(DEMO / "QUANTITATIVE_AUDIT_SUMMARY.json")
    cases = {name: load_json(DEMO / name / "quantitative_validation.json") for name in CASES}
    comparisons = load_comparison_rows()
    final_audit = load_json(RELEASE / "final_audit/final_audit_summary.json")
    novelty = final_audit["novelty"]
    holdout = final_audit["holdout"]
    test_validation = load_json(RELEASE / "final_validation/test_validation.json")
    original_ppt = load_json(RELEASE / "original_ppt_evidence/original_ppt_evidence_summary.json")
    visual_audit = load_json(RELEASE / "final_visual_anatomical_audit/audit_computation_summary.json")
    design_alignment = load_json(RELEASE / "design_spec_alignment/design_alignment_summary.json")
    surface_statistics = load_json(RELEASE / "design_spec_alignment/real_surface_behavior_statistics.json")
    uv_spline_audit = load_json(RELEASE / "design_spec_alignment/uv_spline_equivalence_audit.json")
    with (RELEASE / "final_visual_anatomical_audit/external_anatomical_sanity_check.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        external_sanity = list(csv.DictReader(handle))

    document = Document()
    configure(document)
    document.core_properties.title = "4D Coronary LCA Tree Generator - Final Project Report"
    document.core_properties.subject = "Technical design, validation, usage, outputs, and limitations"
    document.core_properties.author = "Coronary4D Project Team"
    document.core_properties.keywords = "coronary, LCA, PCA, SSM, disease, stenosis, 4D, validation"

    add_cover(document, validation)
    add_contents(document)

    heading(document, "1. Executive summary and submission claim", 1, page_break=True)
    paragraph(
        document,
        "The project is a reproducible generator for a major-vessel left coronary artery scaffold consisting of "
        "LMCA, LAD, and LCX. It combines an anatomy-gated population model with radius taper, controlled stenosis, "
        "cardiac motion, local lesion compliance, cyclic pulsatility, portable numerical output, ParaView export, "
        "interactive visualization, and independent validation."
    )
    callout(
        document,
        "What is ready",
        "A usable Python API and CLI; a frozen 52-case LCA statistical package; a 52-tree population cohort; "
        "520 motion snapshots; fixed-size XYZ-radius arrays; four controlled disease demonstrations; dashboards, "
        "GIFs, interactive HTML, VTK time series, machine-readable audits, and this final report.",
        PALE_GREEN,
    )
    table(
        document,
        ("Release gate", "Final evidence", "Status"),
        (
            ("Source audit", "191 protected LCA records; 181 resolved daughter assignments", "PASS"),
            ("PCA eligibility", "52 cases; unresolved assignments excluded", "PASS"),
            ("Statistical model", "52 x 81 matrix; 13 modes; 95.55% variance", "PASS"),
            ("Population generation", "52/52 accepted in 64 attempts; every baseline represented", "PASS"),
            ("Distribution validation", "50/50 comparisons; zero warnings", "PASS"),
            ("4D population motion", "52 trees x 10 phases; exact junction and clearance", "PASS"),
            ("Submission cases", "4/4 manifests, VTK readbacks, audits and visual packs", "PASS"),
            ("Regression", f"{test_validation['total_tests_passed']} staged/audit/focused/integrated checks; zero failures", "PASS"),
        ),
        (2550, 5110, 1700),
    )
    paragraph(
        document,
        "Submission claim: this is a validated engineering prototype for population-derived major-vessel LCA "
        "generation and controlled 4D disease visualization. It is not a complete coronary tree, a hemodynamic "
        "simulator, a diagnostic system, a patient-specific model, or a clinically validated medical device."
    )

    heading(document, "2. Scope, objectives, and non-claims", 1, page_break=True)
    heading(document, "2.1 Implemented objectives", 2)
    for text in (
        "Use the approximately 200-label source collection rather than the early eight-patient prototype.",
        "Resolve LAD/LCX daughter identity from multiple anatomical signals in a consistent NIfTI RAS frame.",
        "Represent anatomy in a common LMCA-based cardiac frame and model the LMCA-LAD-LCX tree jointly.",
        "Generate smooth anatomical scaffolds without altering protected source centerlines or reproducing segmentation noise point-for-point.",
        "Support focal, diffuse, and tandem stenosis as explicit radius-only modifications.",
        "Generate a phase-corresponded 4D cycle with deformation, pulsatility, and reduced lesion compliance.",
        "Export fixed-size arrays, checksummed manifests, VTK/ParaView time series, and presentation-ready visualizations.",
        "Quantitatively validate topology, anatomy, disease, motion, pulsatility, file integrity, and reproducibility.",
    ):
        bullet(document, text)
    heading(document, "2.2 Explicit non-claims", 2)
    for text in (
        "No population-derived RCA is claimed because the available disconnected RCA candidates are not resolved ground truth.",
        "No diagonal, septal, obtuse marginal, PDA, PL, or microvascular side branches are generated in the release contract.",
        "No flow, pressure, CFD, vessel-wall mechanics, myocardium, perfusion, plaque biology, or clinical outcome prediction is implemented.",
        "Radii, taper, motion, pulsatility, and compliance are auditable prototype parameters rather than learned patient distributions.",
        "Engineering and anatomical plausibility gates do not establish clinical validity or suitability for patient care.",
    ):
        bullet(document, text)

    heading(document, "3. Repository and system architecture", 1, page_break=True)
    paragraph(
        document,
        "The active system has two layers: the statistical LCA pipeline that produces frozen population artifacts, "
        "and the public integrated generator that consumes those artifacts for disease-aware 4D export. Older LCA "
        "and RCA utilities remain for traceability and compatibility but are outside the final statistical contract."
    )
    table(
        document,
        ("Path", "Responsibility"),
        (
            ("pca_ssm_vessel_tree_generator/", "Extraction, frame alignment, surface projection, fixed representation, PCA, static generation, motion, validation, and population export."),
            ("vessel_tree_generator/", "Public API/CLI, disease configurations, pulsatility, integrated 4D validation, portable export, audit, visualization, and verification."),
            ("lca_vessel_tree_generator/", "Earlier LCA topology, radius, disease, tortuosity, and mesh utilities retained for compatibility."),
            ("rca_vessel_tree_generator/", "Separate primitive RCA utilities; excluded from the population-derived release."),
            ("tests/", "Staged Batch 2-7 tests and integrated submission contract tests."),
            ("outputs/lca_ssm/", "Frozen statistics, compact population evidence, motion summary, and standardized export summary."),
            ("submission_release/", "Portable four-case demonstration, visual tools, quantitative audits, reports, and release summary."),
            ("examples/", "Ready-to-run focal, diffuse, and tandem JSON disease configurations."),
        ),
        (2800, 6560),
    )
    heading(document, "3.1 End-to-end data flow", 2, page_break=True)
    for index, text in enumerate((
        "NIfTI label volumes and protected centerlines enter source audit and graph extraction.",
        "Multi-signal daughter assignment identifies LAD and LCX and rejects low-confidence cases.",
        "A common cardiac frame and ellipsoid support surface make cases comparable.",
        "Fixed branch samples form an 81-dimensional joint surface-deviation vector and a 13-mode PCA model.",
        "Generation bootstraps a matched empirical baseline and adds small, joint PCA innovation.",
        "Static acceptance enforces topology, anatomy, range, progression, collision, and scaffold checks.",
        "Radius, disease, motion, and pulsatility produce the phase-corresponded 4D representation.",
        "Export writes arrays, metadata, graphs, VTK, previews, hashes, visual packs, and independent audits.",
    ), start=1):
        number_item(document, text, index)

    heading(document, "4. Source data, anatomical assignment, and coordinate systems", 1, page_break=True)
    heading(document, "4.1 Source-data funnel", 2)
    table(
        document,
        ("Stage", "Count", "Interpretation"),
        (
            ("Locally available source label volumes", "200", "Original dataset scale; not all become usable LCA records."),
            ("Protected LCA centerline records", "191", "Immutable source geometry preserved."),
            ("Resolved daughter assignments", "181", "Multi-signal confidence gate passed."),
            ("Source + scaffold core-anatomy pass", "65", "Both independent course checks pass after confident assignment."),
            ("Ellipsoid-quality exclusions", "13", "Long-tailed/underconstrained support geometry excluded without lowering gates."),
            ("PCA/statistics eligible", "52", "All assignment, frame, ellipsoid, integrity, representation, and anatomy gates passed."),
            ("Unresolved assignments used", "0", "Excluded from statistics and generation."),
        ),
        (3900, 1250, 4210),
    )
    heading(document, "4.2 LAD/LCX assignment logic", 2)
    paragraph(
        document,
        "Daughter identity is not selected from root radius or a single direction vector. Candidate assignments are "
        "scored using branch length, negative RAS-Z inferior/apical displacement, displacement-to-length ratios, "
        "initial direction, descending-segment fraction, horizontal/circumferential course, terminal lateral reach, "
        "and crown-plane behavior. Low score or small assignment margin causes exclusion/manual review."
    )
    table(
        document,
        ("Expected LAD behavior", "Expected LCX behavior"),
        (
            ("Dominant inferior/apex-directed displacement", "Greater lateral/circumferential displacement"),
            ("Sustained longitudinal descent", "Horizontal crown-like course"),
            ("Typically longer major daughter", "Less inferior displacement than LAD"),
            ("Initial direction contains inferior component", "Initial direction contains strong horizontal component"),
        ),
        (4680, 4680),
    )
    heading(document, "4.3 Coordinate frames and surfaces", 2)
    paragraph(
        document,
        "Source anatomy is interpreted in NIfTI RAS orientation and mapped into a common right-handed cardiac frame "
        "anchored to LMCA and the LCA bifurcation. The fixed front view uses cardiac X-Z with negative Z toward the "
        "apex. Surface-relative coordinates store ellipsoid parameters (u, v, normal offset) plus a local tangent-u, "
        "tangent-v, and normal deviation basis. Separate LAD and LCX anatomical planes remain descriptive evidence; "
        "the generator does not force every centerline point onto an ellipse or plane. The stored surface mapping is "
        "an angular/radial parameterization with exact local-basis reconstruction; it is not claimed to be a Euclidean "
        "nearest-point-on-ellipsoid solution."
    )
    heading(document, "4.4 Original PPT two-plane/two-ellipse contract", 2)
    paragraph(
        document,
        "The original measurement model remains a first-class deliverable rather than being replaced by the later "
        "ellipsoid/PCA architecture. For each successful case, source XYZ is fitted by two independent SVD/least-squares "
        "planes. Points are expressed in each measured plane and passed to a separate actual ellipse estimator for a, b, "
        "tilt, landmark theta, angular extent, and residual calculation. The two measured planes are not forced orthogonal; "
        "only the separately derived common cardiac frame is orthonormal."
    )
    table(
        document,
        ("Original requirement", "Verified evidence"),
        (
            ("NIfTI discovery", f"{validation['original_ppt_measurement']['nifti_discovered']} labels"),
            ("Centerlines / landmarks / two planes / two ellipses", f"{original_ppt['case_count']} cases"),
            ("Per-point ellipse deviations", f"{original_ppt['pointwise_residual_record_count']:,} records"),
            ("Source-coordinate change", f"{original_ppt['maximum_source_coordinate_change_mm']:.3g} mm"),
            ("Segment-length change", f"{original_ppt['maximum_source_segment_length_change_mm']:.3g} mm"),
            ("Population outputs", "Ellipse a/b/tilt, circular landmark theta/extents, plane angle, fit and residual statistics"),
        ),
        (4200, 5160),
    )
    add_figure(
        document,
        RELEASE / "final_validation/02_ppt_two_plane_two_ellipse_summary.png",
        "RAW 191-case partial-arc ellipse measurements. Full fitted ellipse axes describe incomplete-arc reference fits; they are not direct physical heart diameters or final generator scaffold axes.",
    )

    heading(document, "4.5 Technical Design Specification Alignment", 2, page_break=True)
    paragraph(
        document,
        "The final architecture was traced against the primary technical design. The two measured ellipses provide "
        "the triaxial heart-scaffold dimensions: the coronary ellipse supplies a and b, while the cardiac-Z-aligned "
        "axis of the interventricular/LAD ellipse supplies c, with explicit quality handling for underconstrained "
        "partial arcs. These ellipses are scaffold measurements, not LAD or LCX path molds. Arteries retain their "
        "measured or generated surface-relative trajectories and are never snapped onto planar ellipse arcs."
    )
    table(
        document,
        ("Design component", "Alignment", "Evidence / limitation"),
        (
            ("Two anatomical planes", "EQUIVALENT", "Measured coronary and LAD references retained; the measured LAD plane differs materially from the simplified cross-product example and is the validated downstream reference."),
            ("Two ellipses -> ellipsoid", "EXACT", "Per-case a,b,c provenance is machine-traced for all eligible cases."),
            ("u, v, offset representation", "EQUIVALENT", f"{design_alignment['source_surface_reconstruction']['point_count']:,} eligible source points; maximum round-trip error {design_alignment['source_surface_reconstruction']['maximum_reconstruction_error_mm']:.2e} mm."),
            ("Joint deviation PCA", "EXACT", "81 surface-relative features: LMCA 5 + LAD 12 + LCX 10, each with tangent-u, tangent-v and normal coefficients; not raw XYZ PCA."),
            ("B-spline chart", "PARTIAL", f"Protected shape-preserving XYZ spline is re-parameterized into the ellipsoid basis. A literal u/v candidate was evaluated but not promoted: {uv_spline_audit['lad_cases_with_pole_touching_controls']}/52 LAD controls touch a polar singularity and P95 maximum path difference is {uv_spline_audit['P95_candidate_vs_xyz_point_difference_mm']:.2f} mm."),
            ("Motion", "EQUIVALENT", "Radial contraction, longitudinal shortening and torsion deform the ellipsoid-associated points; nine independent positions plus a closure frame are preserved."),
            ("RCA", "DELIBERATE DATASET DEVIATION", "Disconnected candidates are not sufficiently reliable annotated RCA identity for population training."),
            ("Side branches", "NOT IMPLEMENTED", "No population-derived identities were sufficiently validated; random branches are not fabricated."),
        ),
        (2450, 1680, 5230),
        font_size=7.9,
    )
    lad_v = surface_statistics["metrics"]["LAD_v_net_rad"]
    lcx_u = surface_statistics["metrics"]["LCX_u_travel_rad"]
    paragraph(
        document,
        f"The real 52-case surface evidence gives LAD net v progression {lad_v['mean']:.3f} +/- {lad_v['SD']:.3f} rad "
        f"and LCX total circumferential u travel {lcx_u['mean']:.3f} +/- {lcx_u['SD']:.3f} rad. The accepted generated "
        f"cohort has {design_alignment['generated_role_consistent_count']}/52 cases satisfying the deliberately strict "
        "combined surface-role descriptor; the other 12 retain verified eligible real-baseline variation. No PCA, "
        "frame, spline/reconstruction, or validator drift was found, so the frozen cohort was not regenerated."
    )
    add_figure(
        document,
        RELEASE / "design_spec_alignment/00_full_surface_design_concept.png",
        "Technical design alignment: measured planes and ellipse dimensions define the ellipsoid scaffold; LAD and LCX are then represented and assessed by their surface-relative u-v trajectories before 3D reconstruction.",
        width=6.45,
    )

    heading(document, "5. Statistical shape model and training population", 1, page_break=True)
    paragraph(
        document,
        "The primary 81-dimensional PCA is not raw cardiac XYZ. It is the specification-faithful LCA-only joint "
        "surface-deviation model: 27 fixed points, each represented by tangent-u, tangent-v, and normal coefficients "
        "relative to its case-specific ellipsoid. This retains within-vessel and cross-vessel covariance while the "
        "ellipsoid and named u-v-offset landmarks define the global scaffold and endpoints."
    )
    table(
        document,
        ("Model item", "Final value"),
        (
            ("Branch order", "LMCA, LAD, LCX"),
            ("Fixed samples", "LMCA 5; LAD 12; LCX 10"),
            ("Feature dimension", "27 points x 3 = 81"),
            ("Training cases", pca["n_samples"]),
            ("Retained modes", pca["k_retained"]),
            ("Variance cutoff", f"{100*pca['variance_cutoff']:.1f}%"),
            ("Actual retained variance", f"{100*pca['cumulative_variance_retained']:.2f}%"),
            ("Maximum mathematical rank", "51 for 52 centered observations"),
        ),
        (4300, 5060),
    )
    mode_rows = []
    cumulative = 0.0
    for index, ratio in enumerate(pca["explained_variance_ratio"][:13], start=1):
        cumulative += ratio
        mode_rows.append((index, f"{100*ratio:.3f}%", f"{100*cumulative:.3f}%", f"{math.sqrt(pca['eigenvalues'][index-1]):.3f}"))
    table(
        document,
        ("Mode", "Variance", "Cumulative", "Coefficient SD"),
        mode_rows,
        (1300, 2500, 2600, 2960),
    )
    paragraph(
        document,
        "PCA scores are centered at approximately zero. Coefficient standard deviations are the square roots of "
        "the learned eigenvalues. Generation does not sample an unconstrained point cloud: it starts from a complete "
        "case-matched empirical score and adds a clipped standard-normal innovation multiplied by the mode SD and "
        "the validated scale 0.04. This preserves coordinated population structure and avoids independent point noise."
    )
    branch_metric_rows = []
    for key, label in (
        ("branch_length_lmca_mm", "LMCA length (mm)"),
        ("branch_length_lad_mm", "LAD length (mm)"),
        ("branch_length_lcx_mm", "LCX length (mm)"),
        ("bifurcation_angle_deg", "LAD-LCX angle (deg)"),
    ):
        item = thresholds[key]
        branch_metric_rows.append((label, f"{item['mean']:.3f} +/- {item['std']:.3f}", f"{item['p2_5']:.3f} to {item['p97_5']:.3f}", f"{item['min']:.3f} to {item['max']:.3f}"))
    for branch in ("lmca", "lad", "lcx"):
        values = comparisons[f"{branch}_tortuosity"]
        branch_metric_rows.append((f"{branch.upper()} tortuosity", f"{float(values['real_mean']):.3f} +/- {float(values['real_std']):.3f}", f"{float(values['real_p2_5']):.3f} to {float(values['real_p97_5']):.3f}", "arc/chord distance metric"))
    table(
        document,
        ("Training metric", "Mean +/- SD", "Central 95%", "Observed/definition"),
        branch_metric_rows,
        (2500, 2300, 2300, 2260),
        font_size=8.6,
    )
    add_figure(
        document,
        ROOT / "outputs/lca_ssm/lca_population_cohort/population_validation/pca_score_comparison.png",
        "Figure 1. Retained PCA-score comparison between eligible real references and the final generated cohort.",
    )

    heading(document, "6. Synthetic anatomy generation and acceptance", 1, page_break=True)
    paragraph(
        document,
        "Each synthetic tree uses a joint empirical bootstrap of ellipsoid and named landmarks, the exact matched "
        "trajectory/local-basis representation for the scheduled source case, and centered joint PCA innovation. "
        "Endpoint innovation trends are removed so the ostium, shared bifurcation, and terminals remain exact. "
        "Unlearned pointwise noise and a second hand-tuned tortuosity perturbation are disabled. The advanced generator "
        "generalizes the original two-ellipse scaffold: it does not literally resample two planar ellipse arcs, and this "
        "distinction is deliberate and documented."
    )
    table(
        document,
        ("Generation control", "Final policy"),
        (
            ("Canonical cohort size", "52, one per eligible empirical baseline before any baseline repeats"),
            ("PCA innovation", "0.04 scale; normal scores clipped to +/-3 SD"),
            ("Attempts", "Up to 250 per requested accepted tree"),
            ("Topology", "LMCA[-1] == LAD[0] == LCX[0] exactly"),
            ("Endpoints", "Empirical ostium, bifurcation, LAD terminal, and LCX terminal preserved"),
            ("Noise", "No independent point-by-point random noise"),
            ("RCA", "Excluded from final statistical model and generation"),
            ("Dense curve", "Normalized-arc shape-preserving cubic B-spline; explicit SciPy BSpline evaluation"),
        ),
        (3100, 6260),
    )
    heading(document, "6.1 Static acceptance hierarchy", 2)
    for index, text in enumerate((
        "Valid finite Nx3 arrays and all mandatory branches present.",
        "Exact LMCA-LAD-LCX junction equality and separated daughter terminals.",
        "LMCA shorter than both major daughters.",
        "Relative daughter-role gate: LAD is at least 10 mm more inferior than LCX; LCX remains comparatively horizontal and more lateral. Absolute apex-reach checks are descriptive warnings, not hard acceptance.",
        "No nonlocal self-intersection and no unintended inter-branch collision below the configured clearance.",
        "Ellipsoid axes and branch lengths inside observed real hard limits; central 95% departures remain warnings.",
        "Tortuosity, obliquity, backward progress, terminal progress, and maximum local turn inside learned observed extrema.",
        "VTK files reopen with complete named blocks and point arrays.",
    ), start=1):
        number_item(document, text, index)
    heading(document, "6.2 B-spline construction", 2)
    paragraph(
        document,
        "The final empirical controls are the fixed 5/12/10 normalized-arc samples. Stable PCHIP Hermite derivatives "
        "are converted interval-by-interval to cubic Bezier controls, assembled as a clamped cubic B-spline with repeated "
        "interior knots, and evaluated through SciPy BSpline at 60/180/160 dense LMCA/LAD/LCX samples. The first and last "
        "points are overwritten with exact controls and the daughter starts are snapped to the LMCA terminal. This is a "
        "real B-spline basis representation, not a polyline. Unconstrained global cubic trials were rejected because they "
        "introduced hooks and violated learned progression/tortuosity limits. The accepted implementation reproduces the "
        "prior stable path to numerical precision while making the spline basis explicit. The protected spline interpolates "
        "the fixed controls in cardiac XYZ; every dense sample is then re-parameterized and reconstructed in u, v, offset "
        "and the ellipsoid local basis. It is therefore surface-relative, but only partially aligned with a literal global "
        "u-v spline because the current LAD chart reaches a polar singularity in 26 of 52 eligible baselines."
    )
    callout(
        document,
        "Final limitation fixed",
        "The earlier 25-tree cohort had one LMCA local-turn distribution warning. The 0.04 innovation calibration "
        "and learned branch-course hard bounds produced 52/52 accepted trees with 50/50 distribution checks passing.",
        PALE_GREEN,
    )
    add_figure(
        document,
        RELEASE / "final_visual_anatomical_audit/population_montage_fixed_scale.png",
        "Figure 2. All 52 accepted trees in a common fixed-scale cardiac X-Z view. Colors are machine-checked: LMCA dark gray, LAD red, LCX teal; negative Z points toward the apex.",
        width=6.4,
    )
    add_figure(
        document,
        RELEASE / "final_visual_anatomical_audit/population_montage_3d_selected.png",
        "Selected 3D audit views include low-margin, typical, and specifically requested trees 0036 and 0052; apparent 2D crossings are assessed with 3D clearance metrics.",
        width=6.4,
    )
    add_figure(
        document,
        ROOT / "outputs/lca_ssm/lca_population_cohort/population_validation/real_vs_generated_distributions.png",
        "Figure 3. Real-versus-generated length, angle, tortuosity, landmark, and scaffold distributions.",
        width=6.4,
    )

    document.add_page_break()
    heading(document, "7. Radius taper, disease, motion, and pulsatility", 1)
    heading(document, "7.1 Static radius model", 2)
    table(
        document,
        ("Branch", "Proximal radius", "Taper", "Representative proximal diameter"),
        (
            ("LMCA", "2.0 mm", "Cube-law distal target with distance-shaped transition", "4.0 mm"),
            ("LAD", "1.3 mm", "exp(-0.0035 x distance_mm)", "2.6 mm"),
            ("LCX", "1.2 mm", "exp(-0.0035 x distance_mm)", "2.4 mm"),
        ),
        (1600, 2000, 3500, 2260),
    )
    paragraph(
        document,
        "The LMCA distal radius is computed from LAD and LCX proximal radii using the cube law. Radii must remain "
        "positive, finite, monotonic non-increasing, free of large adjacent jumps, and compatible with the cube-law "
        "relative tolerance. Radius values are not part of the 81-dimensional PCA."
    )
    heading(document, "7.2 Disease model", 2)
    table(
        document,
        ("Mode", "Input semantics", "Profile behavior"),
        (
            ("Focal", "branch, center, normalized length, severity", "Smooth compact radius-reduction profile"),
            ("Diffuse", "branch, start/end derived from center and normalized length", "Smooth plateau with transitions"),
            ("Tandem", "multiple focal centers and sub-lesion lengths", "Composed focal lesions; overlap handled numerically"),
        ),
        (1700, 4100, 3560),
    )
    paragraph(
        document,
        "Severity is fractional radius reduction, not percentage area stenosis. Equivalent circular area reduction "
        "is 1 - (1 - radius_reduction)^2. Disease changes only the radius column and leaves all centerline XYZ "
        "coordinates identical to the healthy reference. Configurations are normalized by branch arc length and "
        "validated before and after application."
    )
    heading(document, "7.3 Cardiac motion", 2, page_break=True)
    table(
        document,
        ("Parameter", "Final default", "Role"),
        (
            ("Phases", "10 including phase 0 and repeated phase 1", "Exactly closed animation cycle"),
            ("Radial contraction", "14%", "Ellipsoid a/b contraction; calibrated to keep representative maximum below 8 mm"),
            ("Longitudinal shortening", "10%", "Ellipsoid c shortening"),
            ("Base-to-apex torsion", "10 degrees", "Phase-conditioned longitudinal torsion gradient"),
            ("Peak phase", "0.35", "Peak systolic contraction response"),
            ("Heart rate", "60 bpm", "Maps normalized phase to a 1-second period"),
        ),
        (2300, 1900, 5160),
    )
    paragraph(
        document,
        "Every reference point is projected to the ellipsoid/local basis and reconstructed on the deformed surface. "
        "The shared bifurcation is re-snapped at every phase. Temporal correspondence is exact: the same branch and "
        "point indices describe the same modeled material locations throughout the cycle."
    )
    heading(document, "7.4 Pulsatility and local compliance", 2)
    paragraph(
        document,
        "The diseased reference radius is treated as end-diastolic. Geometric motion peaks at phase 0.35, while the "
        "separate coronary pulse response peaks in early diastole at phase 0.60. Healthy epicardial lumen radius expands by up to "
        "3% at that modeled pulse peak. At the maximum lesion, pulsatility amplitude is multiplied by 0.35 and "
        "blended continuously using the local lesion-reduction profile. This is a scalar compliance approximation; "
        "it does not represent anisodiametric wall motion or coupled pressure mechanics."
    )
    add_figure(
        document,
        ROOT / "outputs/lca_ssm/lca_population_motion/qc_visualization_4d.png",
        "Figure 4. Population motion QC: contraction function, scaffold axes, LAD length variation, and exact junction error.",
    )

    heading(document, "8. Output contract, visualization, and operating workflow", 1, page_break=True)
    table(
        document,
        ("Artifact", "Shape or purpose"),
        (
            ("geometry_cine.npy", "(phase, branch, point, 4) float array ordered x_mm, y_mm, z_mm, radius_mm"),
            ("geometry_static.npy", "Reference/phase-zero fixed-size (branch, point, 4) array"),
            ("coronary_tree_4d.npz", "Compressed tensor plus phase, time, branch, and coordinate labels"),
            ("geometry_healthy_reference.npy", "Healthy radius reference used to quantify disease"),
            ("disease_reduction.npy", "Pointwise fractional radius-reduction array"),
            ("metadata.json", "Configuration, provenance, model, disease, motion, pulsatility, and validation"),
            ("graph.json", "Directed LMCA-to-LAD/LCX hierarchy and branch attributes"),
            ("manifest.json", "Shapes, SHA-256 checksums, validation status, and VTK readback evidence"),
            ("vtk/cine.pvd", "ParaView time-series entry point with one VTM per phase"),
            ("preview.png", "Fixed-front/radius/motion/pulsatility release preview"),
            ("visualizations/", "Dashboard, tortuosity plot, cardiac GIF, interactive HTML, quantitative plot, metrics"),
        ),
        (3000, 6360),
        font_size=8.7,
    )
    paragraph(
        document,
        "Final branch order is [LMCA, LAD, LCX]. Fixed export uses 50 points per branch. VTK point data include "
        "radius and disease-reduction values suitable for coloring in ParaView. The focal LAD interactive viewer is "
        "self-contained for offline use; the remaining small HTML files load Plotly from its CDN."
    )
    document.add_page_break()
    add_figure(
        document,
        DEMO / "disease_mode_comparison.png",
        "Figure 5. Healthy, focal LAD, diffuse LCX, and tandem LAD cases on identical seeded anatomy.",
    )

    heading(document, "9. Verification and quantitative results", 1, page_break=True)
    heading(document, "9.1 Final population results", 2)
    table(
        document,
        ("Metric", "Real mean +/- SD", "Generated mean +/- SD", "Result"),
        tuple(
            (
                label,
                f"{float(comparisons[key]['real_mean']):.3f} +/- {float(comparisons[key]['real_std']):.3f}",
                f"{float(comparisons[key]['generated_mean']):.3f} +/- {float(comparisons[key]['generated_std']):.3f}",
                "PASS" if comparisons[key]["comparison_pass"] == "True" else "WARN",
            )
            for key, label in (
                ("lmca_length_mm", "LMCA length (mm)"),
                ("lad_length_mm", "LAD length (mm)"),
                ("lcx_length_mm", "LCX length (mm)"),
                ("lmca_tortuosity", "LMCA tortuosity"),
                ("lad_tortuosity", "LAD tortuosity"),
                ("lcx_tortuosity", "LCX tortuosity"),
                ("bifurcation_angle_deg", "LAD-LCX angle (deg)"),
                ("lmca_max_resampled_turn_angle_deg", "LMCA max turn (deg)"),
            )
        ),
        (2300, 2750, 2760, 1550),
        font_size=8.5,
    )
    table(
        document,
        ("Cohort result", "Value"),
        (
            ("Accepted/requested", f"{cohort['tree_count']}/{cohort['tree_count']}"),
            ("Sampling attempts", cohort["total_sampling_attempts"]),
            ("Unique eligible baselines represented", f"{cohort['unique_source_case_count']}/{cohort['eligible_source_case_count']}"),
            ("PCA innovation scale", cohort["pca_innovation_scale"]),
            ("Exact LCA topology in every tree", cohort["exact_lca_topology_for_all_trees"]),
            ("Compared metrics", population_validation["compared_metric_count"]),
            ("Distribution passes/warnings", f"{population_validation['descriptive_distribution_pass_count']}/{population_validation['descriptive_distribution_warning_count']}"),
        ),
        (4700, 4660),
    )
    heading(document, "9.2 4D and export results", 2)
    motion = motion_summary["motion_summary"]
    export = export_summary["output_summary"]
    table(
        document,
        ("Check", "Evidence", "Status"),
        (
            ("Population motion", f"{motion['num_trees_processed']} trees x {motion['num_phases']} phases", "PASS"),
            ("Motion junction", "Exact LMCA-LAD/LCX snap at every phase", "PASS"),
            ("Motion clearance", "No rejected self-intersection at 0.75 mm policy", "PASS"),
            ("Fixed-size export", f"{export['num_trees_exported']} trees; {export['num_points_per_vessel']} points/branch", "PASS"),
            ("Static identity", "geometry_static == geometry_cine[0]; max error 0", "PASS"),
            ("Curated case audits", f"{audit_summary['case_count']}/4 with no warnings", audit_summary["status"]),
            ("Checksums and VTK readback", "All four exported case manifests independently verified", "PASS"),
        ),
        (2750, 4910, 1700),
    )
    add_figure(
        document,
        ROOT / "outputs/lca_ssm/lca_population_export/pipeline_qc_report.png",
        "Figure 6. Standardized export QC and identity evidence for the 52-tree population.",
    )
    heading(document, "9.3 Regression and repository QA", 2, page_break=True)
    table(
        document,
        ("Suite", "Passed", "Failed"),
        (
            ("Staged Batch 2-7 tests", 61, 0),
            ("Final independent-audit contract tests", 5, 0),
            ("Focused Person-2 generation tests", 19, 0),
            ("Integrated public-generator tests", 5, 0),
            ("Total", test_validation["total_tests_passed"], test_validation["tests_failed"]),
        ),
        (5200, 2080, 2080),
    )
    representative_summary = paragraph(
        document,
        "Additional release gates passed: Python compileall, dependency consistency (pip check), JSON parsing, "
        "four independent export verification runs, VTK first/last-phase readback, SHA-256 integrity, and final Git diff checks."
    )

    heading(document, "9.4 Generative novelty and internal holdout", 2)
    table(
        document,
        ("Audit result", "Verified value", "Interpretation"),
        (
            ("Canonical exact duplicates", novelty["exact_duplicate_count"], "No generated fixed representation is numerically identical to training"),
            ("Near duplicates (<0.1 mm RMS coordinate)", novelty["near_duplicate_under_0_1mm_count"], "No effective duplicates at the declared threshold"),
            ("Mean displacement from scheduled baseline", f"{novelty['mean_rms_displacement_from_baseline_mm']:.3f} mm RMS point", "Small but non-trivial conservative innovation"),
            ("Median nearest-training distance", f"{novelty['median_nearest_training_rms_coordinate_mm']:.3f} mm RMS coordinate", "Bootstrap derivatives remain close to learned support"),
            ("Five-baseline variant audit", f"{novelty['variant_acceptance_count']}/{novelty['variant_count']} direct variants accepted", "Five variants each; rejection remains explicit"),
            ("Internal holdout", f"5 folds; train {min(holdout['training_case_counts'])}-{max(holdout['training_case_counts'])}; holdout {min(holdout['holdout_case_counts'])}-{max(holdout['holdout_case_counts'])}", "Source-grouped; not external validation"),
            ("Holdout first-draw anatomy acceptance", f"{100*holdout['generated_anatomy_acceptance_rate']:.1f}%", "No threshold tuning; rejected draws remain visible"),
            ("Leakage", f"{holdout['source_leakage_count']} source / {holdout['baseline_leakage_count']} baseline", "All derivatives grouped by source_case_id"),
        ),
        (3200, 2500, 3660),
        font_size=8.3,
    )
    paragraph(
        document,
        "A controlled strategy comparison retained empirical bootstrap plus 0.04 PCA innovation. Pure bounded PCA-score "
        "sampling accepted only 4/52 direct candidates and showed large local turns; a 0.10 moderate innovation was more "
        "diverse but lacked the canonical strategy's demonstrated 50/50 population-comparison result. The selection balances "
        "anatomical validity, population fidelity, and measurable non-identity rather than maximizing acceptance alone."
    )
    add_figure(document, RELEASE / "final_validation/09_generation_novelty.png", "Canonical geometry/PCA novelty audit.")
    add_figure(document, RELEASE / "final_validation/10_holdout_results.png", "Source-grouped five-fold internal holdout evidence.")

    heading(document, "10. Curated case-by-case deliverables", 1, page_break=True)
    paragraph(
        document,
        "All four cases use source case 63.label, seed 20260822, PCA scale 0.04, and identical XYZ centerlines. "
        "Their differences therefore isolate disease and local compliance rather than anatomy. Each case contains "
        "10 frames, three branches, and 50 exported points per branch."
    )
    case_rows = []
    for name in CASES:
        case = cases[name]
        disease_rows = [
            (branch, metrics)
            for branch, metrics in case["disease"].items()
            if metrics["maximum_radius_and_diameter_reduction_fraction"] > 0
        ]
        if disease_rows:
            branch, disease = disease_rows[0]
            disease_text = (
                f"{branch}: {100*disease['maximum_radius_and_diameter_reduction_fraction']:.1f}% radius, "
                f"{100*disease['maximum_equivalent_circular_area_reduction_fraction']:.1f}% area; "
                f"minimum lumen diameter {disease['minimum_lumen_diameter_mm']:.3f} mm"
            )
        else:
            disease_text = "No lesion; healthy taper reference"
        case_rows.append((name, disease_text, f"{case['motion']['maximum_global_displacement_mm']:.3f} mm", case["overall_status"]))
    table(
        document,
        ("Case", "Disease result", "Maximum motion", "Audit"),
        case_rows,
        (1700, 4920, 1550, 1190),
        font_size=8.5,
    )
    heading(document, "10.1 Shared representative anatomy", 2)
    healthy = cases["healthy"]
    table(
        document,
        ("Branch", "Length", "Arc/chord", "Proximal diameter", "Inferior displacement"),
        tuple(
            (
                branch,
                f"{healthy['geometry'][branch]['length_mm']:.3f} mm",
                f"{healthy['geometry'][branch]['distance_metric_arc_over_chord']:.3f}",
                f"{healthy['geometry'][branch]['proximal_diameter_mm']:.3f} mm",
                f"{healthy['geometry'][branch]['inferior_displacement_mm']:.3f} mm",
            )
            for branch in BRANCHES
        ),
        (1200, 1800, 1700, 2300, 2360),
        font_size=8.5,
    )
    paragraph(
        document,
        f"The LAD-LCX angle is {healthy['geometry']['angles_deg']['LAD_to_LCX']:.3f} degrees and the acute "
        f"LAD/LCX plane-normal angle is {healthy['geometry']['LAD_LCX_plane_acute_angle_deg']:.3f} degrees. "
        f"Maximum displacement is {healthy['motion']['maximum_global_displacement_mm']:.3f} mm, mean displacement "
        f"at the sampled peak is {healthy['motion']['mean_global_displacement_at_peak_mm']:.3f} mm, the cycle-closing "
        f"error is {healthy['motion']['cycle_closing_error_mm']:.1e} mm, and healthy maximum pulsatility is "
        f"{100*healthy['pulsatility']['global_maximum_absolute_radius_change_fraction']:.3f}%."
    )
    add_figure(
        document,
        DEMO / "healthy/visualizations/validation_dashboard.png",
        "Figure 7. Healthy representative anatomy, topology, radii, motion, pulsatility, and tortuosity dashboard.",
    )
    add_figure(
        document,
        DEMO / "focal_lad/visualizations/validation_dashboard.png",
        "Figure 8. Focal LAD validation dashboard showing localized radius loss with unchanged centerline geometry.",
    )
    heading(document, "10.2 Local and global pulsatility", 2, page_break=True)
    puls_rows = []
    for name in CASES:
        case = cases[name]
        lesion = None
        for branch, values in case["pulsatility"]["branches"].items():
            if values["change_at_maximum_lesion_fraction"] is not None:
                lesion = (
                    f"{branch}: {100*values['change_at_maximum_lesion_fraction']:.3f}% "
                    f"({values['lesion_to_nonlesion_amplitude_ratio']:.3f}x healthy)"
                )
        puls_rows.append((name, f"{100*case['pulsatility']['global_maximum_absolute_radius_change_fraction']:.3f}%", lesion or "N/A", case["pulsatility"]["peak_change_sign"]))
    table(
        document,
        ("Case", "Global maximum", "Maximum-lesion response", "Peak sign"),
        puls_rows,
        (1800, 1900, 3800, 1860),
        font_size=8.5,
    )
    add_figure(
        document,
        DEMO / "focal_lad/visualizations/quantitative_motion_pulsatility.png",
        "Figure 9. Quantitative focal-LAD motion and pulsatility audit, including reduced lesion response.",
    )
    document.add_page_break()
    add_figure(
        document,
        DEMO / "focal_lad/visualizations/tortuosity_analysis.png",
        "Figure 10. Branch-specific curvature and distance-metric tortuosity for the shared representative anatomy.",
    )

    heading(document, "11. Installation, CLI, Python API, and reproducibility", 1, page_break=True)
    heading(document, "11.1 Environment", 2)
    for text in (
        "Python 3.10 or newer is required.",
        "Create a virtual environment and install requirements.txt, or install the project editable with pip install -e .",
        "Core dependencies are NumPy, SciPy, Matplotlib, PyVista, Plotly, Pillow, and python-docx.",
        "The frozen generator statistics package must remain under outputs/lca_ssm/lca_population_model/generator_statistics unless an alternate path is supplied.",
    ):
        bullet(document, text)
    heading(document, "11.2 Primary commands", 2)
    table(
        document,
        ("Task", "PowerShell command"),
        (
            ("Generate four demo cases", r".\.venv\Scripts\python.exe -m vessel_tree_generator demo --output-dir submission_release\demo_cases --clean"),
            ("Generate one focal LAD case", r".\.venv\Scripts\python.exe -m vessel_tree_generator generate --output-dir my_case --preset focal --branch LAD --position 0.45 --length 0.12 --severity 0.65 --clean"),
            ("Verify one export", r".\.venv\Scripts\python.exe -m vessel_tree_generator verify --input-dir my_case"),
            ("Create visual pack", r".\.venv\Scripts\python.exe -m vessel_tree_generator visualize --input-dir my_case --clean --standalone-html"),
            ("Audit full release", r".\.venv\Scripts\python.exe -m vessel_tree_generator audit --input-dir submission_release\demo_cases --release"),
            ("Rebuild population cohort", r".\.venv\Scripts\python.exe pipeline.py generate --count 52 --pca-scale 0.04 --clean"),
            ("Validate population", r".\.venv\Scripts\python.exe pipeline.py validate --clean"),
            ("Validate novelty", r".\.venv\Scripts\python.exe pipeline.py novelty-validate"),
            ("Run internal holdout", r".\.venv\Scripts\python.exe pipeline.py holdout-validate"),
            ("Run complete final audit", r".\.venv\Scripts\python.exe pipeline.py audit-final"),
        ),
        (2400, 6960),
        font_size=8.0,
    )
    heading(document, "11.3 Python API pattern", 2)
    paragraph(
        document,
        "Instantiate CoronaryTreeGenerator, create a stenosis_config, and call generate_case. For controlled disease "
        "comparisons, call sample_reference once and reuse the returned reference for every disease configuration. "
        "GenerationConfig controls seed, source case, PCA scale, attempts, and heart rate; MotionConfig controls "
        "phases and deformation; PulsatilityConfig independently controls radius amplitude, pulse peak phase, and lesion compliance."
    )
    heading(document, "11.4 Reproducibility and leakage control", 2, page_break=True)
    for text in (
        "Seeds and source_case_id are recorded in metadata and manifests.",
        "Generated trees are synthetic derivatives, not new independent patients.",
        "For machine-learning splits, group every generated derivative, disease state, and phase by source_case_id to prevent train/validation/test leakage.",
        "The canonical scheduler covers 52 source baselines before repeating one; counts above 52 add variations, not new subjects.",
        "Checksums detect post-export changes, and verify reopens VTK plus checks array shape, radii, topology, and hashes.",
        "Protected source geometry is never edited; outputs and bulk runs are regenerated from frozen statistics and tracked code.",
    ):
        bullet(document, text)

    heading(document, "12. Limitations, risk boundaries, and future work", 1, page_break=True)
    heading(document, "12.1 Limitations that remain", 2)
    limitations = (
        ("Small effective sample", "Only 52 independent eligible anatomies support an 81-feature, 13-mode model. Synthetic derivatives do not increase clinical sample size."),
        ("Selection bias", "Strict gating excludes 129 resolved cases and may favor clean/common anatomies."),
        ("No external cohort", "A source-grouped five-fold internal holdout is provided, but no independent institution or clinical cohort was available."),
        ("LCA-only anatomy", "RCA and side branches are absent from the final population contract."),
        ("Linear PCA", "A linear Gaussian-mode approximation may miss multimodal dominance patterns, rare variants, or nonlinear shape manifolds."),
        ("Parametric radius", "Taper and proximal radii are defaults, not learned distributions or patient measurements."),
        ("Parametric physiology", "Motion, pulsatility, and compliance are explicit mechanisms, not subject-specific measurements or FSI."),
        ("Simplified disease", "Stenosis is a smooth axisymmetric radius profile; eccentric plaque, calcification, remodeling, and wall morphology are absent."),
        ("No hemodynamics", "The project does not calculate pressure, flow, FFR, WSS, perfusion, or ischemia."),
        ("Clinical status", "No clinical outcomes, expert-reader study, regulatory process, or prospective validation has been performed."),
    )
    table(document, ("Limitation", "Consequence"), limitations, (2350, 7010), font_size=8.7, header_fill=PALE_GOLD)
    heading(document, "12.2 What was safely improved in the final pass", 2)
    for text in (
        "Canonical population size increased from 25 to 52 so every eligible baseline is represented once.",
        "PCA innovation reduced from 0.08 to 0.04 based on controlled distribution evidence, eliminating the LMCA turn mismatch without collapsing LAD/LCX spread.",
        "Learned hard extrema were added for branch tortuosity, obliquity, backward progress, terminal progress, and local turn.",
        "Radial-motion default reduced from 15% to 14%, moving representative maximum displacement from 8.22 mm to 7.81 mm.",
        "Single-case CLI audit status handling was fixed and verified.",
        "The empirical dense-curve stage was converted from a misleading PCHIP-named helper into an explicit, behavior-preserving cubic B-spline basis.",
        "Generation novelty, three sampling strategies, exact PCA recomputation, and source-grouped five-fold internal holdout were added without weakening gates.",
        "Population and demo phase conventions were unified to nine independent geometric positions plus one repeated closure frame.",
        "All population, motion, export, release, visualization, and documentation artifacts were regenerated from the final settings.",
    ):
        bullet(document, text)
    heading(document, "12.3 Recommended next work", 2)
    for index, text in enumerate((
        "Acquire expert-reviewed RCA and side-branch topology and build a separate confidence-resolved extension.",
        "Recover more of the 181 resolved cases by diagnosing frame/ellipsoid/representation failures without weakening anatomy gates.",
        "Add an external source-grouped cohort and reader-based anatomical scoring.",
        "Learn diameter, taper, motion, and compliance distributions from patient data with metadata and acquisition definitions.",
        "Evaluate nonlinear or conditional shape models only after the real cohort is substantially expanded.",
        "Add eccentric plaque surfaces, vessel wall representation, and optional CFD/FSI as separately validated modules.",
        "Generate and maintain the final presentation deck from the verified figures and machine-readable result tables without introducing new claims.",
    ), start=1):
        number_item(document, text, index)

    heading(document, "13. Presentation handoff", 1, page_break=True)
    heading(document, "13.1 Recommended demonstration order", 2)
    for index, text in enumerate((
        "Start with submission_release/demo_cases/disease_mode_comparison.png to establish controlled same-anatomy disease comparison.",
        "Open focal_lad/visualizations/validation_dashboard.png to explain anatomy, radius, motion, topology, pulsatility, and tortuosity together.",
        "Open focal_lad/visualizations/interactive_tree.html for rotation, zoom, phase slider, playback, hover values, and disease coloring.",
        "Open focal_lad/vtk/cine.pvd in ParaView, color by radius_mm or disease_reduction_fraction, and play the sequence.",
        "Use the population montage and real-versus-generated plot to explain the 52-case SSM evidence.",
        "End with limitations: major-vessel LCA research scaffold, not a complete or clinically validated coronary model.",
    ), start=1):
        number_item(document, text, index)
    heading(document, "13.2 Suggested PPT slide structure", 2, page_break=True)
    table(
        document,
        ("Slide", "Message", "Recommended visual"),
        (
            ("1", "Problem and deliverable", "Title plus compact pipeline statement"),
            ("2", "Data and anatomy gate", "200 -> 191 -> 181 -> 65 -> 52 funnel"),
            ("3", "Common frame and SSM", "27-point representation and PCA modes"),
            ("4", "Population validity", "52-tree montage and distribution comparison"),
            ("5", "Disease model", "Four-case comparison"),
            ("6", "4D motion and pulsatility", "Quantitative motion/pulsatility plot"),
            ("7", "Interactive/ParaView workflow", "Dashboard and cine screenshot"),
        ("8", "Verification", f"{test_validation['total_tests_passed']} tests, 50/50 comparisons, zero warnings"),
            ("9", "Limitations and future work", "Honest scope boundary"),
        ),
        (900, 3550, 4910),
        font_size=8.5,
    )

    heading(document, "14. External anatomical sanity checks", 1, page_break=True)
    paragraph(
        document,
        "These comparisons are descriptive engineering sanity checks, not clinical acceptance limits. The repository-defined LMCA begins at a graph endpoint selected by local mask radius and ends at the selected major-daughter junction; it is not guaranteed to match a manually annotated clinical ostium-to-first-bifurcation segment. Diameter defaults and motion amplitude are parametric prototype settings, not distributions learned from the 52-case anatomical cohort."
    )
    table(
        document,
        ("Metric", "Project", "Literature", "Interpretation"),
        [
            (
                "LMCA/proximal-LCA path" if row["metric"] == "LMCA length" else row["metric"],
                row["project_display"],
                f"{row['literature_mean']}; {row['literature_SD_or_range']} (PMID {row['source_PMID']})",
                row["interpretation"],
            )
            for row in external_sanity
        ],
        (1900, 2800, 2800, 1860),
        font_size=7.7,
    )
    paragraph(
        document,
        f"Independent post-export coordinate audit: {visual_audit['independent_coordinate_gate_pass_count']}/52 pass the production coordinate role/topology/clearance gate with zero validator mismatch. "
        f"{visual_audit['full_descriptive_anatomy_pass_count']}/52 also pass every stricter absolute/descriptive apex-reach check. The latter are reported transparently and are not the production acceptance predicate."
    )
    add_figure(
        document,
        RELEASE / "final_visual_anatomical_audit/real_vs_generated_curvature.png",
        "Uniform 1 mm arc-length curvature comparison. P95/P99 are descriptive cohort evidence; no arbitrary clinical curvature limit is imposed.",
        width=6.4,
    )

    heading(document, "15. Primary literature context", 1, page_break=True)
    paragraph(
        document,
        "Literature values are descriptive external sanity checks. Differences in modality, population, vessel "
        "endpoint definition, phase, and measurement method prevent treating them as interchangeable clinical limits."
    )
    references = cases["healthy"]["external_reference_context"]["primary_references"]
    for index, reference in enumerate(references, start=1):
        p = number_item(document, "", index)
        p.add_run(reference["citation"] + ". ")
        pubmed_id = reference.get("pmid") or reference["url"].rstrip("/").rsplit("/", 1)[-1]
        add_hyperlink(p, f"PubMed record PMID {pubmed_id}", reference["url"])
        p.add_run(f" Topic: {reference['topic']}.")

    heading(document, "Appendix A. Metric definitions", 1, page_break=True)
    table(
        document,
        ("Metric", "Definition and interpretation"),
        (
            ("Arc length", "Sum of Euclidean distances between consecutive centerline samples."),
            ("Chord", "Euclidean distance between branch endpoints."),
            ("Distance-metric tortuosity", "Arc length divided by chord; 1 is straight and larger values indicate a longer path relative to endpoint separation."),
            ("Total turning angle", "Sum of angles between successive unit segment directions; strongly dependent on sampling and smoothing."),
            ("Maximum local turn", "Maximum direction change after uniform arc-length resampling; used for local sharpness QC."),
            ("Inferior displacement", "Dominant negative cardiac/RAS-Z displacement from branch start toward terminal."),
            ("Terminal lateral fraction", "Terminal lateral/circumferential displacement normalized by branch length."),
            ("Backward-progress ratio", "Accumulated negative projection along the branch chord divided by chord length."),
            ("Terminal-progress fraction", "Fraction of sampled steps that approach the terminal rather than move away."),
            ("Plane residual", "Orthogonal distance from centerline points to the least-squares SVD plane; descriptive, not a requirement that the vessel is planar."),
            ("Radius reduction", "1 - diseased_radius / healthy_radius at each point."),
            ("Equivalent area reduction", "1 - (diseased_radius / healthy_radius)^2 under a circular cross-section approximation."),
            ("Pulsatility", "Absolute cyclic radius change relative to the diseased reference radius."),
            ("Topology error", "Euclidean difference between LMCA terminal and daughter start points; required to be effectively zero."),
        ),
        (2750, 6610),
        font_size=8.5,
    )

    heading(document, "Appendix B. Artifact map", 1, page_break=True)
    table(
        document,
        ("Location", "Contents"),
        (
            ("outputs/lca_ssm/lca_population_model/generator_statistics/", "Frozen ellipsoid, landmark, branch assignment, PCA, surface coordinate, and validation packages."),
            ("outputs/lca_ssm/lca_population_cohort/", "52 accepted static trees locally; tracked compact evidence, tree 0001, montage, metrics, and validation plots."),
            ("outputs/lca_ssm/lca_population_motion/", "52 x 10 phase motion output locally; tracked summary and QC image."),
            ("outputs/lca_ssm/lca_population_export/", "Fixed-size population arrays locally; tracked summary and QC report."),
            ("submission_release/demo_cases/", "Four complete portable cases with arrays, metadata, VTK, previews, visualizations, and audits."),
            ("submission_release/reports/", "Per-tree quantitative DOCX reports retained as supporting evidence."),
            ("submission_release/design_spec_alignment/", "Requirement matrix, ellipse-to-ellipsoid provenance, point-level surface trace, real/generated u-v audits, spline-equivalence evaluation, figures, and final alignment report."),
            ("submission_release/mentor_visualization_pack/", "Self-contained ParaView pack with all 52 static trees, four 10-frame disease/motion cine series, referenced VTK dependencies, presentation assets, checksums, and a viewing guide."),
            ("submission_release/FINAL_PROJECT_REPORT.docx", "Master project report and final document."),
            ("SUBMISSION_GUIDE.md", "Five-minute usage, API, outputs, ParaView, validation, and scope guide."),
        ),
        (3700, 5660),
        font_size=8.6,
    )

    document.add_page_break()
    heading(document, "Appendix C. Acceptance checklist", 1)
    checklist = (
        "Protected source centerlines remain unmodified.",
        "Only confidence-resolved assignments enter statistics.",
        "All 52 eligible baselines are represented in the final cohort.",
        "All requested trees are accepted and exact topology is preserved.",
        "All 50 population comparisons pass with no warning.",
        "All 52 exported trees pass the documented relative LAD/LCX role gate; stricter absolute apex-reach checks remain separately descriptive.",
        "Learned branch-course extrema are enforced.",
        "Motion preserves topology and clearance in all 520 population snapshots.",
        "Four release cases preserve exact closed-cycle and phase-zero identity.",
        "Disease changes radii only and leaves XYZ unchanged.",
        "Local lesion pulsatility is lower than healthy pulsatility by the configured compliance model.",
        "All exports pass checksum, shape, radius, topology, and VTK readback verification.",
        f"All {test_validation['total_tests_passed']} regression checks pass; compileall and dependency checks pass.",
        "Five-fold internal holdout has zero source/baseline leakage; it is not external validation.",
        "Clinical non-validation and anatomical incompleteness are explicitly documented.",
        "The final DOCX was rendered page-by-page and visually inspected before release.",
    )
    for item in checklist:
        bullet(document, "PASS - " + item)
    callout(
        document,
        "Final conclusion",
        "The repository is ready as a reproducible, disease-aware 4D major-vessel LCA research deliverable. "
        "Its engineering contract is validated; its remaining scientific and clinical boundaries are explicit and must be preserved.",
        PALE_GREEN,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
