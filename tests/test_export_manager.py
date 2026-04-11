import base64
import json

import pytest

from export_manager import ExportManager, PDF_AVAILABLE, ReportData


def _tiny_png_bytes() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )


def _build_report(images=None) -> ReportData:
    return ReportData(
        image_name="evidencia_01",
        image_path="/tmp/evidencia_01.jpg",
        model="gpt-5.4-mini",
        analysis_mode="forense",
        ocr_engine="glm-ocr",
        timestamp="2025-01-01 10:00:00",
        processing_time=1.5,
        dimensions=(1280, 720),
        file_size=4096,
        ocr_result="PLACA ABC1234",
        analysis_content="Analise detalhada do modelo.",
        hash_md5="md5hash",
        hash_sha256="sha256hash",
        image_extension=".jpg",
        yolo_result="- 1x car",
        quality_result="Nitidez: Nítida",
        exif_result="- camera: teste",
        post_processing={
            "summary": "Resumo estruturado",
            "classification": {"document_type": "veiculo"},
        },
        post_processing_markdown="## 🧠 Pós-Processamento Estruturado\n\nResumo estruturado",
        images=images or [],
        preflight_warnings=["YOLO indisponível: a etapa de detecção será pulada."],
        pipeline_telemetry=[
            {
                "stage": "ocr",
                "status": "cached",
                "duration_ms": 0.42,
                "detail": "engine=glm-ocr",
            }
        ],
    )


def test_export_manager_includes_pre_analysis_and_post_processing(tmp_path):
    exporter = ExportManager(tmp_path)
    report = _build_report(images=[("Imagem analisada", _tiny_png_bytes())])

    paths = exporter.export(report, formats=["md", "json"], base_filename="teste_relatorio")

    markdown_content = paths["md"].read_text(encoding="utf-8")
    assert "Dados Extraídos (Pré-Análise)" in markdown_content
    assert "EXIF e GPS" in markdown_content
    assert "YOLO11 (Objetos)" in markdown_content
    assert "Execução da Pipeline" in markdown_content
    assert "Telemetria por Etapa" in markdown_content
    assert "Pós-Processamento Estruturado" in markdown_content
    assert "Resumo estruturado" in markdown_content

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["image"]["extension"] == ".jpg"
    assert payload["analysis"]["pre_analysis"]["yolo_result"] == "- 1x car"
    assert payload["analysis"]["pipeline"]["preflight_warnings"]
    assert payload["analysis"]["pipeline"]["telemetry"][0]["stage"] == "ocr"
    assert payload["analysis"]["post_processing"]["summary"] == "Resumo estruturado"
    assert "images" not in payload["analysis"]


def test_export_manager_embeds_images_in_html(tmp_path):
    exporter = ExportManager(tmp_path)
    report = _build_report(images=[("Imagem analisada", _tiny_png_bytes())])

    paths = exporter.export(report, formats=["html"], base_filename="teste_relatorio_html")

    html_content = paths["html"].read_text(encoding="utf-8")
    assert "data:image/png;base64," in html_content
    assert "Imagem analisada" in html_content
    assert "🖼️ Imagens Analisadas" in html_content
    assert "Execução da Pipeline" in html_content


@pytest.mark.skipif(not PDF_AVAILABLE, reason="reportlab not installed")
def test_export_manager_embeds_images_in_pdf(tmp_path):
    exporter = ExportManager(tmp_path)
    report = _build_report(images=[("Imagem analisada", _tiny_png_bytes())])

    paths = exporter.export(report, formats=["pdf"], base_filename="teste_relatorio_pdf")

    pdf_bytes = paths["pdf"].read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    assert b"/Subtype /Image" in pdf_bytes


@pytest.mark.skipif(not PDF_AVAILABLE, reason="reportlab not installed")
def test_markdown_to_pdf_elements_merges_plaintext_lines_into_single_paragraph(tmp_path):
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph

    exporter = ExportManager(tmp_path)
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("Body", parent=styles["Normal"], alignment=TA_JUSTIFY)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"])
    h3_style = ParagraphStyle("H3", parent=styles["Heading3"])
    bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"], alignment=TA_LEFT)
    code_style = ParagraphStyle("Code", parent=styles["Normal"], alignment=TA_LEFT)

    elements = exporter._markdown_to_pdf_elements(
        "Primeira linha de um parágrafo\nsegunda linha do mesmo parágrafo",
        body_style,
        h2_style,
        h3_style,
        bullet_style,
        code_style,
    )

    paragraphs = [element for element in elements if isinstance(element, Paragraph)]
    assert len(paragraphs) == 1
    assert paragraphs[0].style.alignment == TA_JUSTIFY
