import json
from datetime import datetime
from io import BytesIO

from PIL import Image

from analysis_pipeline import AnalysisPipeline, AnalysisResult, ImageAnalysisReport, ImageData, _strip_think_blocks
from prompt_templates import get_prompt_manager


def test_strip_think_blocks_removes_thinking_and_keeps_answer():
    raw = "<think>\nLet me reason about this.\n</think>\n\nThis is the actual analysis."
    assert _strip_think_blocks(raw) == "This is the actual analysis."


def test_strip_think_blocks_returns_empty_when_only_thinking():
    raw = "<think>\nAll thinking, no answer.\n</think>"
    assert _strip_think_blocks(raw) == ""


def test_strip_think_blocks_preserves_normal_text():
    raw = "Normal analysis content without think blocks."
    assert _strip_think_blocks(raw) == raw


def test_screenshot_prompt_is_available_in_prompt_manager():
    template = get_prompt_manager().get_prompt("screenshots")
    assert template is not None
    assert template.name == "🖥️ Análise de Screenshots/Telas"


def test_screenshot_mode_skips_yolo():
    pipeline = AnalysisPipeline(analysis_mode="screenshots", ocr_engine="none", yolo_model="yolo11s")
    assert pipeline._should_run_yolo() is False


def test_screenshot_mode_builds_ela_context(monkeypatch):
    pipeline = AnalysisPipeline(analysis_mode="screenshots", ocr_engine="none", yolo_model="yolo11s")

    class DummyELAResult:
        @staticmethod
        def get_summary():
            return "ELA suspeita"

    class DummyELA:
        def __init__(self, quality, scale):
            self.quality = quality
            self.scale = scale

        @staticmethod
        def analyze(_image):
            return DummyELAResult()

    monkeypatch.setattr("analysis_pipeline.ELA_AVAILABLE", True)
    monkeypatch.setattr("analysis_pipeline.ELAAnalyzer", DummyELA)

    summary = pipeline._build_ela_context(Image.new("RGB", (10, 10), color="white"))

    assert summary == "ELA suspeita"


def test_shared_pipeline_exports_markdown_and_json(monkeypatch, tmp_path):
    image_path = tmp_path / "amostra.jpg"
    image_path.write_bytes(b"fake-image")

    output_dir = tmp_path / "relatorios"
    output_dir.mkdir()

    pipeline = AnalysisPipeline(analysis_mode="geral", ocr_engine="none", yolo_model="yolo11s")

    prepared_report = ImageAnalysisReport(
        image=ImageData(
            path=image_path,
            name="amostra",
            extension=".jpg",
            size_bytes=10,
            dimensions=(640, 480),
            hash_md5="md5hash",
            hash_sha256="sha256hash",
            base64_data="YmFzZTY0",
            jpeg_bytes=b"jpeg-data",
        ),
        analysis_mode="geral",
        ocr_engine="none",
        ocr_result="[OCR desabilitado]",
        yolo_result="Nenhum objeto primário detectado pelo YOLO.",
        quality_result="Nitidez: Nítida",
        exif_result="Metadados ausentes.",
        prompt_used="prompt",
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
    )

    analysis_result = AnalysisResult(
        model_name="gpt-5.4-mini",
        model_type="openai",
        success=True,
        content="Conteúdo analisado.",
        processing_time=2.0,
        post_processing={"summary": "Resumo do pipeline"},
        post_processing_markdown="## 🧠 Pós-Processamento Estruturado\n\nResumo do pipeline",
    )

    monkeypatch.setattr(pipeline, "preflight", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(pipeline, "prepare_report", lambda *_args, **_kwargs: (prepared_report, None))
    monkeypatch.setattr(pipeline, "run_model", lambda *_args, **_kwargs: analysis_result)

    result = pipeline.process_image(
        image_path=image_path,
        selected_models=[("gpt-5.4-mini", "openai")],
        output_dir=output_dir,
        export_formats=["md", "json"],
        use_cache=False,
    )

    assert result["success"] == 1
    assert result["failed"] == 0

    markdown_path = output_dir / "amostra_gpt54-mini.md"
    json_path = output_dir / "amostra_gpt54-mini.json"

    assert markdown_path.exists()
    assert json_path.exists()
    assert markdown_path in result["reports"]

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["analysis"]["content"] == "Conteúdo analisado."
    assert payload["analysis"]["post_processing"]["summary"] == "Resumo do pipeline"


def test_pipeline_preflight_requires_openai_when_selected(tmp_path):
    pipeline = AnalysisPipeline(analysis_mode="geral", ocr_engine="none", yolo_model="yolo11s")
    pipeline.openai_client.client = None

    errors, warnings = pipeline.preflight([("gpt-5.4-mini", "openai")], tmp_path)

    assert any("OPENAI_API_KEY" in error for error in errors)
    assert warnings == [] or isinstance(warnings, list)


def test_prepare_report_reuses_cached_intermediate_and_skips_yolo_for_documents(monkeypatch, tmp_path):
    image_path = tmp_path / "documento.jpg"
    image_path.write_bytes(b"fake-image")

    pipeline = AnalysisPipeline(analysis_mode="documentos", ocr_engine="glm-ocr", yolo_model="yolo11s")
    sample_image = Image.new("RGB", (640, 480), color="white")
    jpeg_buffer = BytesIO()
    sample_image.save(jpeg_buffer, format="JPEG")
    jpeg_bytes = jpeg_buffer.getvalue()

    monkeypatch.setattr(pipeline.image_processor, "load_image", lambda *_args, **_kwargs: sample_image)
    monkeypatch.setattr(
        pipeline.image_processor,
        "process_loaded_image",
        lambda *_args, **_kwargs: ImageData(
            path=image_path,
            name="documento",
            extension=".jpg",
            size_bytes=10,
            dimensions=(640, 480),
            hash_md5="md5hash",
            hash_sha256="sha256hash",
            base64_data="YmFzZTY0",
            jpeg_bytes=jpeg_bytes,
        ),
    )
    monkeypatch.setattr(pipeline.image_processor, "prepare_for_report", lambda *_args, **_kwargs: b"png-bytes")
    monkeypatch.setattr(
        pipeline,
        "_build_quality_and_exif",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("quality/exif should come from cache")),
    )
    monkeypatch.setattr(
        pipeline,
        "_build_yolo_summary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("YOLO should be skipped in documentos mode")),
    )
    monkeypatch.setattr(
        pipeline.ollama_client,
        "extract_ocr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OCR should come from cache")),
    )
    monkeypatch.setattr(pipeline, "_build_prompt", lambda *_args, **_kwargs: "prompt-cache")

    class DummyCacheManager:
        @staticmethod
        def get_cached_intermediate(_image_path):
            return {
                "ocr_result": "OCR em cache",
                "ocr_engine": "glm-ocr",
                "yolo_result": "- 1x documento",
                "quality_result": "Nitidez: Nítida",
                "exif_data": "Metadados ausentes.",
            }

    report, error = pipeline.prepare_report(image_path, cache_manager=DummyCacheManager())

    assert error is None
    assert report is not None
    assert report.ocr_result == "OCR em cache"
    assert report.quality_result == "Nitidez: Nítida"
    assert report.exif_result == "Metadados ausentes."
    assert report.yolo_result == "[YOLO pulado para este modo de análise]"

    telemetry_by_stage = {entry["stage"]: entry for entry in report.pipeline_telemetry}
    assert telemetry_by_stage["cache_lookup"]["status"] == "hit"
    assert telemetry_by_stage["quality_exif"]["status"] == "cached"
    assert telemetry_by_stage["yolo"]["status"] == "skipped"
    assert telemetry_by_stage["ocr"]["status"] == "cached"


def test_process_image_short_circuits_when_all_requested_reports_are_cached(monkeypatch, tmp_path):
    image_path = tmp_path / "cached.jpg"
    image_path.write_bytes(b"fake-image")

    output_dir = tmp_path / "relatorios"
    output_dir.mkdir()
    markdown_path = output_dir / "cached_gpt54-mini.md"
    json_path = output_dir / "cached_gpt54-mini.json"
    markdown_path.write_text("markdown", encoding="utf-8")
    json_path.write_text("{}", encoding="utf-8")

    pipeline = AnalysisPipeline(analysis_mode="geral", ocr_engine="none", yolo_model="yolo11s")
    monkeypatch.setattr(pipeline, "preflight", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(
        pipeline,
        "prepare_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prepare_report should not run for full cache hits")),
    )

    recorded = []

    class DummyCacheManager:
        @staticmethod
        def is_cached(*_args, **_kwargs):
            return True, str(markdown_path)

        @staticmethod
        def record_analysis(model, processing_time, cache_hit=False):
            recorded.append((model, processing_time, cache_hit))

    monkeypatch.setattr("analysis_pipeline.get_cache_manager", lambda: DummyCacheManager())

    result = pipeline.process_image(
        image_path=image_path,
        selected_models=[("gpt-5.4-mini", "openai")],
        output_dir=output_dir,
        export_formats=["md", "json"],
        use_cache=True,
    )

    assert result["success"] == 1
    assert result["failed"] == 0
    assert result["cached_models"] == ["gpt-5.4-mini"]
    assert result["report"] is None
    assert markdown_path in result["reports"]
    assert recorded == [("gpt-5.4-mini", 0.0, True)]