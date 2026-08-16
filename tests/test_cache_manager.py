import json
from concurrent.futures import ThreadPoolExecutor

from analysis_pipeline import AnalysisPipeline
from cache_manager import CacheManager


def test_cache_manager_returns_cached_report_and_intermediate_data(tmp_path):
    cache_dir = tmp_path / ".vision_cache"
    manager = CacheManager(cache_dir=cache_dir)

    image_path = tmp_path / "evidencia.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    report_path = tmp_path / "evidencia_gpt54-mini.md"
    report_path.write_text("relatorio", encoding="utf-8")

    manager.add_to_cache(
        image_path=image_path,
        model="gpt-5.4-mini",
        analysis_mode="geral",
        ocr_engine="glm-ocr",
        report_path=report_path,
        dimensions=(640, 480),
        ocr_result="OCR em cache",
        yolo_result="- 1x carro",
        quality_result="Nitidez: Nítida",
        exif_data="Sem EXIF",
    )

    is_cached, cached_report = manager.is_cached(image_path, "gpt-5.4-mini", "geral", "glm-ocr")
    cached_intermediate = manager.get_cached_intermediate(image_path)
    stats = manager.get_cache_stats()

    assert is_cached is True
    assert cached_report == str(report_path)
    assert cached_intermediate == {
        "ocr_result": "OCR em cache",
        "ocr_engine": "glm-ocr",
        "yolo_result": "- 1x carro",
        "quality_result": "Nitidez: Nítida",
        "exif_data": "Sem EXIF",
    }
    assert stats["total_entries"] == 1
    assert stats["by_model"]["gpt-5.4-mini"] == 1


def test_cache_manager_invalidates_entry_when_report_is_missing(tmp_path):
    cache_dir = tmp_path / ".vision_cache"
    manager = CacheManager(cache_dir=cache_dir)

    image_path = tmp_path / "evidencia.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    missing_report_path = tmp_path / "ausente.md"

    manager.add_to_cache(
        image_path=image_path,
        model="gpt-5.4-mini",
        analysis_mode="geral",
        ocr_engine="glm-ocr",
        report_path=missing_report_path,
        dimensions=(640, 480),
    )

    is_cached, cached_report = manager.is_cached(image_path, "gpt-5.4-mini", "geral", "glm-ocr")

    assert is_cached is False
    assert cached_report is None
    assert manager.index == {}


def test_cached_artifacts_lookup_handles_dotted_filenames(tmp_path):
    """Regressão: Path.with_suffix cortava no primeiro ponto do nome.

    "IMG_2024.01.15_gpt54-mini.md" virava "IMG_2024.01.pdf", o artefato nunca era
    encontrado e o cache dava miss silencioso — justamente nos nomes de WhatsApp/câmera.
    """
    nomes = [
        "IMG_2024.01.15_gpt54-mini",
        "WhatsApp Image 2024-01-15 at 10.30.45_gpt54-mini",
        "foto.v2_qwen3vl-8b",
        "simples_gpt54-mini",
    ]
    for stem in nomes:
        for fmt in ("md", "pdf"):
            (tmp_path / f"{stem}.{fmt}").write_text("x", encoding="utf-8")

        assert AnalysisPipeline._cached_artifacts_available(
            tmp_path / f"{stem}.md", ["md", "pdf"]
        ), f"cache deveria dar hit para {stem}"

    # E continua retornando False quando um formato realmente falta.
    (tmp_path / "so.md.md").write_text("x", encoding="utf-8")
    assert not AnalysisPipeline._cached_artifacts_available(
        tmp_path / "so.md.md", ["md", "pdf"]
    )


def test_cache_survives_concurrent_writes(tmp_path):
    """Regressão: o lote roda com até 8 workers e corrompia o índice sem lock."""
    manager = CacheManager(cache_dir=tmp_path / ".vision_cache")
    report_path = tmp_path / "relatorio.md"
    report_path.write_text("x", encoding="utf-8")

    imagens = []
    for i in range(40):
        p = tmp_path / f"img{i}.jpg"
        p.write_bytes(f"conteudo-unico-{i}".encode())
        imagens.append(p)

    def gravar(path):
        manager.add_to_cache(path, "qwen3-vl:8b", "forense", "glm-ocr", report_path)
        manager.record_analysis("qwen3-vl:8b", 1.0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(gravar, imagens))

    indice = json.loads((tmp_path / ".vision_cache" / "cache_index.json").read_text(encoding="utf-8"))
    stats = json.loads((tmp_path / ".vision_cache" / "stats_history.json").read_text(encoding="utf-8"))

    assert len(indice) == 40
    assert len(stats) == 40
    assert not list((tmp_path / ".vision_cache").glob("*.tmp"))