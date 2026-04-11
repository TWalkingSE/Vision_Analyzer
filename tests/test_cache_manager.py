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