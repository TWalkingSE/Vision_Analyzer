"""Regressão: as análises forenses devem receber a imagem ORIGINAL, não o JPEG reduzido.

O pipeline passava `Image.open(BytesIO(image_data.jpeg_bytes))` — a versão recomprimida
por `prepare_for_api` — para a extração de EXIF e para as análises forenses. Como o
reencode descarta os metadados, GPS/câmera/data saíam sempre vazios e o EXIFAnalyzer
acusava "EXIF ausente" até em originais intactos (falso positivo forense).
"""

from io import BytesIO

import pytest

from PIL import Image

piexif = pytest.importorskip("piexif")

from analysis_pipeline import AnalysisPipeline, ImageProcessor
from exif_analyzer import EXIFAnalyzer


def _make_image_with_exif(path, software=b"Adobe Photoshop 25.0"):
    """Grava um JPEG com câmera, data e GPS (São Paulo) preenchidos."""
    image = Image.new("RGB", (900, 600), (120, 130, 140))
    exif_bytes = piexif.dump({
        "0th": {
            piexif.ImageIFD.Make: b"Canon",
            piexif.ImageIFD.Model: b"EOS R5",
            piexif.ImageIFD.Software: software,
        },
        "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2024:01:15 10:30:00"},
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: b"S",
            piexif.GPSIFD.GPSLatitude: ((23, 1), (33, 1), (0, 1)),
            piexif.GPSIFD.GPSLongitudeRef: b"W",
            piexif.GPSIFD.GPSLongitude: ((46, 1), (38, 1), (0, 1)),
        },
        "1st": {},
        "thumbnail": None,
    })
    image.save(path, "JPEG", exif=exif_bytes)
    return path


def test_prepare_for_api_discards_exif():
    """Documenta a causa raiz: a versão enviada ao LLM não tem metadado algum."""
    buffer = BytesIO()
    _make_image_with_exif(buffer)
    buffer.seek(0)
    original = Image.open(buffer)

    assert original._getexif(), "a imagem de teste precisa ter EXIF"

    _, jpeg_bytes = ImageProcessor.prepare_for_api(original)
    recompressed = Image.open(BytesIO(jpeg_bytes))

    assert not recompressed._getexif()
    assert "exif" not in recompressed.info


def test_prepare_report_extrai_exif_e_gps_da_imagem_original(tmp_path):
    image_path = _make_image_with_exif(tmp_path / "foto.jpg")

    pipeline = AnalysisPipeline(analysis_mode="forense", ocr_engine="none")
    report, error = pipeline.prepare_report(image_path)

    assert error is None
    assert report is not None

    assert "Canon" in report.exif_result
    assert "EOS R5" in report.exif_result
    assert "2024:01:15 10:30:00" in report.exif_result
    assert "GPS_Lat" in report.exif_result

    assert "GPS: -23.550000, -46.633333" in report.quality_result


def test_exif_analyzer_detecta_edicao_em_imagem_intacta(tmp_path):
    """Antes o veredicto era sempre 'EXIF ausente', mascarando adulteração real."""
    image_path = _make_image_with_exif(tmp_path / "editada.jpg")

    pipeline = AnalysisPipeline(analysis_mode="forense", ocr_engine="none")
    report, error = pipeline.prepare_report(image_path)

    assert error is None
    assert "EXIF ausente" not in report.quality_result
    assert "Adobe Photoshop" in report.quality_result


def test_contexto_forense_declara_a_origem_da_imagem(tmp_path):
    """Cadeia de custódia: o relatório diz sobre qual versão cada análise rodou."""
    image_path = _make_image_with_exif(tmp_path / "custodia.jpg")

    pipeline = AnalysisPipeline(analysis_mode="forense", ocr_engine="none")
    report, _ = pipeline.prepare_report(image_path)

    assert "(origem: arquivo original, 900x600)" in report.quality_result


def test_forensic_context_avisa_quando_roda_sobre_versao_recomprimida():
    """Stego LSB sobre JPEG recomprimido não pode ser conclusivo — precisa avisar."""
    pipeline = AnalysisPipeline(analysis_mode="forense", ocr_engine="none")
    image = Image.new("RGB", (300, 200), (90, 90, 90))

    context = pipeline._build_forensic_context(image, is_original=False)

    if "Steganografia LSB" in context:
        assert "não conclusivo" in context
    assert "(origem: JPEG reduzido" in context
