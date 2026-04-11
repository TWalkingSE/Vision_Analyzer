#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline compartilhado entre a UI Streamlit e a CLI."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
except ImportError as exc:
    raise RuntimeError("Pillow é obrigatório para o pipeline compartilhado") from exc

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False

try:
    import rawpy
    import imageio  # noqa: F401

    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from prompt_templates import get_prompt_manager

    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    get_prompt_manager = None

try:
    from image_preprocessor import quick_analyze

    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    quick_analyze = None

try:
    from object_detector import get_detector, is_yolo_available

    YOLO_AVAILABLE = is_yolo_available()
except ImportError:
    YOLO_AVAILABLE = False
    get_detector = None

try:
    from ela_analyzer import ELAAnalyzer

    ELA_AVAILABLE = True
except ImportError:
    ELA_AVAILABLE = False
    ELAAnalyzer = None

try:
    from post_processor import PostProcessor

    POST_PROCESSOR_AVAILABLE = True
except ImportError:
    POST_PROCESSOR_AVAILABLE = False
    PostProcessor = None

try:
    from export_manager import ExportManager, ReportData

    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False
    ExportManager = None
    ReportData = None

try:
    from cache_manager import get_cache_manager

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    get_cache_manager = None

try:
    from api_utils import (
        API_RETRY_CONFIG,
        InputValidator,
        ValidationConfig,
        get_ollama_limiter,
        get_openai_limiter,
        retry_with_backoff,
    )

    API_UTILS_AVAILABLE = True
except ImportError:
    API_UTILS_AVAILABLE = False
    API_RETRY_CONFIG = None
    InputValidator = None
    ValidationConfig = None
    get_ollama_limiter = None
    get_openai_limiter = None
    retry_with_backoff = None

from runtime_config import (
    HEIF_EXTENSIONS,
    JPEG_QUALITY,
    MAX_IMAGE_SIZE,
    OCR_MODEL,
    OCR_MODEL_ALT,
    OPENAI_MODEL,
    RAW_EXTENSIONS,
    REPORT_IMAGE_SIZE,
    SUPPORTED_EXTENSIONS,
    get_model_short_name,
    sanitize_filename,
)

import re as _re


def _strip_think_blocks(text: str) -> str:
    """Remove blocos <think>...</think> gerados por modelos com raciocínio interno."""
    return _re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


DEFAULT_FALLBACK_PROMPT = """# SYSTEM ROLE
Atue como um analista sênior de imagens.

## CONTEXTO OCR
{ocr_result}

## CONTEXTO YOLO
{yolo_result}

## CONTEXTO DE QUALIDADE
{quality_result}

## CONTEXTO EXIF
{exif_data}

Forneça uma análise objetiva, estruturada e indique incertezas quando necessário.
"""


def _run_with_retry(func):
    """Executa uma função com retry quando os utilitários de API estiverem disponíveis."""
    if API_UTILS_AVAILABLE and retry_with_backoff and API_RETRY_CONFIG:
        return retry_with_backoff(API_RETRY_CONFIG)(func)()
    return func()


@dataclass
class ImageData:
    """Dados de uma imagem preparada para inferência."""

    path: Path
    name: str
    extension: str
    size_bytes: int
    dimensions: Tuple[int, int] = (0, 0)
    hash_md5: str = ""
    hash_sha256: str = ""
    base64_data: str = ""
    jpeg_bytes: bytes = b""

    def __post_init__(self):
        self.extension = self.extension.lower()


@dataclass
class AnalysisResult:
    """Resultado de uma inferência para um único modelo."""

    model_name: str
    success: bool
    model_type: str = ""
    content: str = ""
    error: str = ""
    processing_time: float = 0.0
    tokens_used: int = 0
    cached: bool = False
    post_processing: Dict[str, Any] = field(default_factory=dict)
    post_processing_markdown: str = ""


@dataclass
class ImageAnalysisReport:
    """Relatório agregado de uma imagem para todos os modelos."""

    image: ImageData
    analysis_mode: str = "geral"
    ocr_engine: str = "glm-ocr"
    ocr_result: str = ""
    yolo_result: str = ""
    quality_result: str = ""
    exif_result: str = ""
    prompt_used: str = ""
    analyses: List[AnalysisResult] = field(default_factory=list)
    images: list[tuple[str, bytes]] = field(default_factory=list)
    preflight_warnings: List[str] = field(default_factory=list)
    pipeline_telemetry: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ImageProcessor:
    """Carregamento e preparo de imagens com suporte aos formatos do projeto."""

    @staticmethod
    def find_images(input_dir: Path):
        if not input_dir.exists():
            return

        for file_path in input_dir.rglob("*"):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield file_path

    @staticmethod
    def load_image(path: Path) -> Optional[Image.Image]:
        ext = path.suffix.lower()

        try:
            if ext in RAW_EXTENSIONS:
                if not RAW_SUPPORT:
                    logger.warning("Suporte RAW indisponível para %s", path.name)
                    return None
                with rawpy.imread(str(path)) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        half_size=False,
                        no_auto_bright=False,
                        output_bps=8,
                    )
                return Image.fromarray(rgb)

            if ext in HEIF_EXTENSIONS:
                if not HEIF_SUPPORT:
                    logger.warning("Suporte HEIF indisponível para %s", path.name)
                    return None
                return Image.open(path)

            return Image.open(path)
        except Exception as exc:
            logger.error("Erro ao carregar %s: %s", path.name, exc)
            return None

    @staticmethod
    def prepare_for_api(image: Image.Image) -> Tuple[str, bytes]:
        if image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
            image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        jpeg_bytes = buffer.getvalue()
        base64_data = base64.b64encode(jpeg_bytes).decode("utf-8")
        return base64_data, jpeg_bytes

    @staticmethod
    def prepare_for_report(image: Image.Image) -> bytes:
        """Prepara uma versão PNG da imagem para relatórios HTML/PDF."""
        report_image = image.copy()

        if report_image.mode == "P":
            if "transparency" in report_image.info:
                report_image = report_image.convert("RGBA")
            else:
                report_image = report_image.convert("RGB")
        elif report_image.mode == "LA":
            report_image = report_image.convert("RGBA")
        elif report_image.mode not in ("RGB", "RGBA"):
            report_image = report_image.convert("RGB")

        if report_image.size[0] > REPORT_IMAGE_SIZE[0] or report_image.size[1] > REPORT_IMAGE_SIZE[1]:
            report_image.thumbnail(REPORT_IMAGE_SIZE, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        report_image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    @staticmethod
    def calculate_md5(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    @staticmethod
    def calculate_sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def process_loaded_image(cls, image: Image.Image, path: Path) -> ImageData:
        base64_data, jpeg_bytes = cls.prepare_for_api(image)
        return ImageData(
            path=path,
            name=path.stem,
            extension=path.suffix,
            size_bytes=path.stat().st_size,
            dimensions=image.size,
            hash_md5=cls.calculate_md5(jpeg_bytes),
            hash_sha256=cls.calculate_sha256(jpeg_bytes),
            base64_data=base64_data,
            jpeg_bytes=jpeg_bytes,
        )

    @classmethod
    def process_image(cls, path: Path) -> Optional[ImageData]:
        image = cls.load_image(path)
        if image is None:
            return None
        return cls.process_loaded_image(image, path)


class OpenAIClient:
    """Cliente compartilhado para OpenAI."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key and OPENAI_AVAILABLE else None

    def is_available(self) -> bool:
        return self.client is not None

    def analyze_image(self, image_data: ImageData, system_prompt: str, model: str = OPENAI_MODEL) -> AnalysisResult:
        if not self.is_available():
            return AnalysisResult(model_name=model, model_type="openai", success=False, error="Cliente OpenAI não disponível")

        if API_UTILS_AVAILABLE and get_openai_limiter:
            get_openai_limiter().wait()

        start_time = time.perf_counter()

        try:
            def request():
                return self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analise esta imagem seguindo as instruções do sistema."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data.base64_data}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=4096,
                    temperature=0.1,
                )

            response = _run_with_retry(request)
            elapsed = time.perf_counter() - start_time

            raw_content = response.choices[0].message.content or ""
            cleaned_content = _strip_think_blocks(raw_content)

            if not cleaned_content:
                return AnalysisResult(
                    model_name=model,
                    model_type="openai",
                    success=False,
                    error="Modelo retornou resposta vazia",
                    processing_time=elapsed,
                    tokens_used=response.usage.total_tokens if response.usage else 0,
                )

            return AnalysisResult(
                model_name=model,
                model_type="openai",
                success=True,
                content=cleaned_content,
                processing_time=elapsed,
                tokens_used=response.usage.total_tokens if response.usage else 0,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            return AnalysisResult(
                model_name=model,
                model_type="openai",
                success=False,
                error=str(exc),
                processing_time=elapsed,
            )


class OllamaClient:
    """Cliente compartilhado para modelos locais e OCR."""

    def __init__(self):
        self.available = OLLAMA_AVAILABLE

    def is_available(self) -> bool:
        return self.available

    def unload_model(self, model: str) -> None:
        """Descarrega um modelo da VRAM enviando keep_alive=0."""
        if not self.available:
            return
        try:
            ollama.chat(model=model, messages=[], keep_alive=0)
            logger.info("♻️ Modelo %s descarregado da VRAM", model)
        except Exception as exc:
            logger.debug("Falha ao descarregar modelo %s: %s", model, exc)

    def unload_models(self, models: List[str]) -> None:
        """Descarrega uma lista de modelos da VRAM."""
        for model in models:
            self.unload_model(model)

    def _chat(self, **kwargs):
        if API_UTILS_AVAILABLE and get_ollama_limiter:
            get_ollama_limiter().wait()
        return _run_with_retry(lambda: ollama.chat(**kwargs))

    def extract_ocr_glm(self, image_data: ImageData) -> str:
        if not self.is_available():
            return "[GLM OCR não disponível - Ollama offline]"

        try:
            response = self._chat(
                model=OCR_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": "Extract all visible text from this image. Return only the raw text, nothing else.",
                        "images": [image_data.base64_data],
                    }
                ],
                options={"temperature": 0.1, "num_predict": 1024, "num_ctx": 4096},
            )
            return response["message"]["content"].strip() or "[Nenhum texto detectado]"
        except Exception as exc:
            return f"[Erro GLM OCR: {exc}]"

    def extract_ocr_olmocr2(self, image_data: ImageData) -> str:
        if not self.is_available():
            return "[OLMoOCR2 não disponível - Ollama offline]"

        try:
            response = self._chat(
                model=OCR_MODEL_ALT,
                messages=[
                    {
                        "role": "user",
                        "content": "Extract all visible text from this image. Return only the raw text, nothing else.",
                        "images": [image_data.base64_data],
                    }
                ],
                options={"temperature": 0.1, "num_predict": 1024, "num_ctx": 4096},
            )
            return response["message"]["content"].strip() or "[Nenhum texto detectado]"
        except Exception as exc:
            return f"[Erro OLMoOCR2: {exc}]"

    def extract_ocr(self, image_data: ImageData, engine: str = "glm-ocr") -> str:
        if engine == "none":
            return "[OCR desabilitado]"
        if engine == "olmocr2":
            return self.extract_ocr_olmocr2(image_data)
        return self.extract_ocr_glm(image_data)

    def analyze_image(self, image_data: ImageData, system_prompt: str, model: str) -> AnalysisResult:
        if not self.is_available():
            return AnalysisResult(model_name=model, model_type="ollama", success=False, error="Ollama não disponível")

        start_time = time.perf_counter()

        try:
            response = self._chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "Analise esta imagem seguindo as instruções do sistema.",
                        "images": [image_data.base64_data],
                    },
                ],
                options={"temperature": 0.3, "num_predict": 8192, "num_ctx": 8192},
            )
            elapsed = time.perf_counter() - start_time

            raw_content = response["message"]["content"] or ""
            cleaned_content = _strip_think_blocks(raw_content)

            if not cleaned_content:
                return AnalysisResult(
                    model_name=model,
                    model_type="ollama",
                    success=False,
                    error="Modelo retornou resposta vazia (possível truncamento do bloco de raciocínio)",
                    processing_time=elapsed,
                )

            return AnalysisResult(
                model_name=model,
                model_type="ollama",
                success=True,
                content=cleaned_content,
                processing_time=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            return AnalysisResult(
                model_name=model,
                model_type="ollama",
                success=False,
                error=str(exc),
                processing_time=elapsed,
            )


def build_report_data(report: ImageAnalysisReport, analysis: AnalysisResult) -> ReportData:
    """Converte o relatório compartilhado no payload usado por todos os exports."""
    if ReportData is None:
        raise RuntimeError("ExportManager não disponível")

    return ReportData(
        image_name=report.image.name,
        image_path=str(report.image.path),
        model=analysis.model_name,
        analysis_mode=report.analysis_mode,
        ocr_engine=report.ocr_engine,
        timestamp=report.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        processing_time=analysis.processing_time,
        dimensions=report.image.dimensions,
        file_size=report.image.size_bytes,
        ocr_result=report.ocr_result,
        analysis_content=analysis.content,
        hash_md5=report.image.hash_md5,
        hash_sha256=report.image.hash_sha256,
        image_extension=report.image.extension,
        yolo_result=report.yolo_result,
        quality_result=report.quality_result,
        exif_result=report.exif_result,
        post_processing=analysis.post_processing,
        post_processing_markdown=analysis.post_processing_markdown,
        images=list(report.images),
        preflight_warnings=list(report.preflight_warnings),
        pipeline_telemetry=list(report.pipeline_telemetry),
    )


class AnalysisPipeline:
    """Pipeline compartilhado de preparação, inferência, pós-processamento e exportação."""

    def __init__(self, analysis_mode: str = "geral", ocr_engine: str = "glm-ocr", yolo_model: str = "yolo11s"):
        self.analysis_mode = analysis_mode
        self.ocr_engine = ocr_engine
        self.yolo_model = yolo_model
        self.image_processor = ImageProcessor()
        self.openai_client = OpenAIClient()
        self.ollama_client = OllamaClient()
        self.post_processor = PostProcessor() if POST_PROCESSOR_AVAILABLE else None
        self.validator = InputValidator(ValidationConfig()) if API_UTILS_AVAILABLE and InputValidator else None

    def _build_prompt(self, ocr_result: str, yolo_result: str, quality_result: str, exif_data: str) -> str:
        if PROMPTS_AVAILABLE and get_prompt_manager:
            try:
                manager = get_prompt_manager()
                template = manager.get_prompt(self.analysis_mode)
                if template:
                    return template.format_prompt(
                        ocr_result=ocr_result,
                        yolo_result=yolo_result,
                        quality_result=quality_result,
                        exif_data=exif_data,
                    )
            except Exception as exc:
                logger.debug("Falha ao resolver prompt compartilhado: %s", exc)

        return DEFAULT_FALLBACK_PROMPT.format(
            ocr_result=ocr_result,
            yolo_result=yolo_result,
            quality_result=quality_result,
            exif_data=exif_data,
        )

    def _validate_file(self, image_path: Path) -> Optional[str]:
        if not self.validator:
            return None

        is_valid, message = self.validator.validate_file(image_path)
        return None if is_valid else message

    def _validate_image(self, image: Image.Image) -> Optional[str]:
        if not self.validator:
            return None

        is_valid, message = self.validator.validate_image(image)
        return None if is_valid else message

    def _build_quality_and_exif(self, image: Image.Image) -> Tuple[str, str]:
        quality_str = "[Qualidade não avaliada]"
        exif_str = "[EXIF não avaliado]"

        if not PREPROCESSOR_AVAILABLE or quick_analyze is None:
            return quality_str, exif_str

        quality_info = quick_analyze(image)
        if not quality_info:
            return quality_str, exif_str

        blur_txt = "Borrada" if quality_info.get("is_blurry") else "Nítida"
        brightness = quality_info.get("brightness", 0)
        brightness_txt = "Escura" if brightness < 0.3 else ("Clara" if brightness > 0.7 else "Equilibrada")
        recommendations = ", ".join(quality_info.get("recommendations", []))
        quality_str = (
            f"Nitidez: {blur_txt}\n"
            f"Iluminação: {brightness_txt}\n"
            f"Avisos: {recommendations if recommendations else 'Nenhum'}"
        )

        exif_dict = quality_info.get("exif_data", {})
        if exif_dict:
            exif_str = "\n".join(f"- {key}: {value}" for key, value in exif_dict.items())
        else:
            exif_str = "Metadados de câmera ou GPS ausentes."

        return quality_str, exif_str

    def _build_yolo_summary(self, image: Image.Image) -> str:
        if not YOLO_AVAILABLE or get_detector is None:
            return "[YOLO não executado]"

        detector = get_detector(self.yolo_model)
        if detector is None:
            return "[YOLO não executado]"

        try:
            yolo_result = detector.detect(image)
        except Exception as exc:
            logger.warning("Falha no YOLO para %s: %s", self.yolo_model, exc)
            return "[YOLO não executado]"

        if yolo_result.total_objects == 0:
            return "Nenhum objeto primário detectado pelo YOLO."

        summary = yolo_result.get_summary()
        lines = ["Objetos detectados pela máquina:"]
        for object_class, count in summary.items():
            lines.append(f"- {count}x {object_class}")
        return "\n".join(lines)

    def _build_ela_context(self, image: Image.Image) -> str:
        if self.analysis_mode not in {"forense", "screenshots"} or not ELA_AVAILABLE or ELAAnalyzer is None:
            return ""

        try:
            ela = ELAAnalyzer(quality=95, scale=15)
            return ela.analyze(image).get_summary()
        except Exception as exc:
            logger.warning("Falha no ELA: %s", exc)
            return ""

    @staticmethod
    def _record_stage(
        telemetry: List[Dict[str, Any]],
        stage: str,
        start_time: float,
        status: str = "completed",
        detail: str = "",
    ) -> None:
        telemetry.append(
            {
                "stage": stage,
                "status": status,
                "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "detail": detail,
            }
        )

    def _should_run_yolo(self) -> bool:
        return self.analysis_mode not in {"documentos", "screenshots"}

    @staticmethod
    def _ocr_failed(result: str) -> bool:
        lowered = result.lower()
        return lowered.startswith("[erro") or "não disponível" in lowered or "nao disponivel" in lowered

    def _extract_ocr_with_fallback(self, image_data: ImageData) -> Tuple[str, str, str]:
        if self.ocr_engine == "none":
            return "[OCR desabilitado]", "skipped", "ocr_engine=none"

        fallback_engine = {"glm-ocr": "olmocr2", "olmocr2": "glm-ocr"}.get(self.ocr_engine)
        primary_result = self.ollama_client.extract_ocr(image_data, engine=self.ocr_engine)
        if not self._ocr_failed(primary_result) or fallback_engine is None:
            status = "completed" if not self._ocr_failed(primary_result) else "failed"
            return primary_result, status, f"engine={self.ocr_engine}"

        fallback_result = self.ollama_client.extract_ocr(image_data, engine=fallback_engine)
        if not self._ocr_failed(fallback_result):
            return fallback_result, "fallback", f"{self.ocr_engine}->{fallback_engine}"

        return primary_result, "failed", f"engine={self.ocr_engine}; fallback={fallback_engine}"

    def preflight(self, selected_models: List[Tuple[str, str]], output_dir: Path) -> Tuple[List[str], List[str]]:
        errors: List[str] = []
        warnings: List[str] = []

        if not selected_models:
            errors.append("Nenhum modelo selecionado para execução.")

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            probe_file = output_dir / ".vision_write_probe"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink(missing_ok=True)
        except Exception as exc:
            errors.append(f"Diretório de saída indisponível: {exc}")

        needs_openai = any(model_type == "openai" for _, model_type in selected_models)
        if needs_openai and not self.openai_client.is_available():
            errors.append("Modelo OpenAI selecionado, mas a chave OPENAI_API_KEY não está configurada.")

        needs_ollama = self.ocr_engine != "none" or any(model_type == "ollama" for _, model_type in selected_models)
        if needs_ollama and not self.ollama_client.is_available():
            errors.append("OCR/modelo Ollama selecionado, mas o serviço Ollama não está disponível.")

        if self._should_run_yolo() and (not YOLO_AVAILABLE or get_detector is None):
            warnings.append("YOLO indisponível: a etapa de detecção será pulada.")

        if self.post_processor is None:
            warnings.append("Pós-processamento indisponível: o relatório não terá extração estruturada.")

        return errors, warnings

    def prepare_report(
        self,
        image_path: Path,
        cache_manager=None,
    ) -> Tuple[Optional[ImageAnalysisReport], Optional[str]]:
        telemetry: List[Dict[str, Any]] = []

        file_validation_start = time.perf_counter()
        file_error = self._validate_file(image_path)
        if file_error:
            self._record_stage(telemetry, "validate_file", file_validation_start, status="failed", detail=file_error)
            return None, file_error
        self._record_stage(telemetry, "validate_file", file_validation_start)

        load_start = time.perf_counter()
        loaded_image = self.image_processor.load_image(image_path)
        if loaded_image is None:
            self._record_stage(telemetry, "load_image", load_start, status="failed", detail="Falha ao carregar imagem")
            return None, "Falha ao carregar imagem"
        self._record_stage(telemetry, "load_image", load_start)

        image_validation_start = time.perf_counter()
        image_error = self._validate_image(loaded_image)
        if image_error:
            self._record_stage(telemetry, "validate_image", image_validation_start, status="failed", detail=image_error)
            return None, image_error
        self._record_stage(telemetry, "validate_image", image_validation_start)

        prepare_image_start = time.perf_counter()
        image_data = self.image_processor.process_loaded_image(loaded_image, image_path)
        image_for_analysis = Image.open(BytesIO(image_data.jpeg_bytes))
        report_images = [("Imagem analisada", self.image_processor.prepare_for_report(loaded_image))]
        self._record_stage(telemetry, "prepare_image", prepare_image_start)

        cache_lookup_start = time.perf_counter()
        cached_intermediate = cache_manager.get_cached_intermediate(image_path) if cache_manager else None
        cache_status = "hit" if cached_intermediate else "miss"
        self._record_stage(telemetry, "cache_lookup", cache_lookup_start, status=cache_status)

        quality_exif_start = time.perf_counter()
        if cached_intermediate and cached_intermediate.get("quality_result") and cached_intermediate.get("exif_data"):
            quality_str = cached_intermediate["quality_result"]
            exif_str = cached_intermediate["exif_data"]
            self._record_stage(telemetry, "quality_exif", quality_exif_start, status="cached")
        else:
            quality_str, exif_str = self._build_quality_and_exif(image_for_analysis)
            self._record_stage(telemetry, "quality_exif", quality_exif_start)

        yolo_start = time.perf_counter()
        if not self._should_run_yolo():
            yolo_str = "[YOLO pulado para este modo de análise]"
            self._record_stage(telemetry, "yolo", yolo_start, status="skipped", detail=self.analysis_mode)
        elif cached_intermediate and cached_intermediate.get("yolo_result"):
            yolo_str = cached_intermediate["yolo_result"]
            self._record_stage(telemetry, "yolo", yolo_start, status="cached", detail=self.yolo_model)
        else:
            yolo_str = self._build_yolo_summary(image_for_analysis)
            self._record_stage(telemetry, "yolo", yolo_start, detail=self.yolo_model)

        ela_start = time.perf_counter()
        ela_str = self._build_ela_context(image_for_analysis)
        ela_status = "completed" if ela_str else "skipped"
        ela_detail = "forense" if ela_str else self.analysis_mode
        self._record_stage(telemetry, "ela", ela_start, status=ela_status, detail=ela_detail)
        if ela_str:
            quality_str = f"{quality_str}\n\n--- ANÁLISE ELA (Error Level Analysis) ---\n{ela_str}"

        ocr_start = time.perf_counter()
        if (
            cached_intermediate
            and cached_intermediate.get("ocr_result")
            and cached_intermediate.get("ocr_engine") == self.ocr_engine
            and self.ocr_engine != "none"
        ):
            ocr_result = cached_intermediate["ocr_result"]
            self._record_stage(telemetry, "ocr", ocr_start, status="cached", detail=f"engine={self.ocr_engine}")
        else:
            ocr_result, ocr_status, ocr_detail = self._extract_ocr_with_fallback(image_data)
            self._record_stage(telemetry, "ocr", ocr_start, status=ocr_status, detail=ocr_detail)

        prompt_start = time.perf_counter()
        system_prompt = self._build_prompt(ocr_result, yolo_str, quality_str, exif_str)
        self._record_stage(telemetry, "prompt", prompt_start, detail=self.analysis_mode)

        return (
            ImageAnalysisReport(
                image=image_data,
                analysis_mode=self.analysis_mode,
                ocr_engine=self.ocr_engine,
                ocr_result=ocr_result,
                yolo_result=yolo_str,
                quality_result=quality_str,
                exif_result=exif_str,
                prompt_used=system_prompt,
                images=report_images,
                pipeline_telemetry=telemetry,
            ),
            None,
        )

    def _run_post_processing(self, report: ImageAnalysisReport, analysis: AnalysisResult) -> None:
        if not self.post_processor or not analysis.success:
            return

        try:
            pp_result = self.post_processor.process(
                ocr_text=report.ocr_result,
                llm_analysis=analysis.content,
                exif_data=report.exif_result,
                yolo_result=report.yolo_result,
                quality_result=report.quality_result,
            )
            analysis.post_processing = self.post_processor.to_dict(pp_result)
            analysis.post_processing_markdown = self.post_processor.format_report_section(pp_result)
        except Exception as exc:
            logger.debug("Pós-processamento falhou para %s: %s", analysis.model_name, exc)

    def run_model(self, report: ImageAnalysisReport, model_id: str, model_type: str) -> AnalysisResult:
        if model_type == "openai":
            analysis = self.openai_client.analyze_image(report.image, report.prompt_used, model_id)
        else:
            analysis = self.ollama_client.analyze_image(report.image, report.prompt_used, model_id)

        self._run_post_processing(report, analysis)
        return analysis

    def export_analysis(
        self,
        report: ImageAnalysisReport,
        analysis: AnalysisResult,
        output_dir: Path,
        export_formats: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        if not EXPORT_AVAILABLE or ExportManager is None:
            raise RuntimeError("ExportManager não disponível")

        formats = list(export_formats or ["md"])
        if "md" not in formats:
            formats.insert(0, "md")

        exporter = ExportManager(output_dir)
        report_data = build_report_data(report, analysis)
        base_filename = f"{sanitize_filename(report.image.name)}_{get_model_short_name(analysis.model_name)}"
        return exporter.export(report_data, formats=formats, base_filename=base_filename)

    @staticmethod
    def _cached_artifacts_available(cached_report: Path, export_formats: List[str]) -> bool:
        base_path = cached_report.with_suffix("")
        for fmt in export_formats:
            if not base_path.with_suffix(f".{fmt}").exists():
                return False
        return True

    def process_image(
        self,
        image_path: Path,
        selected_models: List[Tuple[str, str]],
        output_dir: Path,
        export_formats: Optional[List[str]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        formats = list(export_formats or ["md"])
        if "md" not in formats:
            formats.insert(0, "md")

        result = {
            "image": image_path.name,
            "success": 0,
            "failed": 0,
            "reports": [],
            "errors": [],
            "warnings": [],
            "telemetry": [],
            "cached_models": [],
            "report": None,
        }

        preflight_errors, preflight_warnings = self.preflight(selected_models, output_dir)
        result["warnings"].extend(preflight_warnings)
        if preflight_errors:
            result["failed"] = len(selected_models)
            result["errors"].extend(preflight_errors)
            return result

        metrics_manager = get_cache_manager() if CACHE_AVAILABLE and get_cache_manager else None
        cache_manager = metrics_manager if use_cache else None

        pending_models: List[Tuple[str, str]] = []
        for model_id, model_type in selected_models:
            cached_path = None
            if cache_manager:
                is_cached, cached_file = cache_manager.is_cached(image_path, model_id, self.analysis_mode, self.ocr_engine)
                if is_cached and cached_file:
                    cached_path = Path(cached_file)
                    if self._cached_artifacts_available(cached_path, formats):
                        result["success"] += 1
                        result["reports"].append(cached_path)
                        result["cached_models"].append(model_id)
                        if metrics_manager:
                            metrics_manager.record_analysis(model_id, 0.0, cache_hit=True)
                        continue

            pending_models.append((model_id, model_type))

        if not pending_models:
            return result

        report, error = self.prepare_report(image_path, cache_manager=cache_manager)
        if error or report is None:
            result["failed"] += len(pending_models)
            result["errors"].append(error or "Falha ao preparar relatório")
            return result

        report.preflight_warnings.extend(preflight_warnings)
        result["report"] = report
        result["telemetry"] = list(report.pipeline_telemetry)

        for model_id, model_type in pending_models:
            analysis = self.run_model(report, model_id, model_type)
            report.analyses.append(analysis)

            if not analysis.success:
                result["failed"] += 1
                result["errors"].append(f"{model_id}: {analysis.error}")
                continue

            try:
                exported = self.export_analysis(report, analysis, output_dir, formats)
                md_path = exported.get("md")
                if md_path is None:
                    md_candidates = [path for path in exported.values() if path.suffix.lower() == ".md"]
                    md_path = md_candidates[0] if md_candidates else None

                if md_path:
                    result["reports"].append(md_path)

                result["success"] += 1

                if metrics_manager:
                    metrics_manager.record_analysis(model_id, analysis.processing_time, cache_hit=False)

                if cache_manager and md_path:
                    cache_manager.add_to_cache(
                        image_path,
                        model_id,
                        self.analysis_mode,
                        self.ocr_engine,
                        md_path,
                        dimensions=report.image.dimensions,
                        ocr_result=report.ocr_result,
                        yolo_result=report.yolo_result,
                        quality_result=report.quality_result,
                        exif_data=report.exif_result,
                    )
            except Exception as exc:
                result["failed"] += 1
                result["errors"].append(f"{model_id}: {exc}")

        # Descarregar modelos Ollama da VRAM após finalizar todas as inferências
        ollama_models_used: List[str] = []
        for model_id, model_type in pending_models:
            if model_type == "ollama" and model_id not in ollama_models_used:
                ollama_models_used.append(model_id)
        if self.ocr_engine != "none":
            ocr_model = OCR_MODEL if self.ocr_engine == "glm-ocr" else OCR_MODEL_ALT
            if ocr_model not in ollama_models_used:
                ollama_models_used.append(ocr_model)
        if ollama_models_used:
            self.ollama_client.unload_models(ollama_models_used)

        return result