#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Constantes e utilitarios compartilhados de runtime do Vision Analyzer."""

from __future__ import annotations

import re
import subprocess

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".avif",
    ".gif",
    ".ico",
    ".heic",
    ".heif",
    ".raw",
    ".cr2",
    ".nef",
    ".arw",
    ".orf",
    ".rw2",
}

RAW_EXTENSIONS = {".raw", ".cr2", ".nef", ".arw", ".orf", ".rw2"}
HEIF_EXTENSIONS = {".heic", ".heif"}

MAX_IMAGE_SIZE = (2048, 2048)
REPORT_IMAGE_SIZE = (1600, 1600)
JPEG_QUALITY = 90

OPENAI_MODEL = "gpt-5.4-mini"
OCR_MODEL = "glm-ocr:bf16"
OCR_MODEL_ALT = "richardyoung/olmocr2:7b-q8"

OLLAMA_VISION_PREFIXES = [
    "qwen",
    "gemma",
    "gemma4",
    "llava",
    "bakllava",
    "ministral",
    "minicpm",
    "moondream",
    "cogvlm",
    "llama-vision",
    "deepseek-vl",
    "internvl",
]

MODEL_SHORT_NAMES = {
    "gpt-5.4-mini": "gpt54-mini",
    "gpt-5-mini": "gpt5-mini",
    "qwen3-vl:8b-thinking-q8_0": "qwen3-vl",
    "gemma3:12b-it-q8_0": "gemma3-12b",
    "ministral-3:14b-instruct-2512-q8_0": "ministral3",
    "qwen3.5:2b": "qwen35-2b",
    "qwen3-vl:2b": "qwen3vl-2b",
    "qwen3.5:4b": "qwen35-4b",
    "qwen3-vl:4b": "qwen3vl-4b",
    "qwen3-vl:8b": "qwen3vl-8b",
    "gemma4:e2b": "gemma4-e2b",
    "qwen3.5:9b-q8_0": "qwen35-9b",
    "gemma4:e4b-it-q8_0": "gemma4-e4b",
    "qwen3-vl:32b": "qwen3vl-32b",
    "gemma4:31b": "gemma4-31b",
}

GPU_MODEL_PROFILES = {
    "4gb": {
        "label": "4 GB VRAM",
        "models": [
            ("qwen3.5:2b", "Qwen3.5 2B (2.7 GB)"),
            ("qwen3-vl:2b", "Qwen3-VL 2B (1.9 GB)"),
        ],
    },
    "6gb": {
        "label": "6 GB VRAM",
        "models": [
            ("qwen3.5:4b", "Qwen3.5 4B (3.4 GB)"),
            ("qwen3-vl:4b", "Qwen3-VL 4B (3.3 GB)"),
        ],
    },
    "8gb": {
        "label": "8 GB VRAM",
        "models": [
            ("qwen3-vl:8b", "Qwen3-VL 8B (6.1 GB)"),
            ("gemma4:e2b", "Gemma4 E2B (7.2 GB)"),
        ],
    },
    "16gb": {
        "label": "16 GB VRAM",
        "models": [
            ("qwen3.5:9b-q8_0", "Qwen3.5 9B Q8 (11 GB)"),
            ("gemma4:e4b-it-q8_0", "Gemma4 E4B Q8 (12 GB)"),
        ],
    },
    "24gb": {
        "label": "24 GB VRAM",
        "models": [
            ("qwen3-vl:32b", "Qwen3-VL 32B (21 GB)"),
            ("gemma4:31b", "Gemma4 31B (20 GB)"),
        ],
    },
    "32gb": {
        "label": "32 GB VRAM",
        "models": [
            ("qwen3-vl:32b", "Qwen3-VL 32B (21 GB)"),
            ("gemma4:31b", "Gemma4 31B (20 GB)"),
        ],
    },
}


def sanitize_filename(filename: str) -> str:
    """Normaliza nomes de arquivo para uso seguro em exports."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    filename = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", filename)

    if len(filename) > 200:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        filename = name[:195] + ("." + ext if ext else "")

    return filename.strip()


def get_model_short_name(model: str) -> str:
    """Retorna um nome curto e estavel para o modelo."""
    return MODEL_SHORT_NAMES.get(model, model.split(":")[0].replace(".", ""))


def detect_vram_gb() -> float:
    """Detecta a quantidade de VRAM em GB. Retorna 0 se nao houver GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return round(props.total_memory / (1024 ** 3), 1)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return round(float(result.stdout.strip()) / 1024, 1)
    except Exception:
        pass

    return 0.0


def get_recommended_gpu_profile(vram_gb: float) -> str:
    """Retorna o tier de GPU recomendado para a VRAM detectada."""
    if vram_gb >= 28:
        return "32gb"
    if vram_gb >= 20:
        return "24gb"
    if vram_gb >= 12:
        return "16gb"
    if vram_gb >= 7.5:
        return "8gb"
    if vram_gb >= 5.5:
        return "6gb"
    if vram_gb >= 3.5:
        return "4gb"
    return ""