#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Image Analyzer - Interface Streamlit
==============================================
Interface gráfica moderna para análise de imagens em lote.

Executar com: streamlit run app.py
"""

import os
import sys
import base64
import hashlib
import json
import tempfile
import time
from pathlib import Path
from datetime import datetime
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional, Generator
import threading
from queue import Queue

import streamlit as st

from runtime_config import (
    GPU_MODEL_PROFILES,
    HEIF_EXTENSIONS,
    JPEG_QUALITY,
    MAX_IMAGE_SIZE,
    OCR_MODEL,
    OCR_MODEL_ALT,
    OLLAMA_VISION_PREFIXES,
    OPENAI_MODEL,
    RAW_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    detect_vram_gb,
    get_model_short_name as shared_get_model_short_name,
    get_recommended_gpu_profile,
    sanitize_filename as shared_sanitize_filename,
)
from batch_checkpoint import (
    BatchCheckpointManager,
    build_batch_job_config,
    build_batch_signature,
    get_default_checkpoint_path,
)

# ============================================================================
# CONFIGURACAO DA PAGINA
# ============================================================================

st.set_page_config(
    page_title="Vision Analyzer",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# IMPORTS E DEPENDENCIAS
# ============================================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Módulos locais
try:
    from cache_manager import get_cache_manager, CacheManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from analysis_pipeline import AnalysisPipeline
    SHARED_PIPELINE_AVAILABLE = True
except ImportError:
    SHARED_PIPELINE_AVAILABLE = False
    
try:
    from export_manager import ExportManager, ReportData, get_available_formats
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False

try:
    from prompt_templates import get_prompt_manager
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False

try:
    from image_preprocessor import ImagePreprocessor
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False

try:
    from api_utils import get_openai_limiter, get_ollama_limiter
    API_UTILS_AVAILABLE = True
except ImportError:
    API_UTILS_AVAILABLE = False

try:
    from object_detector import ObjectDetector, get_detector, is_yolo_available, FORENSIC_CLASSES
    YOLO_AVAILABLE = is_yolo_available()
except ImportError:
    YOLO_AVAILABLE = False

try:
    from chat_assistant import ChatAssistant, create_assistant, ChatSession
    CHAT_AVAILABLE = True
except ImportError:
    CHAT_AVAILABLE = False

try:
    from export_manager import generate_consolidated_pdf, ReportData
    CONSOLIDATED_PDF_AVAILABLE = True
except ImportError:
    CONSOLIDATED_PDF_AVAILABLE = False

try:
    from video_processor import VideoProcessor
    VIDEO_PROCESSOR_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSOR_AVAILABLE = False

try:
    from batch_summary import BatchSummaryBuilder, report_to_summary_entry
    BATCH_SUMMARY_AVAILABLE = True
except ImportError:
    BATCH_SUMMARY_AVAILABLE = False

try:
    from semantic_search import SemanticSearchEngine
except ImportError:
    SemanticSearchEngine = None

try:
    from ela_analyzer import ELAAnalyzer, ELAResult
    ELA_AVAILABLE = True
except ImportError:
    ELA_AVAILABLE = False

try:
    from post_processor import PostProcessor
    POST_PROCESSOR_AVAILABLE = True
except ImportError:
    POST_PROCESSOR_AVAILABLE = False

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False

try:
    import rawpy
    import imageio
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

# Opções de OCR
OCR_ENGINES = {
    "glm-ocr": {
        "name": "GLM OCR (glm-ocr:bf16)",
        "description": "OCR primário para documentos e texto estruturado",
        "requires": "ollama"
    },
    "olmocr2": {
        "name": "OLMoOCR2 (7b-q8)",
        "description": "Alternativa robusto para OCR em cenas complexas",
        "requires": "ollama"
    },
    "none": {
        "name": "Desabilitado",
        "description": "Não extrair texto",
        "requires": None
    }
}

# ============================================================================
# PROMPTS DE SISTEMA
# ============================================================================

# Prompt padrão para análise geral de imagens
_FALLBACK_PROMPT = """Analise a imagem a seguir com foco investigativo e descritivo.

**OCR:**
{ocr_result}

**Objetos detectados:**
{yolo_result}

**Qualidade da imagem:**
{quality_result}

**Metadados EXIF:**
{exif_data}

Gere uma analise detalhada em Markdown descrevendo o conteudo da imagem, elementos relevantes, e quaisquer indicios de manipulacao ou artefatos suspeitos.
"""

# Dicionário de prompts disponíveis
ANALYSIS_PROMPTS = {
    "geral": {
        "name": "📷 Análise Geral",
        "description": "Análise descritiva para acessibilidade e documentação",
        "prompt": _FALLBACK_PROMPT
    },
    "forense": {
        "name": "🔍 Análise Forense",
        "description": "Laudo pericial para investigação policial",
        "prompt": _FALLBACK_PROMPT
    },
    "analise_profunda": {
        "name": "🧠 Análise Profunda",
        "description": "Semiótica, materiais, proxêmica e micro-detalhes (Chain of Thought)",
        "prompt": _FALLBACK_PROMPT
    },
    "screenshots": {
        "name": "🖥️ Análise de Screenshots/Telas",
        "description": "Conversas, páginas web, e-mails e interfaces com foco em hierarquia e sinais de edição",
        "prompt": _FALLBACK_PROMPT
    },
    "forense_completo": {
        "name": "🔬 Análise Forense Completa",
        "description": "Laudo pericial completo com integridade de imagem, manipulação e análise criminal",
        "prompt": _FALLBACK_PROMPT
    }
}


# ============================================================================
# RESOLUCAO DE PROMPTS
# ============================================================================

def resolve_prompt(analysis_mode: str, ocr_result: str = "[OCR não executado]",
                   yolo_result: str = "[YOLO não executado]",
                   quality_result: str = "[Qualidade não avaliada]",
                   exif_data: str = "[EXIF não avaliado]") -> str:
    """Resolve o prompt do sistema, buscando primeiro no prompt_templates, depois no fallback local."""
    fmt_kwargs = dict(ocr_result=ocr_result, yolo_result=yolo_result,
                      quality_result=quality_result, exif_data=exif_data)
    
    # Tentar módulo de prompts primeiro
    if PROMPTS_AVAILABLE:
        from prompt_templates import get_prompt_manager
        mgr = get_prompt_manager()
        template = mgr.get_prompt(analysis_mode)
        if template:
            return template.format_prompt(**fmt_kwargs)
    
    # Fallback local
    if analysis_mode in ANALYSIS_PROMPTS:
        return ANALYSIS_PROMPTS[analysis_mode]["prompt"].format(**fmt_kwargs)
    
    # Último recurso: modo geral
    return ANALYSIS_PROMPTS["geral"]["prompt"].format(**fmt_kwargs)


def resolve_mode_info(analysis_mode: str) -> dict:
    """Retorna nome e descrição do modo, buscando em prompt_templates ou fallback local."""
    if PROMPTS_AVAILABLE:
        from prompt_templates import get_prompt_manager
        mgr = get_prompt_manager()
        template = mgr.get_prompt(analysis_mode)
        if template:
            return {"name": template.name, "description": template.description}
    
    if analysis_mode in ANALYSIS_PROMPTS:
        return ANALYSIS_PROMPTS[analysis_mode]
    
    return ANALYSIS_PROMPTS["geral"]


# ============================================================================
# ESTILOS CSS
# ============================================================================

def apply_custom_css():
    st.markdown("""
    <style>
    /* Cards de status */
    .status-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .status-online {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%);
        border: 1px solid #3d7a4d;
    }
    .status-offline {
        background: linear-gradient(135deg, #4a1a1a 0%, #5a2d2d 100%);
        border: 1px solid #7a3d3d;
    }
    
    /* Thumbnails de imagem */
    .image-card {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        text-align: center;
    }
    
    /* Progress */
    .progress-text {
        font-family: monospace;
        font-size: 0.9rem;
    }
    
    /* Log area */
    .log-container {
        background: #0e1117;
        border: 1px solid #333;
        border-radius: 5px;
        padding: 10px;
        font-family: monospace;
        font-size: 0.85rem;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Model badges */
    .model-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .badge-openai { background: #10a37f; color: white; }
    .badge-ollama { background: #7c3aed; color: white; }
    
    /* Stats */
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #4fc3f7;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# FUNÇÕES UTILITÁRIAS
# ============================================================================

@st.cache_data(ttl=60)
def check_ollama_status() -> tuple[bool, list[str]]:
    """Verifica status do Ollama e modelos disponíveis."""
    if not OLLAMA_AVAILABLE:
        return False, []
    
    try:
        result = ollama.list()
        # SDK novo retorna objetos Model, não dicts
        if hasattr(result, 'models'):
            model_names = [m.model for m in result.models]
        else:
            # Fallback para versões antigas
            model_names = [m.get('name', m.get('model', '')) for m in result.get('models', [])]
        return True, model_names
    except Exception as e:
        print(f"Erro Ollama: {e}")
        return False, []


def is_model_in_ollama(model_name: str, ollama_models: list[str]) -> bool:
    """Verifica se um modelo está disponível no Ollama, comparando por prefixo (ignora tag)."""
    base = model_name.split(":")[0]
    for m in ollama_models:
        if m == model_name or m.split(":")[0] == base:
            return True
    return False


def check_openai_status() -> bool:
    """Verifica se API OpenAI está configurada."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return bool(api_key and api_key.startswith("sk-"))


def load_image(path: Path) -> Optional[Image.Image]:
    """Carrega imagem de qualquer formato suportado."""
    ext = path.suffix.lower()
    
    try:
        if ext in RAW_EXTENSIONS:
            if not RAW_SUPPORT:
                return None
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
            return Image.fromarray(rgb)
        
        elif ext in HEIF_EXTENSIONS:
            if not HEIF_SUPPORT:
                return None
            return Image.open(path)
        
        else:
            return Image.open(path)
            
    except Exception as e:
        st.error(f"Erro ao carregar {path.name}: {e}")
        return None


def prepare_image_for_api(image: Image.Image) -> tuple[str, bytes]:
    """Prepara imagem para APIs (converte para JPEG RGB, retorna base64)."""
    if image.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
        image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
    jpeg_bytes = buffer.getvalue()
    base64_data = base64.b64encode(jpeg_bytes).decode('utf-8')
    
    return base64_data, jpeg_bytes


def find_images(directory: Path) -> list[Path]:
    """Busca imagens recursivamente."""
    images = []
    if directory.exists():
        for file_path in directory.rglob("*"):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(file_path)
    return sorted(images)


def sanitize_filename(name: str) -> str:
    """Remove caracteres inválidos."""
    return shared_sanitize_filename(name)


def get_model_short_name(model: str) -> str:
    """Retorna nome curto do modelo."""
    return shared_get_model_short_name(model)


# ============================================================================
# FUNÇÕES DE IA
# ============================================================================


def analyze_with_openai(base64_data: str, system_prompt: str) -> tuple[bool, str, float]:
    """Analisa com GPT-5.4-mini. Retorna (sucesso, conteúdo/erro, tempo)."""
    if not OPENAI_AVAILABLE:
        return False, "OpenAI não instalado", 0
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "API key não configurada", 0
    
    # Rate limiting
    if API_UTILS_AVAILABLE:
        get_openai_limiter().wait()
    
    start = time.time()
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analise esta imagem seguindo as instruções."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_data}",
                        "detail": "high"
                    }}
                ]}
            ],
            max_tokens=4096,
            temperature=0.1
        )
        elapsed = time.time() - start
        import re as _re
        raw = response.choices[0].message.content or ""
        cleaned = _re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        if not cleaned:
            return False, "Modelo retornou resposta vazia", elapsed
        return True, cleaned, elapsed
    except Exception as e:
        return False, str(e), time.time() - start


def analyze_with_ollama(base64_data: str, system_prompt: str, model: str) -> tuple[bool, str, float]:
    """Analisa com modelo Ollama. Retorna (sucesso, conteúdo/erro, tempo)."""
    if not OLLAMA_AVAILABLE:
        return False, "Ollama não instalado", 0
    
    # Rate limiting
    if API_UTILS_AVAILABLE:
        get_ollama_limiter().wait()
    
    start = time.time()
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Analise esta imagem seguindo as instruções.", "images": [base64_data]}
            ],
            options={"temperature": 0.3, "num_predict": 8192, "num_ctx": 8192}
        )
        elapsed = time.time() - start
        import re as _re
        raw = response['message']['content'] or ""
        cleaned = _re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        if not cleaned:
            return False, "Modelo retornou resposta vazia (possível truncamento do bloco de raciocínio)", elapsed
        return True, cleaned, elapsed
    except Exception as e:
        return False, str(e), time.time() - start


def unload_ollama_models(models: list[str]) -> None:
    """Descarrega modelos Ollama da VRAM após uso."""
    if not OLLAMA_AVAILABLE:
        return
    for model in models:
        try:
            ollama.chat(model=model, messages=[], keep_alive=0)
        except Exception:
            pass



# ============================================================================
# INTERFACE - SIDEBAR
# ============================================================================

def render_sidebar():
    """Renderiza a sidebar com configurações."""
    
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        
        # Status dos serviços
        st.markdown("### 🔌 Status dos Serviços")
        
        # OpenAI
        openai_ok = check_openai_status()
        if openai_ok:
            st.success("✅ OpenAI API configurada")
        else:
            st.error("❌ OpenAI API não configurada")
            with st.expander("Como configurar?"):
                st.markdown("""
                1. Crie um arquivo `.env` na pasta do projeto
                2. Adicione: `OPENAI_API_KEY=sk-...`
                3. Reinicie o aplicativo
                """)
        
        # Ollama
        ollama_ok, ollama_models = check_ollama_status()
        if ollama_ok:
            st.success(f"✅ Ollama online ({len(ollama_models)} modelos)")
        else:
            st.error("❌ Ollama offline")
            with st.expander("Como configurar?"):
                st.markdown("""
                1. Instale o Ollama: https://ollama.ai
                2. Execute: `ollama serve`
                3. Baixe modelos: `ollama pull gemma3:12b-it-q8_0`
                """)
        
        st.markdown("---")
        
        # Seleção do modo de análise
        st.markdown("### 🧭 Modo de Análise")
        
        # Usar prompts do módulo se disponível, senão usar fallback
        if PROMPTS_AVAILABLE:
            prompt_mgr = get_prompt_manager()
            all_prompts = prompt_mgr.get_all_prompts()
            prompt_options = list(all_prompts.keys())
            # v.name já inclui emoji, não adicionar v.icon novamente
            prompt_labels = {k: v.name for k, v in all_prompts.items()}
        else:
            prompt_options = list(ANALYSIS_PROMPTS.keys())
            prompt_labels = {k: v["name"] for k, v in ANALYSIS_PROMPTS.items()}
        
        analysis_mode = st.selectbox(
            "Selecione o tipo de análise:",
            options=prompt_options,
            format_func=lambda x: prompt_labels.get(x, x),
            key="analysis_mode"
        )
        
        # Mostrar descrição do modo selecionado
        if PROMPTS_AVAILABLE:
            prompt_info = prompt_mgr.get_prompt(analysis_mode)
            if prompt_info:
                st.info(f"**{prompt_info.description}**")
        else:
            mode_info = ANALYSIS_PROMPTS.get(analysis_mode, {})
            st.info(f"**{mode_info.get('description', '')}**")
        
        st.markdown("---")
        
        # Seleção de modelos
        st.markdown("### 🤖 Modelos de Análise")
        
        selected_models = []
        
        # OpenAI (padrão)
        st.markdown("**🧠 OpenAI (API) - Padrão**")
        use_openai = st.checkbox(
            "GPT-5.4-mini (recomendado)",
            value=openai_ok,
            disabled=not openai_ok,
            key="use_openai"
        )
        if use_openai:
            selected_models.append((OPENAI_MODEL, "openai"))
        
        # Ollama (opcional)
        st.markdown("**🦙 Ollama (modelos locais) - Opcional**")
        
        # Perfil de GPU
        if ollama_ok:
            vram = detect_vram_gb()
            auto_profile = get_recommended_gpu_profile(vram)
            
            profile_options = ["auto", "4gb", "6gb", "8gb", "16gb", "24gb", "32gb", "custom"]
            profile_labels = {
                "auto": f"🪄 Auto-detectar ({vram:.0f} GB)" if vram > 0 else "🪄 Auto-detectar (sem GPU)",
                "4gb": "⚪ 4 GB VRAM",
                "6gb": "🔵 6 GB VRAM",
                "8gb": "🟢 8 GB VRAM",
                "16gb": "🟡 16 GB VRAM",
                "24gb": "🟠 24 GB VRAM",
                "32gb": "🔴 32 GB VRAM",
                "custom": "🛠️ Customizado",
            }
            
            gpu_profile = st.selectbox(
                "Perfil de GPU:",
                options=profile_options,
                format_func=lambda x: profile_labels.get(x, x),
                key="gpu_profile",
                help="Seleciona modelos Ollama recomendados para sua VRAM"
            )
            
            # Resolver perfil
            active_profile = auto_profile if gpu_profile == "auto" else gpu_profile
            
            if active_profile in GPU_MODEL_PROFILES and gpu_profile != "custom":
                profile_data = GPU_MODEL_PROFILES[active_profile]
                st.caption(f"🎯 Modelos recomendados para **{profile_data['label']}**:")
                if active_profile == "32gb":
                    st.caption("💡 Neste tier, os modelos de 24 GB podem rodar com OCR simultaneamente.")
                
                for model_id, display_name in profile_data["models"]:
                    installed = model_id in ollama_models if ollama_models else False
                    label = f"{'✅ Instalado' if installed else '⬇️ Não instalado'} - {display_name}"
                    use_model = st.checkbox(
                        label,
                        value=False,
                        key=f"use_gpu_{model_id}",
                        help=("Instalado" if installed else f"Não instalado - execute: ollama pull {model_id}")
                    )
                    if use_model:
                        selected_models.append((model_id, "ollama"))
            elif gpu_profile == "auto" and not auto_profile:
                st.caption("⚠️ GPU não detectada ou VRAM insuficiente para modelos locais")
        
        if ollama_ok and ollama_models:
            # Modelos instalados (em expander para não poluir)
            with st.expander("📦 Outros modelos Ollama instalados"):
                # Filtrar modelos com capacidade de visão
                vision_models = [
                    m for m in ollama_models
                    if any(vp in m.split(":")[0].lower() for vp in OLLAMA_VISION_PREFIXES)
                ]
                
                if vision_models:
                    for model_id in sorted(vision_models):
                        use_model = st.checkbox(
                            model_id,
                            value=False,
                            key=f"use_{model_id}"
                        )
                        if use_model:
                            selected_models.append((model_id, "ollama"))
                else:
                    st.caption("Nenhum modelo de visão detectado no Ollama")
            
            # Modelo customizado
            custom_model = st.text_input(
                "Ou digite um modelo Ollama:",
                key="custom_ollama_model",
                placeholder="ex: llava:13b"
            )
            if custom_model and custom_model.strip():
                selected_models.append((custom_model.strip(), "ollama"))
        else:
            st.caption("Ollama offline ou sem modelos instalados")
        
        # OCR
        st.markdown("---")
        st.markdown("### 🔤 OCR (Extração de Texto)")
        
        # Verificar disponibilidade dos engines via Ollama
        glm_available = is_model_in_ollama(OCR_MODEL, ollama_models) if ollama_ok else False
        olmocr2_available = is_model_in_ollama(OCR_MODEL_ALT, ollama_models) if ollama_ok else False
        
        # Construir opções disponíveis
        ocr_options = []
        ocr_labels = {}
        
        if glm_available:
            ocr_options.append("glm-ocr")
            ocr_labels["glm-ocr"] = "🔠 GLM OCR (glm-ocr:bf16)"
        if olmocr2_available:
            ocr_options.append("olmocr2")
            ocr_labels["olmocr2"] = "📖 OLMoOCR2 (7b-q8)"
        ocr_options.append("none")
        ocr_labels["none"] = "❌ Desabilitado"
        
        # Seleção do engine (prioridade: GLM > OLMoOCR2)
        if glm_available:
            default_ocr = "glm-ocr"
        elif olmocr2_available:
            default_ocr = "olmocr2"
        else:
            default_ocr = "none"
        
        ocr_engine = st.radio(
            "Motor de OCR:",
            options=ocr_options,
            format_func=lambda x: ocr_labels.get(x, x),
            index=ocr_options.index(default_ocr) if default_ocr in ocr_options else 0,
            key="ocr_engine",
            horizontal=False
        )
        
        # Status dos engines
        col1, col2 = st.columns(2)
        with col1:
            st.caption("✅ GLM OCR disponível" if glm_available else "❌ GLM OCR indisponível")
        with col2:
            st.caption("✅ OLMoOCR2 disponível" if olmocr2_available else "❌ OLMoOCR2 indisponível")
        
        # Dicas de instalação
        if not glm_available:
            st.caption("📥 Instalar GLM OCR: `ollama pull glm-ocr:bf16`")
        if not olmocr2_available:
            st.caption("📥 Instalar OLMoOCR2: `ollama pull richardyoung/olmocr2:7b-q8`")
        
        st.markdown("---")
        
        # Processamento paralelo
        st.markdown("### ⚡ Processamento")
        
        workers = st.slider(
            "Workers paralelos",
            min_value=1,
            max_value=8,
            value=1,
            key="workers",
            help="Número de imagens processadas simultaneamente (1 = sequencial)"
        )
        
        if workers > 1:
            st.info(f"⚡ **Modo paralelo**: {workers} imagens simultâneas")
        else:
            st.caption("➡️ Modo sequencial (1 imagem por vez)")
        
        # Seleção de modelo YOLO para batch
        if YOLO_AVAILABLE:
            st.selectbox(
                "🎯 Modelo YOLO (batch)",
                options=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
                format_func=lambda x: {
                    "yolo11n": "Nano (rápido)", "yolo11s": "Small",
                    "yolo11m": "Medium", "yolo11l": "Large", "yolo11x": "XLarge (preciso)"
                }.get(x, x),
                index=1,
                key="yolo_batch_model"
            )
        
        st.markdown("---")
        
        # Formatos de exportação
        st.markdown("### 📤 Exportação")
        
        export_formats = ["md"]  # MD sempre habilitado
        
        if EXPORT_AVAILABLE:
            available_formats = get_available_formats()
            
            col1, col2 = st.columns(2)
            with col1:
                if "json" in available_formats:
                    if st.checkbox("📄 JSON", key="export_json"):
                        export_formats.append("json")
                if "html" in available_formats:
                    if st.checkbox("🌐 HTML", key="export_html"):
                        export_formats.append("html")
            with col2:
                if "pdf" in available_formats:
                    if st.checkbox("📕 PDF", key="export_pdf"):
                        export_formats.append("pdf")
                if "docx" in available_formats:
                    if st.checkbox("📝 DOCX", key="export_docx"):
                        export_formats.append("docx")
        else:
            st.caption("📄 Apenas Markdown (instale reportlab e python-docx para mais)")
        
        st.markdown("---")
        
        # Opções avançadas
        with st.expander("🧩 Opções avançadas"):
            use_cache = st.checkbox("🗃️ Usar cache", value=CACHE_AVAILABLE, disabled=not CACHE_AVAILABLE,
                                   help="Evita reprocessar imagens já analisadas")
            auto_enhance = st.checkbox("✨ Auto-correção", value=False, disabled=not PREPROCESSOR_AVAILABLE,
                                      help="Corrige brilho/contraste automaticamente")
            skip_blurry = st.checkbox("🚫 Pular borradas", value=False, disabled=not PREPROCESSOR_AVAILABLE,
                                     help="Ignora imagens muito borradas")
            resume_batch = st.checkbox(
                "♻️ Retomar lote interrompido",
                value=False,
                help="Usa checkpoint por imagem no diretório de saída e pula arquivos já concluídos com a mesma configuração",
            )
            stream_single = st.checkbox(
                "⚡ Streaming (imagem única)",
                value=True,
                help="Mostra a resposta do modelo token a token na análise individual. "
                     "Vale apenas para um modelo por vez e sem cache.",
            )
        
        st.markdown("---")
        
        # Diretórios
        st.markdown("### 📁 Diretórios")
        
        input_dir = st.text_input(
            "Pasta de entrada",
            value="./imagens_entrada",
            key="input_dir"
        )
        
        output_dir = st.text_input(
            "Pasta de saída",
            value="./relatorios_saida",
            key="output_dir"
        )
        
        # Salvar no session state (apenas variáveis que não são keys de widgets)
        st.session_state['selected_models'] = selected_models
        st.session_state['export_formats'] = export_formats
        st.session_state['use_cache'] = use_cache if CACHE_AVAILABLE else False
        st.session_state['auto_enhance'] = auto_enhance if PREPROCESSOR_AVAILABLE else False
        st.session_state['skip_blurry'] = skip_blurry if PREPROCESSOR_AVAILABLE else False
        st.session_state['resume_batch'] = resume_batch
        st.session_state['stream_single_analysis'] = stream_single


        return selected_models, ocr_engine, Path(input_dir), Path(output_dir), analysis_mode, workers


# ============================================================================
# INTERFACE - MAIN
# ============================================================================

def render_header():
    """Renderiza o cabeçalho."""
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("# 👁️")
    with col2:
        st.markdown("# Vision Analyzer")
        st.caption("Análise de imagens com múltiplos modelos de IA")


def render_image_gallery(images: list[Path]):
    """Renderiza galeria de imagens encontradas."""
    
    if not images:
        st.warning("Nenhuma imagem encontrada no diretorio de entrada.")
        st.info(f"Coloque imagens na pasta: `{st.session_state.get('input_dir', './imagens_entrada')}`")
        return
    
    st.markdown(f"### 🖼️ Imagens encontradas ({len(images)})")
    
    # Grid de thumbnails
    cols = st.columns(6)
    
    for idx, img_path in enumerate(images[:24]):  # Limitar a 24 para performance
        with cols[idx % 6]:
            try:
                img = load_image(img_path)
                if img:
                    # Criar thumbnail
                    thumb = img.copy()
                    thumb.thumbnail((150, 150))
                    st.image(thumb, caption=img_path.name[:15] + "...", width='stretch')
            except Exception:
                st.markdown(f"📄 {img_path.name[:10]}...")
    
    if len(images) > 24:
        st.caption(f"... e mais {len(images) - 24} imagens")


def render_analysis_panel(images: list[Path], selected_models: list, ocr_engine: str, output_dir: Path, analysis_mode: str, workers: int):
    """Painel de execução da análise."""
    
    st.markdown("---")
    st.markdown("### 🚀 Executar análise")
    
    if not images:
        st.warning("Adicione imagens ao diretório de entrada primeiro.")
        return
    
    if not selected_models:
        st.warning("Selecione pelo menos um modelo na sidebar.")
        return
    
    # Mostrar modo selecionado
    mode_info = resolve_mode_info(analysis_mode)
    st.info(f"**Modo selecionado:** {mode_info['name']} - {mode_info['description']}")
    
    # Mostrar OCR selecionado
    ocr_info = OCR_ENGINES.get(ocr_engine, {})
    ocr_name = ocr_info.get('name', ocr_engine)
    
    default_execution_mode = "single" if len(images) == 1 else "batch"
    execution_options = ["single"] if len(images) == 1 else ["single", "batch"]
    if len(images) == 1:
        st.session_state["analysis_execution_mode"] = "single"

    execution_mode = st.radio(
        "Modo de execução",
        options=execution_options,
        index=0 if default_execution_mode == "single" else 1,
        horizontal=True,
        format_func=lambda mode: "🖼️ Imagem única" if mode == "single" else "📚 Lote completo",
        key="analysis_execution_mode",
    )

    export_formats = st.session_state.get('export_formats', ['md'])

    if execution_mode == "single":
        selected_image = st.selectbox(
            "Imagem para analisar",
            options=images,
            format_func=lambda path: path.name,
            key="single_analysis_image",
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Imagem", "1")
        with col2:
            st.metric("Modelos", len(selected_models))
        with col3:
            st.metric("Análises", len(selected_models))
        with col4:
            st.metric("OCR", {"glm-ocr": "🔠 GLM", "olmocr2": "📖 OLMo"}.get(ocr_engine, "❌ Off"))

        if len(images) > 1:
            st.caption("Executa apenas a imagem selecionada. Workers e checkpoint de lote nao sao usados neste modo.")
        else:
            st.caption("Uma unica imagem detectada. A interface usara o fluxo dedicado de analise individual.")

        if st.button("▶️ Analisar imagem", type="primary", width='stretch'):
            result = run_single_image_analysis(
                selected_image,
                selected_models,
                ocr_engine,
                output_dir,
                analysis_mode,
                export_formats,
            )
            st.session_state["single_analysis_result"] = {
                "image": str(selected_image),
                "result": {
                    "image": result.get("image"),
                    "success": result.get("success", 0),
                    "failed": result.get("failed", 0),
                    "errors": list(result.get("errors", [])),
                    "warnings": list(result.get("warnings", [])),
                    "telemetry": list(result.get("telemetry", [])),
                    "cached_models": list(result.get("cached_models", [])),
                    "reports": [str(path) for path in result.get("reports", [])],
                },
            }

        persisted_single = st.session_state.get("single_analysis_result")
        if persisted_single and persisted_single.get("image") == str(selected_image):
            render_single_analysis_result(selected_image, persisted_single["result"])
        return

    # Resumo de lote
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Imagens", len(images))
    with col2:
        st.metric("Modelos", len(selected_models))
    with col3:
        st.metric("Análises", len(images) * len(selected_models))
    with col4:
        st.metric("OCR", {"glm-ocr": "🔠 GLM", "olmocr2": "📖 OLMo"}.get(ocr_engine, "❌ Off"))
    with col5:
        st.metric("Workers", str(workers))
    
    # Botão de início
    if st.button("▶️ Iniciar análise em lote", type="primary", width='stretch'):
        run_batch_analysis(images, selected_models, ocr_engine, output_dir, analysis_mode, workers, export_formats)


def render_single_analysis_result(image_path: Path, result: dict):
    """Renderiza o resultado persistido da análise individual."""
    st.markdown("### 🖼️ Resultado da análise individual")
    st.caption(f"Imagem: {image_path.name}")

    warnings = result.get("warnings") or []
    errors = result.get("errors") or []
    cached_models = result.get("cached_models") or []
    report_paths = [Path(path) for path in result.get("reports", [])]
    telemetry = result.get("telemetry") or []

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sucesso", str(result.get("success", 0)))
    with col2:
        st.metric("Falhas", str(result.get("failed", 0)))
    with col3:
        st.metric("Relatórios", str(len(report_paths)))

    if warnings:
        for warning in warnings:
            st.warning(warning)

    if errors:
        for error in errors:
            st.error(error)

    if cached_models:
        cached_labels = ", ".join(get_model_short_name(model) for model in cached_models)
        st.info(f"Resultados reaproveitados do cache: {cached_labels}")

    if telemetry:
        st.markdown("#### Telemetria da pipeline")
        st.dataframe(telemetry, width='stretch', hide_index=True)

    if report_paths:
        st.markdown("#### Relatórios gerados")
        for idx, report_path in enumerate(report_paths):
            if not report_path.exists():
                continue
            render_report_preview(report_path, key_prefix=f"single_report_{idx}")


def render_report_preview(report_path: Path, key_prefix: str):
    """Mostra uma visualização rápida de um relatório gerado."""
    content = report_path.read_text(encoding="utf-8")
    json_report_path = report_path.with_suffix('.json')
    report_payload = None

    if json_report_path.exists():
        try:
            report_payload = json.loads(json_report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            st.warning(f"Nao foi possivel ler o JSON estruturado de {report_path.name}: {exc}")

    with st.expander(f"📄 {report_path.name}"):
        if report_payload:
            tab_markdown, tab_structured = st.tabs(["📝 Markdown", "🧠 Estruturado"])
        else:
            tab_markdown = st.container()
            tab_structured = None

        with tab_markdown:
            st.markdown(
                f'<div class="report-container">{content}</div>',
                unsafe_allow_html=True,
            )

        if tab_structured and report_payload:
            with tab_structured:
                image_payload = report_payload.get("image", {})
                analysis_payload = report_payload.get("analysis", {})
                pre_payload = analysis_payload.get("pre_analysis", {})
                post_payload = analysis_payload.get("post_processing", {})

                col_meta1, col_meta2, col_meta3 = st.columns(3)
                with col_meta1:
                    st.metric("Modelo", analysis_payload.get("model", "-"))
                with col_meta2:
                    st.metric("Modo", analysis_payload.get("mode", "-"))
                with col_meta3:
                    st.metric("OCR", analysis_payload.get("ocr_engine", "-"))

                st.markdown("**Imagem**")
                st.caption(image_payload.get("name") or report_path.stem)
                st.markdown("**OCR**")
                st.code(analysis_payload.get("ocr_result") or "[Nenhum texto detectado]", language="text")
                st.markdown("**EXIF e GPS**")
                st.code(pre_payload.get("exif_result") or "[EXIF nao avaliado]", language="text")
                st.markdown("**YOLO11**")
                st.code(pre_payload.get("yolo_result") or "[YOLO nao executado]", language="text")
                st.markdown("**Qualidade**")
                st.code(pre_payload.get("quality_result") or "[Qualidade nao avaliada]", language="text")

                if post_payload:
                    summary_text = post_payload.get("summary")
                    if summary_text:
                        st.info(summary_text)

                    classification = post_payload.get("classification") or {}
                    validation = post_payload.get("validation") or {}
                    entities = post_payload.get("entities") or {}
                    timeline = post_payload.get("timeline") or []

                    col_post1, col_post2 = st.columns(2)
                    with col_post1:
                        st.markdown("**Classificacao**")
                        if classification:
                            st.json(classification)
                        else:
                            st.caption("Sem classificacao estruturada.")
                    with col_post2:
                        st.markdown("**Validacao OCR x LLM**")
                        if validation:
                            st.json(validation)
                        else:
                            st.caption("Sem validacao estruturada.")

                    entity_items = {key: value for key, value in entities.items() if value}
                    if entity_items:
                        st.markdown("**Entidades extraidas**")
                        st.json(entity_items)

                    if timeline:
                        st.markdown("**Linha do tempo**")
                        for item in timeline:
                            timestamp = item.get("timestamp") or item.get("parsed_dt") or "Sem data"
                            description = item.get("description") or "Sem descricao"
                            st.write(f"- {timestamp}: {description}")

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "📥 Baixar Markdown",
                data=content,
                file_name=report_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_md",
            )
        with download_col2:
            if json_report_path.exists():
                st.download_button(
                    "📥 Baixar JSON",
                    data=json_report_path.read_text(encoding="utf-8"),
                    file_name=json_report_path.name,
                    mime="application/json",
                    key=f"{key_prefix}_download_json",
                )


def process_single_image_task(
    img_path: Path, 
    selected_models: list, 
    ocr_engine: str, 
    output_dir: Path, 
    analysis_mode: str,
    yolo_model: str = "yolo11s",
    use_cache: bool = False,
    export_formats: list = None,
) -> dict:
    """
    Processa uma única imagem com todos os modelos selecionados.
    Retorna dict com resultados para uso em paralelo.
    """
    if not SHARED_PIPELINE_AVAILABLE:
        return {
            "image": img_path.name,
            "success": 0,
            "failed": len(selected_models),
            "reports": [],
            "errors": ["Pipeline compartilhado não disponível"],
        }

    try:
        pipeline = AnalysisPipeline(
            analysis_mode=analysis_mode,
            ocr_engine=ocr_engine,
            yolo_model=yolo_model,
        )
        return pipeline.process_image(
            image_path=img_path,
            selected_models=selected_models,
            output_dir=output_dir,
            export_formats=export_formats,
            use_cache=use_cache,
        )
    except Exception as e:
        return {
            "image": img_path.name,
            "success": 0,
            "failed": len(selected_models),
            "reports": [],
            "errors": [str(e)],
        }


def run_single_image_analysis(
    image_path: Path,
    selected_models: list,
    ocr_engine: str,
    output_dir: Path,
    analysis_mode: str = "geral",
    export_formats: list = None,
) -> dict:
    """Executa a análise dedicada de uma única imagem na UI."""
    if not SHARED_PIPELINE_AVAILABLE:
        st.error("Pipeline compartilhado não disponível. Verifique o módulo analysis_pipeline.py.")
        return {
            "image": image_path.name,
            "success": 0,
            "failed": len(selected_models),
            "reports": [],
            "errors": ["Pipeline compartilhado não disponível"],
            "warnings": [],
            "telemetry": [],
            "cached_models": [],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    export_formats = export_formats or ["md"]
    yolo_batch_model = st.session_state.get('yolo_batch_model', 'yolo11s')
    use_cache_flag = st.session_state.get('use_cache', False)

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text(f"🧠 Preparando análise de {image_path.name}...")
    progress_bar.progress(0.2)

    pipeline = AnalysisPipeline(
        analysis_mode=analysis_mode,
        ocr_engine=ocr_engine,
        yolo_model=yolo_batch_model,
    )

    # Streaming só compensa quando há um único modelo: com vários, os textos
    # se sobreporiam no mesmo placeholder.
    use_streaming = (
        st.session_state.get('stream_single_analysis', True)
        and len(selected_models) == 1
        and not use_cache_flag
    )

    if use_streaming:
        result = _run_streamed_single_analysis(
            pipeline, image_path, selected_models[0], output_dir, export_formats, status_text, progress_bar
        )
    else:
        result = pipeline.process_image(
            image_path=image_path,
            selected_models=selected_models,
            output_dir=output_dir,
            export_formats=export_formats,
            use_cache=use_cache_flag,
        )

    # O pipeline não descarrega mais a VRAM por imagem; na análise avulsa liberamos aqui.
    pipeline.unload_models()

    progress_bar.progress(1.0)
    status_text.text("✅ Análise individual concluída!")

    if result.get("success", 0) > 0:
        st.success(f"✅ Análise concluída! {result['success']} relatório(s) gerado(s).")
    if result.get("failed", 0) > 0:
        st.warning(f"⚠️ {result['failed']} modelo(s) falharam nesta execução.")

    return result


def _run_streamed_single_analysis(
    pipeline,
    image_path: Path,
    model: tuple,
    output_dir: Path,
    export_formats: list,
    status_text,
    progress_bar,
) -> dict:
    """Executa a análise de uma imagem exibindo a resposta token a token."""
    model_id, model_type = model
    result = {
        "image": image_path.name, "success": 0, "failed": 0, "reports": [],
        "errors": [], "warnings": [], "telemetry": [], "cached_models": [], "report": None,
    }

    preflight_errors, preflight_warnings = pipeline.preflight([model], output_dir)
    result["warnings"].extend(preflight_warnings)
    if preflight_errors:
        result["failed"] = 1
        result["errors"].extend(preflight_errors)
        return result

    status_text.text(f"🔬 Executando OCR, YOLO e análises forenses em {image_path.name}...")
    report, error = pipeline.prepare_report(image_path)
    if error or report is None:
        result["failed"] = 1
        result["errors"].append(error or "Falha ao preparar relatório")
        return result

    report.preflight_warnings.extend(preflight_warnings)
    result["report"] = report
    result["telemetry"] = list(report.pipeline_telemetry)

    progress_bar.progress(0.5)
    status_text.text(f"🤖 {model_id} analisando...")

    st.markdown(f"#### 🤖 {get_model_short_name(model_id)}")
    stream_placeholder = st.empty()

    chunks = []
    start = time.perf_counter()
    for chunk in pipeline.stream_model(report, model_id, model_type):
        chunks.append(chunk)
        stream_placeholder.markdown("".join(chunks))
    elapsed = time.perf_counter() - start

    analysis = pipeline.finalize_streamed_analysis(
        report, model_id, model_type, "".join(chunks), elapsed
    )
    report.analyses.append(analysis)
    stream_placeholder.markdown(analysis.content)

    if not analysis.success:
        result["failed"] = 1
        result["errors"].append(f"{model_id}: {analysis.error}")
        return result

    progress_bar.progress(0.9)
    status_text.text("💾 Exportando relatório...")
    try:
        exported = pipeline.export_analysis(report, analysis, output_dir, export_formats)
        md_path = exported.get("md")
        if md_path:
            result["reports"].append(md_path)
        result["success"] = 1
    except Exception as exc:
        result["failed"] = 1
        result["errors"].append(f"{model_id}: {exc}")

    return result


def run_batch_analysis(images: list[Path], selected_models: list, ocr_engine: str, output_dir: Path, analysis_mode: str = "geral", workers: int = 1, export_formats: list = None):
    """Executa a análise em lote (sequencial ou paralelo)."""
    if not SHARED_PIPELINE_AVAILABLE:
        st.error("Pipeline compartilhado não disponível. Verifique o módulo analysis_pipeline.py.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    export_formats = export_formats or ["md"]

    yolo_batch_model = st.session_state.get('yolo_batch_model', 'yolo11s')
    use_cache_flag = st.session_state.get('use_cache', False)
    resume_batch = st.session_state.get('resume_batch', False)

    checkpoint_job_config = build_batch_job_config(
        selected_models=selected_models,
        analysis_mode=analysis_mode,
        ocr_engine=ocr_engine,
        export_formats=export_formats,
        yolo_model=yolo_batch_model,
    )
    checkpoint_manager = BatchCheckpointManager(
        checkpoint_path=get_default_checkpoint_path(output_dir),
        job_signature=build_batch_signature(checkpoint_job_config),
        job_config=checkpoint_job_config,
    )
    resume_state = checkpoint_manager.prepare_run(images, resume=resume_batch)
    requested_images = len(images)
    images = resume_state.pending_images
    
    total_images = len(images)
    total_tasks = total_images * len(selected_models)
    
    # Containers de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    # Placeholder, não container: escrever dentro de um st.container() ACRESCENTA um bloco
    # novo a cada chamada, e um lote de centenas de imagens empilhava centenas de blocos
    # repetindo as mesmas 10 últimas linhas.
    log_placeholder = st.empty()

    results = {
        "success": 0,
        "failed": 0,
        "reports": [],
        "resumed_skipped": len(resume_state.skipped_entries),
    }
    # Relatórios completos do pipeline, usados para o sumário consolidado do lote.
    batch_reports = []

    for entry in resume_state.skipped_entries:
        for report_path in entry.get("reports", []):
            report_file = Path(report_path)
            if report_file.exists():
                results["reports"].append(report_file)
    
    logs = []
    
    def add_log(msg: str):
        logs.append(f"`{datetime.now().strftime('%H:%M:%S')}` {msg}")
        log_placeholder.markdown("<br>".join(logs[-10:]), unsafe_allow_html=True)
    
    mode_str = f"⚡ Paralelo ({workers} workers)" if workers > 1 else "➡️ Sequencial"
    add_log(f"🚀 Iniciando análise em lote... [{mode_str}]")
    add_log(f"🖼️ {requested_images} imagens solicitadas - 🤖 {len(selected_models)} modelos")
    if resume_batch:
        add_log(f"♻️ Checkpoint: {checkpoint_manager.checkpoint_path.name}")
    if resume_state.reset_reason:
        add_log(f"♻️ {resume_state.reset_reason}")
    if results["resumed_skipped"] > 0:
        add_log(f"♻️ {results['resumed_skipped']} imagens já concluídas foram puladas")
    add_log(f"🧠 {total_images} imagens pendentes = {total_tasks} análises nesta execução")

    if not images:
        progress_bar.progress(1.0)
        status_text.text("✅ Nenhuma imagem pendente. Lote já concluído para esta configuração.")
        add_log("🏁 Nenhuma imagem pendente; apenas relatórios existentes foram reaproveitados")
        st.success("✅ Nenhuma imagem pendente. Os relatórios existentes permanecem válidos para esta configuração.")
        if results["reports"]:
            st.markdown("### 📑 Relatórios disponíveis")
            for report_path in results["reports"][:10]:
                with st.expander(f"📄 {report_path.name}"):
                    st.markdown(report_path.read_text(encoding='utf-8'))
        return
    
    if workers > 1:
        # ===== MODO PARALELO =====
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        add_log(f"⚡ Processando {workers} imagens simultaneamente...")
        
        completed_images = 0
        completed_analyses = 0
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submeter tarefas
            futures = {
                executor.submit(
                    process_single_image_task,
                    img_path, selected_models, ocr_engine, output_dir, analysis_mode,
                    yolo_batch_model, use_cache_flag, export_formats
                ): img_path 
                for img_path in images
            }
            
            # Processar resultados conforme completam
            for future in as_completed(futures):
                completed_images += 1
                img_path = futures[future]
                
                try:
                    task_result = future.result()
                    checkpoint_manager.record_result(
                        image_path=img_path,
                        success_count=task_result["success"],
                        failed_count=task_result["failed"],
                        reports=task_result["reports"],
                        errors=task_result["errors"],
                    )
                    
                    results["success"] += task_result["success"]
                    results["failed"] += task_result["failed"]
                    results["reports"].extend([Path(p) for p in task_result["reports"]])
                    if task_result.get("report") is not None:
                        batch_reports.append(task_result["report"])
                    completed_analyses += task_result["success"] + task_result["failed"]
                    
                    # Progresso granular: por análise, não só por imagem
                    progress = min(completed_analyses / total_tasks, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(
                        f"🖼️ Imagens: {completed_images}/{total_images} | "
                        f"🤖 Análises: {completed_analyses}/{total_tasks} ({progress*100:.0f}%)"
                    )
                    
                    if task_result["success"] > 0:
                        add_log(f"✅ **{task_result['image']}**: {task_result['success']}/{len(selected_models)} relatórios")
                    if task_result.get("cached_models"):
                        cached_labels = ", ".join(get_model_short_name(model) for model in task_result["cached_models"])
                        add_log(f"   🗃️ Cache: {cached_labels}")
                    if task_result["errors"]:
                        for err in task_result["errors"][:2]:
                            add_log(f"   ⚠️ {err[:60]}...")
                            
                except Exception as e:
                    add_log(f"❌ Erro em {img_path.name}: {e}")
                    results["failed"] += len(selected_models)
                    completed_analyses += len(selected_models)
    else:
        # ===== MODO SEQUENCIAL (original) =====
        pipeline = AnalysisPipeline(
            analysis_mode=analysis_mode,
            ocr_engine=ocr_engine,
            yolo_model=yolo_batch_model,
        )
        current_task = 0
        
        for img_idx, img_path in enumerate(images):
            add_log(f"🖼️ Processando: **{img_path.name}**")

            status_text.text(f"🧠 Preparando análise compartilhada para {img_path.name}...")
            task_result = pipeline.process_image(
                image_path=img_path,
                selected_models=selected_models,
                output_dir=output_dir,
                export_formats=export_formats,
                use_cache=use_cache_flag,
            )
            checkpoint_manager.record_result(
                image_path=img_path,
                success_count=task_result["success"],
                failed_count=task_result["failed"],
                reports=task_result["reports"],
                errors=task_result["errors"],
            )

            results["success"] += task_result["success"]
            results["failed"] += task_result["failed"]
            results["reports"].extend([Path(p) for p in task_result["reports"]])
            if task_result.get("report") is not None:
                batch_reports.append(task_result["report"])

            current_task += task_result["success"] + task_result["failed"]
            progress = min(current_task / total_tasks, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"🤖 [{current_task}/{total_tasks}] {img_path.name} concluída")

            if task_result["success"] > 0:
                add_log(f"   ✅ {task_result['success']} relatórios gerados")
            if task_result.get("cached_models"):
                cached_labels = ", ".join(get_model_short_name(model) for model in task_result["cached_models"])
                add_log(f"   🗃️ Cache: {cached_labels}")
            if task_result["errors"]:
                for err in task_result["errors"][:3]:
                    add_log(f"   ⚠️ {err[:80]}")
    
    # Finalizado
    progress_bar.progress(1.0)
    status_text.text("✅ Análise concluída!")
    
    add_log("---")
    add_log("🏁 **Análise finalizada!**")
    add_log(
        f"   ✅ Sucesso: {results['success']} | ❌ Falhas: {results['failed']} | ♻️ Puladas: {results['resumed_skipped']}"
    )
    
    # Mostrar resultados
    st.success(f"✅ Análise concluída! {results['success']} relatórios gerados nesta execução.")
    if results["resumed_skipped"] > 0:
        st.info(f"♻️ {results['resumed_skipped']} imagens já concluídas foram reaproveitadas do checkpoint.")
    
    if results["reports"]:
        st.markdown("### 📑 Relatórios disponíveis")
        for report_path in results["reports"][:10]:
            with st.expander(f"📄 {report_path.name}"):
                st.markdown(report_path.read_text(encoding='utf-8'))

    # Sumário consolidado do lote (entidades, timeline, GPS, objetos, duplicatas)
    render_batch_summary(batch_reports, output_dir, add_log)

    # Descarregar modelos Ollama da VRAM, uma vez ao fim do lote.
    ollama_models_to_unload = [m[0] for m in selected_models if isinstance(m, tuple) and m[1] == "ollama"]
    if ocr_engine and ocr_engine != "none":
        ocr_model = OCR_MODEL if ocr_engine == "glm-ocr" else OCR_MODEL_ALT
        if ocr_model not in ollama_models_to_unload:
            ollama_models_to_unload.append(ocr_model)
    if ollama_models_to_unload:
        unload_ollama_models(ollama_models_to_unload)


def render_batch_summary(batch_reports: list, output_dir: Path, add_log=None) -> None:
    """Gera e exibe o sumário consolidado do lote, salvando resumo_lote.md.

    Estava disponível apenas na CLI, embora anunciado como funcionalidade do produto.
    """
    if not BATCH_SUMMARY_AVAILABLE or not batch_reports:
        return

    try:
        entries = [report_to_summary_entry(report) for report in batch_reports if report is not None]
        if not entries:
            return

        summary = BatchSummaryBuilder().build(entries)
        markdown = summary.to_markdown()

        summary_path = output_dir / "resumo_lote.md"
        summary_path.write_text(markdown, encoding="utf-8")
        if add_log:
            add_log(f"📊 Sumário consolidado salvo em `{summary_path.name}`")

        st.markdown("### 📊 Sumário consolidado do lote")

        cols = st.columns(4)
        cols[0].metric("Imagens", summary.total_images)
        cols[1].metric("Entidades", sum(summary.entity_counts.values()))
        cols[2].metric("Pontos GPS", len(summary.gps_coordinates))
        cols[3].metric("Objetos", sum(summary.top_objects.values()))

        with st.expander("📄 Ver relatório consolidado", expanded=False):
            st.markdown(markdown)

        st.download_button(
            "⬇️ Baixar resumo_lote.md",
            data=markdown,
            file_name="resumo_lote.md",
            mime="text/markdown",
            key="download_batch_summary",
        )
    except Exception as exc:
        st.warning(f"⚠️ Não foi possível gerar o sumário do lote: {exc}")


# ============================================================================
# CONFIGURAÇÕES E SOBRE
# ============================================================================

def render_settings_panel():
    """Painel de configurações avançadas."""
    st.markdown("### ⚙️ Configurações avançadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cache
        st.markdown("#### 🗃️ Cache de análises")
        if CACHE_AVAILABLE:
            cache_mgr = get_cache_manager()
            stats = cache_mgr.get_cache_stats()
            
            st.metric("Entradas em Cache", stats["total_entries"])
            st.metric("Tamanho Total", f"{stats['total_size_mb']:.2f} MB")
            
            if st.button("🧹 Limpar cache"):
                cache_mgr.clear_cache()
                st.success("Cache limpo!")
                st.rerun()
        else:
            st.warning("⚠️ Módulo de cache não disponível")
        
        st.markdown("---")
        
        # Formatos de Exportação
        st.markdown("#### 📤 Formatos de exportação")
        if EXPORT_AVAILABLE:
            formats = get_available_formats()
            st.write("Formatos disponíveis:")
            for fmt in formats:
                st.write(f"- {fmt.upper()}")
        else:
            st.warning("⚠️ Módulo de exportação não disponível")
            st.caption("Instale: `pip install reportlab python-docx`")
    
    with col2:
        # Templates de Prompts
        st.markdown("#### 📝 Templates de prompts")
        if PROMPTS_AVAILABLE:
            prompt_mgr = get_prompt_manager()
            all_prompts = prompt_mgr.get_all_prompts()
            
            st.write(f"**{len(all_prompts)}** templates disponíveis")
            
            # Listar categorias
            categories = prompt_mgr.get_categories()
            for cat in categories:
                prompts_in_cat = prompt_mgr.get_prompts_by_category(cat)
                with st.expander(f"📂 {cat.title()} ({len(prompts_in_cat)})"):
                    for p in prompts_in_cat:
                        st.write(f"{p.icon} **{p.name}**")
                        st.caption(p.description)
            
            st.markdown("---")
            
            # Criar novo prompt
            st.markdown("##### ✍️ Criar novo prompt")
            with st.expander("Criar Prompt Customizado"):
                new_name = st.text_input("Nome do Prompt")
                new_desc = st.text_input("Descrição")
                new_cat = st.selectbox("Categoria", ["geral", "forense", "documentos", "screenshots", "veiculos", "pessoas", "custom"])
                new_prompt = st.text_area("Prompt (use {ocr_result} para OCR)", height=200)
                
                if st.button("💾 Salvar prompt"):
                    if new_name and new_prompt:
                        prompt_mgr.create_custom_prompt(new_name, new_desc, new_cat, new_prompt)
                        st.success(f"Prompt '{new_name}' criado!")
                        st.rerun()
                    else:
                        st.error("Nome e Prompt são obrigatórios")
        else:
            st.warning("⚠️ Módulo de prompts não disponível")
        
        st.markdown("---")
        
        # Pré-processamento
        st.markdown("#### 🧪 Pré-processamento")
        if PREPROCESSOR_AVAILABLE:
            st.write("✅ Auto-rotação EXIF")
            st.write("✅ Detecção de blur")
            st.write("✅ Análise de qualidade")
            st.write("✅ Correção automática")
        else:
            st.warning("⚠️ Pré-processador não disponível")
            st.caption("Instale: `pip install opencv-python numpy`")
    
    st.markdown("---")
    
    # Status dos módulos
    st.markdown("#### 📊 Status dos módulos")
    
    modules = [
        ("Cache", CACHE_AVAILABLE),
        ("Export", EXPORT_AVAILABLE),
        ("YOLO", YOLO_AVAILABLE),
        ("Chat", CHAT_AVAILABLE),
        ("Pós-Proc", POST_PROCESSOR_AVAILABLE),
    ]
    
    cols = st.columns(len(modules))
    for i, (name, available) in enumerate(modules):
        with cols[i]:
            status = "OK" if available else "OFF"
            st.metric(name, status)


def render_chat_panel(images: list, selected_models: list):
    """Painel de chat interativo."""
    st.markdown("### 💬 Chat interativo")
    st.caption("Converse com a IA sobre uma imagem específica")
    
    if not CHAT_AVAILABLE:
        st.error("❌ Módulo de chat não disponível")
        return
    
    if not images:
        st.warning("⚠️ Nenhuma imagem disponível para chat")
        return
    
    if not selected_models:
        st.warning("⚠️ Selecione pelo menos um modelo na sidebar")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Seleção de imagem
        image_names = [img.name for img in images]
        selected_image_name = st.selectbox("🖼️ Selecione a imagem:", image_names, key="chat_image")
        selected_image_path = images[image_names.index(selected_image_name)]
        
        # Preview da imagem
        img = load_image(selected_image_path)
        if img:
            st.image(img, caption=selected_image_name, width='stretch')
        
        # Seleção de modelo
        model_options = [(m[0], f"{m[0]} ({'OpenAI' if m[1] == 'openai' else 'Ollama'})") for m in selected_models]
        selected_model_id = st.selectbox(
            "🤖 Modelo para chat:",
            options=[m[0] for m in model_options],
            format_func=lambda x: dict(model_options).get(x, x),
            key="chat_model"
        )
        selected_model_type = "openai" if any(m[1] == "openai" and m[0] == selected_model_id for m in selected_models) else "ollama"
        
        # Botão para iniciar/resetar sessão
        if st.button("🔄 Nova sessão", width='stretch'):
            if 'chat_session' in st.session_state:
                del st.session_state['chat_session']
            if 'chat_messages' in st.session_state:
                del st.session_state['chat_messages']
            st.rerun()
    
    with col2:
        # Área de chat
        st.markdown("#### 🗨️ Conversa")
        
        # Inicializar histórico de mensagens
        if 'chat_messages' not in st.session_state:
            st.session_state['chat_messages'] = []
        
        # Container para mensagens
        chat_container = st.container(height=400)
        
        with chat_container:
            for msg in st.session_state['chat_messages']:
                if msg['role'] == 'user':
                    st.chat_message("user").write(msg['content'])
                else:
                    st.chat_message("assistant").write(msg['content'])
        
        # Input de mensagem
        user_input = st.chat_input("Digite sua pergunta sobre a imagem...")
        
        if user_input and img:
            # Adicionar mensagem do usuário
            st.session_state['chat_messages'].append({"role": "user", "content": user_input})
            
            # Preparar imagem
            base64_data, _ = prepare_image_for_api(img)
            
            # Reutilizar assistente do session_state ou criar novo
            need_new_session = (
                'chat_assistant' not in st.session_state
                or st.session_state.get('chat_image_path') != str(selected_image_path)
                or st.session_state.get('chat_model_id') != selected_model_id
            )
            
            if need_new_session:
                assistant = create_assistant(selected_model_id, selected_model_type)
                assistant.start_session(base64_data, str(selected_image_path))
                st.session_state['chat_assistant'] = assistant
                st.session_state['chat_image_path'] = str(selected_image_path)
                st.session_state['chat_model_id'] = selected_model_id
                # Restaurar histórico anterior (exceto a mensagem recém-adicionada)
                for msg in st.session_state['chat_messages'][:-1]:
                    assistant.session.add_message(msg['role'], msg['content'])
            else:
                assistant = st.session_state['chat_assistant']
            
            # Gerar resposta com streaming
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                include_image = len([m for m in st.session_state['chat_messages'] if m['role'] == 'user']) <= 1
                
                for chunk in assistant.chat_stream(user_input, include_image=include_image):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "|")
                
                response_placeholder.markdown(full_response)
            
            # Salvar resposta
            st.session_state['chat_messages'].append({"role": "assistant", "content": full_response})
            st.rerun()


def render_detection_panel(images: list):
    """Painel de detecção de objetos."""
    st.markdown("### 🎯 Detecção de objetos (YOLO11)")
    
    if not YOLO_AVAILABLE:
        st.error("❌ YOLO11 não disponível")
        st.info("Instale com: `pip install ultralytics`")
        return
    
    if not images:
        st.warning("⚠️ Nenhuma imagem disponível")
        return
    
    # 1. Parâmetros da página
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Seleção de imagem
        image_names = [img.name for img in images]
        selected_image_name = st.selectbox("🖼️ Selecione a imagem:", image_names, key="detect_image")
        selected_image_path = images[image_names.index(selected_image_name)]
        
        st.markdown("**Configurações do Modelo**")
        
        model_name = st.selectbox(
            "Modelo YOLO",
            options=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
            format_func=lambda x: {
                "yolo11n": "YOLO11-N (Nano/Super Rápido)",
                "yolo11s": "YOLO11-S (Rápido)",
                "yolo11m": "YOLO11-M (Balanceado)",
                "yolo11l": "YOLO11-L (Preciso)",
                "yolo11x": "YOLO11-X (Máxima Precisão)"
            }.get(x, x),
            index=1,
            key="yolo_model"
        )
        
        # Confiança mínima
        conf_thresh = st.slider("Confiança mínima:", 0.1, 0.9, 0.25, 0.05, key="yolo_conf")
        
        # Filtro de classes
        st.markdown("**Filtrar classes:**")
        filter_classes = st.multiselect(
            "Classes de interesse:",
            options=list(FORENSIC_CLASSES.keys()),
            format_func=lambda x: FORENSIC_CLASSES.get(x, x),
            default=[],
            key="yolo_classes"
        )
        
        # Overlay toggles
        st.markdown("**Camadas visuais:**")
        show_yolo_overlay = st.checkbox("📦 Mostrar bounding boxes do YOLO", value=True, key="show_yolo")
        show_ocr_overlay = False  # OCR overlay removido (engines locais removidos)
        
        # Botão de detecção
        run_detection = st.button("🔍 Detectar objetos", type="primary", width='stretch')
    
    with col2:
        # Carregar imagem
        img = load_image(selected_image_path)
        
        if img and run_detection:
            with st.spinner(f"🔄 Carregando {model_name}..."):
                try:
                    detector = ObjectDetector(model_name=model_name, confidence_threshold=conf_thresh)
                    if detector:
                        classes_arg = filter_classes if filter_classes else None
                        
                        if show_yolo_overlay:
                            annotated_img, result = detector.detect_and_draw(img, classes_filter=classes_arg)
                        else:
                            result = detector.detect(img, classes_filter=classes_arg)
                            annotated_img = img.copy()
                        
                        # (OCR overlay removido - engines locais não disponíveis)
                        
                        # Mostrar imagem com caixas
                        st.image(annotated_img, caption="Objetos Detectados", width='stretch')
                        
                        # Sumário
                        if result.total_objects > 0:
                            st.markdown("#### 📦 Objetos detectados")
                            
                            summary = result.get_summary()
                            cols = st.columns(min(len(summary), 4))
                            
                            for i, (cls, count) in enumerate(sorted(summary.items(), key=lambda x: -x[1])):
                                icon = str(FORENSIC_CLASSES.get(cls, cls))
                                with cols[i % len(cols)]:
                                    st.metric(icon, count)
                            
                            # Tabela detalhada
                            with st.expander("📋 Detalhes"):
                                for det in result.detections:
                                    st.write(f"- **{det.class_name}**: {det.confidence:.0%} @ ({det.center[0]}, {det.center[1]})")
                            
                            # Download do resultado
                            st.download_button(
                                "📥 Baixar resumo",
                                data=detector.get_forensic_summary(result),
                                file_name=f"deteccao_{selected_image_name}.md",
                                mime="text/markdown"
                            )
                            st.markdown(f"#### Resultados\n{detector.get_forensic_summary(result)}")
                            
                            # Auto-Crop: galeria de recortes
                            crops = detector.extract_crops(img, result)
                            if crops:
                                st.markdown("#### ✂️ Recortes de objetos detectados")
                                crop_cols = st.columns(min(len(crops), 4))
                                for j, (crop_img, crop_det) in enumerate(crops[:12]):
                                    with crop_cols[j % len(crop_cols)]:
                                        st.image(
                                            crop_img,
                                            caption=f"{crop_det.class_name} ({crop_det.confidence:.0%})",
                                            width='stretch'
                                        )
                        else:
                            st.info("Nenhum objeto retornado com essa confiança.")
                            
                except Exception as e:
                    st.error(f"❌ Erro ao detectar objetos: {str(e)}")
        
        elif img:
            st.image(img, caption=selected_image_name, width='stretch')
            st.caption("👆 Clique em 'Detectar objetos' para iniciar")


def render_ela_panel(images: list):
    """Painel de Error Level Analysis (ELA) para detecção de manipulação."""
    st.markdown("### 🔬 Detecção de manipulação (ELA)")

    if not ELA_AVAILABLE:
        st.error("❌ Módulo ELA não disponível.")
        st.info("Verifique se o arquivo `ela_analyzer.py` está no projeto.")
        return

    if not images:
        st.warning("⚠️ Nenhuma imagem disponível")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        image_names = [img.name for img in images]
        selected_name = st.selectbox("🖼️ Selecione a imagem:", image_names, key="ela_image")
        selected_path = images[image_names.index(selected_name)]

        st.markdown("**Configurações ELA**")

        quality = st.slider(
            "Qualidade JPEG de recompressão:",
            min_value=90, max_value=100, value=95, step=1,
            key="ela_quality",
            help="Qualidade usada na recompressão. 95 é o padrão. Valores mais baixos amplificam diferenças."
        )

        scale = st.slider(
            "Fator de amplificação:",
            min_value=1, max_value=50, value=15, step=1,
            key="ela_scale",
            help="Amplifica a diferença para visualização. 15 é o padrão."
        )

        view_mode = st.radio(
            "Modo de visualização:",
            options=["ELA Amplificado", "Mapa de Calor", "Overlay sobre original"],
            key="ela_view_mode"
        )

        if view_mode == "Overlay sobre original":
            overlay_alpha = st.slider(
                "Opacidade do overlay:", 0.1, 0.9, 0.5, 0.05,
                key="ela_alpha"
            )

        run_ela = st.button("🔬 Executar ELA", type="primary", width='stretch')

    with col2:
        img = load_image(selected_path)

        if img and run_ela:
            with st.spinner("🔄 Executando Error Level Analysis..."):
                try:
                    analyzer = ELAAnalyzer(quality=quality, scale=scale)
                    result = analyzer.analyze(img)

                    # Visualização
                    if view_mode == "Mapa de Calor":
                        display_img = analyzer.generate_heatmap(result.ela_image)
                        if display_img is None:
                            display_img = result.ela_image
                        caption = "Mapa de Calor ELA (vermelho = mais erro)"
                    elif view_mode == "Overlay sobre original":
                        heatmap = analyzer.generate_heatmap(result.ela_image)
                        overlay_src = heatmap if heatmap else result.ela_image
                        display_img = analyzer.overlay(img, overlay_src, overlay_alpha)
                        caption = f"Overlay ELA (α={overlay_alpha:.0%})"
                    else:
                        display_img = result.ela_image
                        caption = f"ELA amplificado ({scale}x)"

                    st.image(display_img, caption=caption, width='stretch')

                    # Veredicto
                    verdict_colors = {
                        "sem_indicios": "green",
                        "inconclusivo": "orange",
                        "suspeito": "red",
                        "indeterminado": "gray",
                    }
                    verdict_labels = {
                        "sem_indicios": "Sem indicios de manipulacao",
                        "inconclusivo": "Inconclusivo",
                        "suspeito": "Possivel manipulacao detectada",
                        "indeterminado": "Indeterminado",
                    }
                    color = verdict_colors.get(result.verdict, "gray")
                    label = verdict_labels.get(result.verdict, result.verdict)

                    st.markdown(
                        f'<div style="padding:12px;border-left:4px solid {color};'
                        f'background:rgba(0,0,0,0.05);border-radius:4px;margin:8px 0">'
                        f'<strong style="color:{color};font-size:1.1em">{label}</strong><br>'
                        f'<span style="font-size:0.9em">{result.detail}</span></div>',
                        unsafe_allow_html=True,
                    )

                    # Métricas
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Erro Máx", f"{result.max_error:.1f}")
                    m2.metric("Erro Médio", f"{result.mean_error:.2f}")
                    m3.metric("Desvio Padrão", f"{result.std_error:.2f}")
                    m4.metric("Pixels Suspeitos", f"{result.suspicious_percent:.2f}%")

                    if result.hotspot_count > 0:
                        st.info(f"🎯 {result.hotspot_count} região(ões) com concentração de erro detectada(s).")

                    # Detalhes expandíveis
                    with st.expander("📋 Relatório completo ELA"):
                        st.code(result.get_summary(), language="text")
                        st.markdown(f"""
**Como interpretar:**
- **Imagem uniforme (toda escura ou cinza):** Sem evidências de manipulação localizada.
- **Regiões brilhantes/coloridas concentradas:** Podem indicar edição (clone, colagem, inpainting).
- **Bordas brilhantes generalizadas:** Normal - bordas com alto contraste geram mais erro.
- **Qualidade JPEG:** Imagens salvas várias vezes em JPEG podem gerar falsos positivos.

**Parametros usados:** Qualidade={result.quality_used}, Amplificacao={result.scale_factor}x
""")

                    # Download
                    st.download_button(
                        "📥 Baixar relatório ELA",
                        data=result.get_summary(),
                        file_name=f"ela_{selected_name}.txt",
                        mime="text/plain",
                    )

                except Exception as e:
                    st.error(f"❌ Erro na análise ELA: {e}")

        elif img:
            st.image(img, caption=selected_name, width='stretch')
            st.caption("👆 Clique em 'Executar ELA' para analisar a imagem")


def render_video_panel(input_dir: Path):
    """Painel para extração de frames de vídeo."""
    st.markdown("### 🎞️ Extrator de keyframes de vídeo")
    st.info("Faça upload de um vídeo e extraia frames espaçados no tempo para salvar na pasta de Imagens de Entrada. O BatchAnalyzer processará essas imagens depois.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**1. Configurações de Extração**")
        frame_interval = st.number_input(
            "Intervalo de Frames a extrair a cada 1 frame salvo:", 
            min_value=1, max_value=3000, value=30, step=10,
            help="Em um vídeo de 30 FPS, o valor 30 significa extrair 1 frame a cada segundo de vídeo."
        )
        
        uploaded_video = st.file_uploader("📤 Faça o upload do vídeo (MP4/AVI)", type=["mp4", "avi", "mov", "mkv"])
        
        extract_btn = st.button("🚀 Extrair keyframes", type="primary", width='stretch', disabled=not uploaded_video)
        
    with col2:
        if uploaded_video and extract_btn:
            if not VIDEO_PROCESSOR_AVAILABLE:
                st.error("❌ Módulo VideoProcessor não disponível. Instale as dependências: `pip install opencv-python`")
                return
            
            with st.spinner("Descompactando vídeo e exportando JPGs para o 'input_dir'..."):
                try:
                    # Salva video localmente provisorio
                    temp_video_path = input_dir / uploaded_video.name
                    with open(temp_video_path, "wb") as f:
                        f.write(uploaded_video.read())
                    
                    processor = VideoProcessor(output_dir=input_dir, frame_interval=frame_interval)
                    extracted = processor.extract_keyframes(str(temp_video_path))
                    
                    # Remover o video original do input_dir para não dar erro no BatchAnalyzer (que só lê jpg/png)
                    temp_video_path.unlink()
                    
                    st.success(f"✅ Sucesso! {len(extracted)} keyframes foram salvos na pasta de imagens.")
                    st.info("➡️ Agora clique na aba 'Análise' para processar esse lote com os modelos LLM e YOLO11.")
                    
                except Exception as e:
                    st.error(f"Erro no extrator de vídeo: {e}") 


def render_reports_viewer(output_dir: Path):
    """Visualizador de relatórios com Busca Semântica Avançada."""
    st.markdown("### 📑 Relatórios e busca semântica (ChromaDB)")
    
    reports = list(output_dir.glob("*.md"))
    if not reports:
        st.info("Nenhum relatório encontrado.")
        return

    db_path = output_dir / ".chroma_db"

    def get_semantic_engine():
        if not SemanticSearchEngine:
            raise RuntimeError("Semantic Search indisponível. Instale: pip install chromadb")

        engine_key = f"semantic_search_engine::{db_path}"
        engine = st.session_state.get(engine_key)
        if engine is None:
            engine = SemanticSearchEngine(db_path=db_path)
            st.session_state[engine_key] = engine
        return engine
            
    # Layout Principal
    tab_busca, tab_lista = st.tabs(["🔎 Busca semântica (IA)", "📂 Todos os arquivos"])
    
    with tab_busca:
        st.write("Digite o que você procura. A IA buscará o contexto em todos os laudos gerados (Ex: 'Foto noturna com carro prata').")
        st.caption("Busca semântica 100% local via Ollama embeddings.")
        query = st.text_input("Buscar nos relatórios:", key="semantic_query")
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("Buscar", type="primary"):
                if not query:
                    st.warning("Digite um termo para buscar nos laudos.")
                else:
                    try:
                        engine = get_semantic_engine()
                        with st.spinner("Vetorizando novos relatórios e buscando contexto..."):
                            # Indexa arquivos faltantes rapidamente antes da busca
                            engine.index_reports(output_dir)
                            resultados = engine.search(query, top_k=5)
                            
                            if resultados:
                                st.success(f"Encontrados {len(resultados)} relatórios relevantes!")
                                for idx, res in enumerate(resultados):
                                    with st.expander(f"📄 {res['filename']} (Score Distância: {res['distance']:.3f})"):
                                        st.write(f"...{res['snippet']}")
                                        st.caption(f"Arquivo: {res['path']}")
                                        if st.button(f"Abrir {res['filename']}", key=f"semantic_open_{idx}"):
                                            st.session_state["report_to_open"] = res["filename"]
                                            st.rerun()
                            else:
                                st.warning("Nenhum resultado similar encontrado.")
                    except Exception as e:
                        st.error(str(e))
                    
        with col_btn2:
            if st.button("🔄 Forçar re-indexação", help="Atualiza o ChromaDB com todos os arquivos MD da pasta"):
                try:
                    engine = get_semantic_engine()
                    with st.spinner("Indexando..."):
                        count = engine.index_reports(output_dir)
                        st.success(f"{count} novos relatórios foram indexados!")
                except Exception as e:
                    st.error(str(e))
                        
    with tab_lista:
        # Visualizador Padrão Antigo
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"**Arquivos ({len(reports)})**")
            report_names = [r.name for r in sorted(reports, key=lambda x: x.stat().st_mtime, reverse=True)]

            pending_report = st.session_state.pop("report_to_open", None)
            if pending_report in report_names:
                st.session_state["view_report"] = pending_report
            
            selected_report = st.selectbox(
                "Selecione para visualizar",
                options=report_names,
                label_visibility="collapsed",
                key="view_report"
            )
            
            # Botões de ação em massa
            st.markdown("---")
            if CONSOLIDATED_PDF_AVAILABLE:
                if st.button("📕 Gerar PDF consolidado", width='stretch'):
                    with st.spinner("Gerando PDF consolidado..."):
                        try:
                            report_data_list = []
                            for rp in reports:
                                content = rp.read_text(encoding='utf-8')
                                report_data_list.append(ReportData(
                                    image_name=rp.stem.rsplit('_', 1)[0] if '_' in rp.stem else rp.stem,
                                    image_path=str(rp),
                                    model=rp.stem.rsplit('_', 1)[-1] if '_' in rp.stem else "unknown",
                                    analysis_mode="geral",
                                    ocr_engine="unknown",
                                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    processing_time=0.0,
                                    dimensions=(0, 0),
                                    file_size=rp.stat().st_size,
                                    ocr_result="",
                                    analysis_content=content,
                                    hash_md5="",
                                    hash_sha256=""
                                ))
                            pdf_path = output_dir / "consolidado.pdf"
                            generate_consolidated_pdf(report_data_list, pdf_path)
                            st.success(f"PDF gerado: `{pdf_path.name}`")
                            with open(pdf_path, "rb") as f:
                                st.download_button("📥 Baixar PDF", data=f.read(), file_name="consolidado.pdf", mime="application/pdf")
                        except Exception as ex:
                            st.error(f"Erro ao gerar PDF: {ex}")
            
            if st.button("🗑️ Apagar selecionado", type="secondary", width='stretch'):
                p = output_dir / selected_report
                if p.exists():
                    p.unlink()
                    st.success("Removido!")
                    st.rerun()
                    
        with col2:
            if selected_report:
                report_path = output_dir / selected_report
                if report_path.exists():
                    with open(report_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    report_payload = None
                    json_report_path = report_path.with_suffix('.json')
                    if json_report_path.exists():
                        try:
                            with open(json_report_path, "r", encoding="utf-8") as f:
                                report_payload = json.load(f)
                        except Exception as ex:
                            st.warning(f"Não foi possível ler o JSON estruturado: {ex}")

                    if report_payload:
                        tab_markdown, tab_structured = st.tabs(["📝 Markdown", "🧠 Estruturado"])
                    else:
                        tab_markdown = st.container()
                        tab_structured = None
                    
                    with tab_markdown:
                        st.markdown("**Conteúdo**")
                        st.markdown(
                            f'<div class="report-container">{content}</div>',
                            unsafe_allow_html=True
                        )

                    if tab_structured and report_payload:
                        with tab_structured:
                            image_payload = report_payload.get("image", {})
                            analysis_payload = report_payload.get("analysis", {})
                            pre_payload = analysis_payload.get("pre_analysis", {})
                            post_payload = analysis_payload.get("post_processing", {})

                            col_meta1, col_meta2, col_meta3 = st.columns(3)
                            with col_meta1:
                                st.metric("Modelo", analysis_payload.get("model", "-"))
                            with col_meta2:
                                st.metric("Modo", analysis_payload.get("mode", "-"))
                            with col_meta3:
                                st.metric("OCR", analysis_payload.get("ocr_engine", "-"))

                            st.markdown("#### Pré-análise")
                            st.markdown("**OCR**")
                            st.code(analysis_payload.get("ocr_result") or "[Nenhum texto detectado]", language="text")
                            st.markdown("**EXIF e GPS**")
                            st.code(pre_payload.get("exif_result") or "[EXIF não avaliado]", language="text")
                            st.markdown("**YOLO11**")
                            st.code(pre_payload.get("yolo_result") or "[YOLO não executado]", language="text")
                            st.markdown("**Qualidade**")
                            st.code(pre_payload.get("quality_result") or "[Qualidade não avaliada]", language="text")

                            if post_payload:
                                st.markdown("#### Pós-processamento")

                                summary_text = post_payload.get("summary")
                                if summary_text:
                                    st.info(summary_text)

                                classification = post_payload.get("classification") or {}
                                validation = post_payload.get("validation") or {}
                                entities = post_payload.get("entities") or {}
                                timeline = post_payload.get("timeline") or []

                                col_post1, col_post2 = st.columns(2)
                                with col_post1:
                                    st.markdown("**Classificação**")
                                    if classification:
                                        st.json(classification)
                                    else:
                                        st.caption("Sem classificação estruturada.")
                                with col_post2:
                                    st.markdown("**Validação OCR x LLM**")
                                    if validation:
                                        st.json(validation)
                                    else:
                                        st.caption("Sem validação estruturada.")

                                entity_items = {key: value for key, value in entities.items() if value}
                                if entity_items:
                                    st.markdown("**Entidades extraídas**")
                                    st.json(entity_items)

                                if timeline:
                                    st.markdown("**Linha do tempo**")
                                    for item in timeline:
                                        timestamp = item.get("timestamp") or item.get("parsed_dt") or "Sem data"
                                        description = item.get("description") or "Sem descrição"
                                        st.write(f"- {timestamp}: {description}")
                            else:
                                st.caption("Nenhum pós-processamento estruturado disponível neste relatório.")

                            with st.expander("Ver JSON bruto"):
                                st.json(report_payload)
                    
                    # Download
                    download_col1, download_col2 = st.columns(2)
                    with download_col1:
                        st.download_button(
                            "📥 Baixar Markdown",
                            data=content,
                            file_name=selected_report,
                            mime="text/markdown",
                            key="btn_download_report"
                        )
                    with download_col2:
                        if json_report_path.exists():
                            st.download_button(
                                "📥 Baixar JSON",
                                data=json_report_path.read_text(encoding="utf-8"),
                                file_name=json_report_path.name,
                                mime="application/json",
                                key="btn_download_report_json"
                            )
                    
                    # Re-análise sem reprocessar OCR/YOLO
                    if CACHE_AVAILABLE:
                        # Tentar extrair nome da imagem do relatório
                        img_name = selected_report.rsplit('_', 1)[0] if '_' in selected_report else ""
                        if img_name:
                            input_dir_path = Path(st.session_state.get("input_dir", "./imagens_entrada"))
                            matching = list(input_dir_path.glob(f"{img_name}.*"))
                            if matching:
                                image_path = matching[0]
                                cached = get_cache_manager().get_cached_intermediate(image_path)
                                if cached:
                                    with st.expander("Re-analisar com outro prompt"):
                                        st.caption("Usa OCR, YOLO e Quality ja processados - apenas re-executa o LLM")
                                        new_mode = st.selectbox(
                                            "Novo modo de análise:",
                                            options=list(ANALYSIS_PROMPTS.keys()),
                                            format_func=lambda x: ANALYSIS_PROMPTS[x]["name"],
                                            key="reanalyze_mode"
                                        )
                                        reanalyze_model = st.selectbox(
                                            "Modelo:",
                                            options=[OPENAI_MODEL] + ([m for m in (st.session_state.get("selected_models") or []) if isinstance(m, tuple) and m[1] == "ollama"] if st.session_state.get("selected_models") else []),
                                            key="reanalyze_model_sel"
                                        )
                                        if st.button("🔄 Re-analisar", key="btn_reanalyze"):
                                            new_prompt = resolve_prompt(
                                                new_mode,
                                                ocr_result=cached["ocr_result"],
                                                yolo_result=cached["yolo_result"],
                                                quality_result=cached["quality_result"],
                                                exif_data=cached["exif_data"]
                                            )
                                            img = load_image(image_path)
                                            if img:
                                                b64, _ = prepare_image_for_api(img)
                                                model_id = reanalyze_model if isinstance(reanalyze_model, str) else reanalyze_model[0]
                                                with st.spinner(f"Re-analisando com {model_id}..."):
                                                    if model_id.startswith("gpt"):
                                                        ok, result, t = analyze_with_openai(b64, new_prompt)
                                                    else:
                                                        ok, result, t = analyze_with_ollama(b64, new_prompt, model_id)
                                                if ok:
                                                    st.success(f"✅ Re-análise concluída em {t:.1f}s")
                                                    st.markdown(result)
                                                else:
                                                    st.error(f"Erro: {result}")
                                                if not model_id.startswith("gpt"):
                                                    unload_ollama_models([model_id])


def render_dashboard_panel():
    """Painel de métricas e histórico de análises."""
    st.markdown("### 📊 Dashboard de métricas")
    
    if not CACHE_AVAILABLE:
        st.warning("Cache não disponível. O dashboard requer o módulo cache_manager.")
        return
    
    cache = get_cache_manager()
    stats = cache.get_cache_stats()
    history = cache.get_stats_history()
    
    # Métricas resumo
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analises em cache", stats["total_entries"])
    with col2:
        st.metric("Tamanho total", f"{stats['total_size_mb']:.1f} MB")
    with col3:
        st.metric("Analises no historico", len(history))
    with col4:
        cache_hits = sum(1 for h in history if h.get("cache_hit"))
        st.metric("Cache hits", cache_hits)
    
    if not history:
        st.info("Nenhuma análise registrada ainda. Execute análises para popular o dashboard.")
        return
    
    st.markdown("---")
    
    # Tempo médio por modelo
    st.markdown("#### ⏱️ Tempo Médio por Modelo")
    model_times = {}
    for h in history:
        m = h.get("model", "?")
        t = h.get("processing_time", 0)
        if not h.get("cache_hit"):
            model_times.setdefault(m, []).append(t)
    
    if model_times:
        chart_data = {
            "Modelo": [],
            "Tempo Médio (s)": [],
            "Análises": [],
        }
        for model, times in sorted(model_times.items()):
            chart_data["Modelo"].append(model)
            chart_data["Tempo Médio (s)"].append(round(sum(times) / len(times), 2))
            chart_data["Análises"].append(len(times))
        
        st.bar_chart(chart_data, x="Modelo", y="Tempo Médio (s)")
    
    # Análises por dia
    st.markdown("#### Analises por dia")
    day_counts = {}
    for h in history:
        day = h.get("date", "?")
        day_counts[day] = day_counts.get(day, 0) + 1
    
    if day_counts:
        st.bar_chart({"Data": list(day_counts.keys()), "Análises": list(day_counts.values())}, x="Data", y="Análises")
    
    # Distribuição por modelo (tabela)
    st.markdown("#### Distribuicao por modelo")
    if stats["by_model"]:
        st.table({"Modelo": list(stats["by_model"].keys()), "Análises": list(stats["by_model"].values())})


def render_compare_panel(images: list, selected_models: list, analysis_mode: str):
    """Painel comparador A/B para analisar a mesma imagem com 2 modelos lado a lado."""
    st.markdown("### ⚖️ Comparador A/B de modelos")
    
    if not images:
        st.warning("Nenhuma imagem disponivel para comparacao.")
        return
    
    # Seleção de imagem
    image_names = [img.name for img in images]
    selected_name = st.selectbox("Imagem para comparar:", image_names, key="compare_image")
    selected_path = images[image_names.index(selected_name)]
    
    # Construir lista de modelos disponíveis para seleção
    available_models = []
    
    # OpenAI
    if check_openai_status():
        available_models.append((OPENAI_MODEL, "openai"))
    
    # Modelos Ollama
    ollama_ok, ollama_models_list = check_ollama_status()
    if ollama_ok and ollama_models_list:
        vision = [
            m for m in ollama_models_list
            if any(vp in m.split(":")[0].lower() for vp in OLLAMA_VISION_PREFIXES)
        ]
        for m in sorted(vision):
            available_models.append((m, "ollama"))
    
    if len(available_models) < 2:
        st.warning("Sao necessarios pelo menos 2 modelos disponiveis para comparacao.")
        return
    
    model_labels = [f"{'OpenAI' if t == 'openai' else 'Ollama'} - {m}" for m, t in available_models]
    
    col_a, col_b = st.columns(2)
    with col_a:
        idx_a = st.selectbox("Modelo A:", range(len(available_models)), format_func=lambda i: model_labels[i], key="cmp_model_a")
    with col_b:
        default_b = min(1, len(available_models) - 1)
        idx_b = st.selectbox("Modelo B:", range(len(available_models)), format_func=lambda i: model_labels[i], index=default_b, key="cmp_model_b")
    
    model_a = available_models[idx_a]
    model_b = available_models[idx_b]
    
    if model_a == model_b:
        st.warning("Selecione modelos diferentes para comparar.")
        return
    
    # Mostrar preview
    img = load_image(selected_path)
    if img:
        thumb = img.copy()
        thumb.thumbnail((400, 400))
        st.image(thumb, caption=selected_name, width='content')
    
    run_compare = st.button("⚖️ Comparar", type="primary", width='stretch', key="btn_compare")
    
    if run_compare and img:
        base64_data, _ = prepare_image_for_api(img)
        system_prompt = resolve_prompt(analysis_mode)
        
        col_ra, col_rb = st.columns(2)
        
        with col_ra:
            st.markdown(f"#### {model_labels[idx_a]}")
            with st.spinner(f"Analisando com {model_a[0]}..."):
                if model_a[1] == "openai":
                    ok, result_a, time_a = analyze_with_openai(base64_data, system_prompt)
                else:
                    ok, result_a, time_a = analyze_with_ollama(base64_data, system_prompt, model_a[0])
            
            if ok:
                st.caption(f"⏱️ {time_a:.1f}s")
                st.markdown(result_a)
            else:
                st.error(f"Erro: {result_a}")
        
        with col_rb:
            st.markdown(f"#### {model_labels[idx_b]}")
            with st.spinner(f"Analisando com {model_b[0]}..."):
                if model_b[1] == "openai":
                    ok, result_b, time_b = analyze_with_openai(base64_data, system_prompt)
                else:
                    ok, result_b, time_b = analyze_with_ollama(base64_data, system_prompt, model_b[0])
            
            if ok:
                st.caption(f"⏱️ {time_b:.1f}s")
                st.markdown(result_b)
            else:
                st.error(f"Erro: {result_b}")

        # Descarregar modelos Ollama usados na comparação
        ollama_to_unload = [m[0] for m in (model_a, model_b) if m[1] == "ollama"]
        if ollama_to_unload:
            unload_ollama_models(ollama_to_unload)


def render_about_panel():
    """Painel sobre o sistema."""
    st.markdown("""
    ## 👁️ Vision Analyzer v4.0
    
    Sistema avançado de análise individual e em lote usando múltiplos modelos de IA.
    
    ### ✨ Funcionalidades
    
    - **16 formatos** de imagem suportados (JPG, PNG, HEIC, RAW, etc.)
    - **2 engines de OCR**: GLM OCR (glm-ocr:bf16) e OLMoOCR2 (richardyoung/olmocr2:7b-q8)
    - **GPT-5.4-mini padrão** + modelos Ollama locais otimizados por GPU
    - **Perfis de GPU**: Auto-detecção de VRAM (4/6/8/16/24/32 GB) com modelos recomendados
    - **8 modos de análise**: Geral, Profunda, Forense, Documentos, Screenshots/Telas, Veículos, Pessoas, Acessibilidade
    - **5 formatos de exportação**: MD, JSON, HTML, PDF (com Markdown), DOCX
    - **Análise individual dedicada**: Executa uma imagem sem cair no fluxo de lote
    - **Comparador A/B**: Análise lado-a-lado com 2 modelos
    - **Dashboard de Métricas**: Tempo por modelo, cache hits, análises por dia
    - **Detecção de Manipulação (ELA)**: Error Level Analysis com mapa de calor e veredicto
    - **Upload Drag-and-Drop**: Arraste imagens direto para a interface
    - **Auto-Crop YOLO**: Galeria de recortes automáticos das detecções
    - **Overlay Visual**: YOLO boxes + regiões OCR sobrepostos na imagem
    - **Re-análise Inteligente**: Re-execute LLM sem reprocessar OCR/YOLO
    - **Cache inteligente** com dados intermediários para re-análise
    - **Processamento paralelo** com até 8 workers
    
    ### 🧭 Modos de Análise
    
    | Modo | Descrição |
    |------|-----------|
    | 📷 **Geral** | Descricao para acessibilidade e documentacao |
    | 🧠 **Profunda** | Chain of Thought com semiotica e cinesica |
    | 🔍 **Forense** | Laudo pericial para investigacao policial |
    | 📄 **Documentos** | Extracao de campos de documentos |
    | 🖥️ **Screenshots/Telas** | Conversas, paginas web, emails e apps com foco em hierarquia e integridade |
    | 🚗 **Veiculos** | Identificacao de marca, modelo e placa |
    | 👤 **Pessoas** | Descricao de caracteristicas fisicas |
    | ♿ **Acessibilidade** | Alt-text otimizado para leitores de tela |
    
    ### 🎯 Perfis de GPU
    
    | VRAM | Modelos |
    |------|---------|
    | **4 GB** | Qwen3.5 2B, Qwen3-VL 2B |
    | **6 GB** | Qwen3.5 4B, Qwen3-VL 4B |
    | **8 GB** | Qwen3-VL 8B, Gemma4 E2B |
    | **16 GB** | Qwen3.5 9B Q8, Gemma4 E4B Q8 |
    | **24 GB** | Qwen3-VL 32B, Gemma4 31B |
    | **32 GB** | Modelos do perfil 24 GB com margem para OCR simultaneo |
    
    ### 🆕 Novidades v4.0
    
    - Tema dark na interface Streamlit
    - Hash SHA-256 adicionado aos metadados
    - Perfis de GPU com deteccao via CUDA e fallback por nvidia-smi
    - Janela de contexto do Ollama otimizada para 4096 e 8192 tokens
    - Exportacao em Markdown, JSON, HTML, PDF e DOCX
    - Comparador A/B, dashboard, ELA e busca semantica
    
    ---
    
    **Desenvolvido com** 🐍 Python + 🎈 Streamlit + 🤖 AI
    """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Função principal."""
    
    apply_custom_css()
    
    # Sidebar
    selected_models, ocr_engine, input_dir, output_dir, analysis_mode, workers = render_sidebar()
    
    # Header
    render_header()
    
    # Encontrar imagens (compartilhado entre abas)
    images = find_images(input_dir)
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📷 Análise", 
        "💬 Chat", 
        "🎯 Detecção",
        "🔬 ELA",
        "⚖️ Comparar",
        "📊 Dashboard",
        "🎞️ Vídeo HD",
        "📑 Relatórios", 
        "⚙️ Configurações", 
        "ℹ️ Sobre"
    ])
    
    with tab1:
        # Upload de imagens (Drag-and-Drop)
        uploaded_files = st.file_uploader(
            "📤 Arraste imagens ou clique para enviar",
            type=[ext.lstrip('.') for ext in SUPPORTED_EXTENSIONS if ext not in RAW_EXTENSIONS],
            accept_multiple_files=True,
            key="file_uploader",
            help="As imagens serão salvas na pasta de entrada"
        )
        if uploaded_files:
            input_dir.mkdir(parents=True, exist_ok=True)
            saved_count = 0
            for uploaded_file in uploaded_files:
                dest = input_dir / uploaded_file.name
                if not dest.exists():
                    dest.write_bytes(uploaded_file.getbuffer())
                    saved_count += 1
            if saved_count > 0:
                st.success(f"✅ {saved_count} imagem(ns) salva(s) em `{input_dir}`")
                st.rerun()
            # Refresh images list after upload
            images = find_images(input_dir)
        
        # Galeria
        render_image_gallery(images)
        
        # Painel de análise
        render_analysis_panel(images, selected_models, ocr_engine, output_dir, analysis_mode, workers)
    
    with tab2:
        render_chat_panel(images, selected_models)
    
    with tab3:
        render_detection_panel(images)
    
    with tab4:
        render_ela_panel(images)
    
    with tab5:
        render_compare_panel(images, selected_models, analysis_mode)
    
    with tab6:
        render_dashboard_panel()
        
    with tab7:
        render_video_panel(input_dir)
    
    with tab8:
        render_reports_viewer(output_dir)
    
    with tab9:
        render_settings_panel()
    
    with tab10:
        render_about_panel()


if __name__ == "__main__":
    main()