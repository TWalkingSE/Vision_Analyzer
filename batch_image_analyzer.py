#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖼️ Batch Image Analyzer - Sistema de Análise de Imagens em Lote
================================================================
Análise de imagens usando múltiplos modelos de IA (OpenAI + Ollama)
com suporte a RAW, HEIC e OCR integrado.

Author: Vision Analyzer Pro
Version: 1.0.0
Python: 3.10+
"""

import sys
import logging
from pathlib import Path
from typing import Optional

from runtime_config import (
    GPU_MODEL_PROFILES,
    OPENAI_MODEL,
    SUPPORTED_EXTENSIONS,
    detect_vram_gb,
    get_recommended_gpu_profile,
)
from batch_checkpoint import (
    BatchCheckpointManager,
    build_batch_job_config,
    build_batch_signature,
    get_default_checkpoint_path,
)

# O console padrão do Windows usa cp1252, que não codifica os emojis presentes nos nomes
# dos modos e nas mensagens de log — sem isto, até `--help` aborta com UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DEPENDÊNCIAS E IMPORTS
# ============================================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("⚠️ python-dotenv não instalado. Usando variáveis de ambiente do sistema.")

try:
    from prompt_templates import get_prompt_manager
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    logger.warning("⚠️ Prompt Templates não disponível. Usando prompts locais reduzidos.")

# analysis_pipeline concentra as sondas de dependência (Pillow, HEIF, RAW, YOLO, OpenAI,
# Ollama, pós-processamento). Elas eram redeclaradas aqui e ficaram apenas como variáveis
# de escrita, nunca lidas — a CLI delega tudo ao pipeline.
try:
    from analysis_pipeline import AnalysisPipeline, ImageProcessor
except ImportError as exc:
    logger.error(
        "❌ analysis_pipeline.py não pôde ser importado — ele é o núcleo compartilhado "
        "entre a CLI e a interface, e não há mais caminho alternativo. Erro: %s",
        exc,
    )
    sys.exit(1)

try:
    from duplicate_detector import DuplicateDetector
    DUPLICATE_DETECTOR_AVAILABLE = True
except ImportError:
    DUPLICATE_DETECTOR_AVAILABLE = False
    DuplicateDetector = None

try:
    from batch_summary import BatchSummaryBuilder, report_to_summary_entry
    BATCH_SUMMARY_AVAILABLE = True
except ImportError:
    BATCH_SUMMARY_AVAILABLE = False
    BatchSummaryBuilder = None
    report_to_summary_entry = None


# ============================================================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================================================

# Diretórios padrão
INPUT_DIR = Path("./imagens_entrada")
OUTPUT_DIR = Path("./relatorios_saida")

# Opções de OCR
OCR_ENGINES = ["glm-ocr", "olmocr2", "none"]


# ============================================================================
# PROMPTS DE SISTEMA
# ============================================================================

# ============================================================================
# PROMPTS DE SISTEMA (fallback mínimo — prompts completos em prompt_templates.py)
# ============================================================================

_FALLBACK_PROMPT = """Analise a imagem a seguir com foco investigativo e descritivo.

**OCR:**
{ocr_result}

**Objetos detectados:**
{yolo_result}

**Qualidade da imagem:**
{quality_result}

**Metadados EXIF:**
{exif_data}

Gere uma análise detalhada em Markdown descrevendo o conteúdo da imagem, elementos relevantes, e quaisquer indícios de manipulação ou artefatos suspeitos.
"""

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

# Prompt padrão (para compatibilidade)
SYSTEM_PROMPT_TEMPLATE = _FALLBACK_PROMPT


def get_available_analysis_modes() -> dict:
    """Retorna modos disponíveis preferindo prompt_templates quando presente."""
    if PROMPTS_AVAILABLE:
        try:
            mgr = get_prompt_manager()
            return {
                key: {
                    "name": prompt.name,
                    "description": prompt.description,
                    "prompt": prompt.prompt,
                }
                for key, prompt in mgr.get_all_prompts().items()
            }
        except Exception as exc:
            logger.warning(f"⚠️ Falha ao carregar prompt_templates: {exc}")

    return ANALYSIS_PROMPTS


# ============================================================================
# PIPELINE COMPARTILHADO
# ============================================================================
# As dataclasses (ImageData/AnalysisResult/ImageAnalysisReport), os clientes de API
# (OpenAIClient/OllamaClient), o ImageProcessor e o ReportGenerator viviam duplicados
# aqui como fallback para o caso de analysis_pipeline nao importar. Esse fallback nunca
# rodava na pratica e ja havia divergido do original (ignorava cache e formatos de
# export, e chamava a OpenAI sempre com o modelo padrao). Fonte unica agora e o
# analysis_pipeline.py.

# ============================================================================
# ANALISADOR PRINCIPAL
# ============================================================================

class BatchImageAnalyzer:
    """Orquestrador principal do pipeline de análise."""
    
    def __init__(
        self, 
        input_dir: Path = INPUT_DIR, 
        output_dir: Path = OUTPUT_DIR,
        analysis_mode: str = "geral",
        ocr_engine: str = "glm-ocr",
        workers: int = 1,
        models: list = None,
        resume: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.analysis_mode = analysis_mode
        self.ocr_engine = ocr_engine
        self.workers = max(1, min(workers, 8))  # Entre 1 e 8 workers
        self.explicit_models = models  # Modelos especificados pelo usuário
        self.resume_checkpoint = resume
        self.processor = ImageProcessor()
        # Pipeline compartilhado: única implementação de inferência, cache e exportação.
        # Mantido no atributo para que o unload da VRAM aconteça uma vez ao fim do lote.
        self.pipeline = self._create_shared_pipeline()


        # Validar modo de análise
        available_modes = get_available_analysis_modes()
        if analysis_mode not in available_modes:
            logger.warning(f"⚠️ Modo '{analysis_mode}' inválido. Usando 'geral'.")
            self.analysis_mode = "geral"
        
        # Validar engine de OCR
        if ocr_engine not in OCR_ENGINES:
            logger.warning(f"⚠️ OCR engine '{ocr_engine}' inválido. Usando 'glm-ocr'.")
            self.ocr_engine = "glm-ocr"
        
        # Estatísticas (thread-safe com lock)
        import threading
        self._stats_lock = threading.Lock()
        self.stats = {
            "total_images": 0,
            "pending_images": 0,
            "processed": 0,
            "failed": 0,
            "reports_generated": 0,
            "resumed_skipped": 0,
            "analysis_mode": self.analysis_mode,
            "ocr_engine": self.ocr_engine,
            "workers": self.workers
        }

    def _create_shared_pipeline(self) -> AnalysisPipeline:
        """Cria uma instância do pipeline compartilhado."""
        return AnalysisPipeline(
            analysis_mode=self.analysis_mode,
            ocr_engine=self.ocr_engine,
            yolo_model="yolo11s",
        )

    def _get_available_models(self) -> list[tuple[str, str]]:
        """Retorna lista de modelos disponíveis: [(nome, tipo)]"""
        # Se o usuário especificou modelos, usar somente esses
        if self.explicit_models:
            return self.explicit_models

        # Padrão: somente GPT-5.4-mini (se disponível)
        models = []

        if self.pipeline.openai_client.is_available():
            models.append((OPENAI_MODEL, "openai"))

        return models

    def _create_checkpoint_manager(
        self,
        selected_models: list[tuple[str, str]],
        export_formats: list[str],
    ) -> BatchCheckpointManager:
        job_config = build_batch_job_config(
            selected_models=selected_models,
            analysis_mode=self.analysis_mode,
            ocr_engine=self.ocr_engine,
            export_formats=export_formats,
            yolo_model="yolo11s",
        )
        return BatchCheckpointManager(
            checkpoint_path=get_default_checkpoint_path(self.output_dir),
            job_signature=build_batch_signature(job_config),
            job_config=job_config,
        )

    def _record_checkpoint_result(
        self,
        checkpoint_manager: Optional[BatchCheckpointManager],
        image_path: Path,
        task_result: dict,
    ) -> None:
        if checkpoint_manager is None:
            return

        checkpoint_manager.record_result(
            image_path=image_path,
            success_count=task_result.get("success", 0),
            failed_count=task_result.get("failed", 0),
            reports=task_result.get("reports", []),
            errors=task_result.get("errors", []),
        )

    def _accumulate_stats(self, task_result: dict, image_path: Path) -> None:
        """Contabiliza o resultado de uma imagem e loga sucessos/erros."""
        if task_result.get("success", 0) > 0:
            with self._stats_lock:
                self.stats["processed"] += 1
                self.stats["reports_generated"] += len(task_result.get("reports", []))
            logger.info(f"   ✅ {image_path.name}: {task_result['success']} relatórios")
        else:
            with self._stats_lock:
                self.stats["failed"] += 1

        for error in task_result.get("errors", []):
            logger.error(f"   ❌ {image_path.name}: {error}")

    def _log_final_summary(self) -> None:
        logger.info("\n" + "=" * 60)
        logger.info("📊 RESUMO FINAL")
        logger.info("=" * 60)
        logger.info(f"   Total de imagens solicitadas: {self.stats['total_images']}")
        logger.info(f"   Imagens pendentes nesta execução: {self.stats['pending_images']}")
        logger.info(f"   Processadas com sucesso: {self.stats['processed']}")
        logger.info(f"   Falhas: {self.stats['failed']}")
        logger.info(f"   Relatórios gerados: {self.stats['reports_generated']}")
        if self.stats["resumed_skipped"] > 0:
            logger.info(f"   Puladas por checkpoint: {self.stats['resumed_skipped']}")
        logger.info(f"   Diretório de saída: {self.output_dir.absolute()}")
        logger.info("=" * 60)
    
    def _process_single_image(
        self,
        image_path: Path,
        index: int,
        total: int,
        selected_models: list[tuple[str, str]],
        pipeline: Optional[AnalysisPipeline] = None,
    ) -> dict:
        """
        Processa uma única imagem (para execução paralela).
        """
        logger.info(f"[{index}/{total}] 📷 Processando: {image_path.name}")
        failed_count = max(len(selected_models), 1)

        # Em modo paralelo cada worker recebe seu próprio pipeline (via `pipeline`);
        # no sequencial reutiliza o do analisador.
        pipeline = pipeline or self.pipeline

        try:
            task_result = pipeline.process_image(
                image_path=image_path,
                selected_models=selected_models,
                output_dir=self.output_dir,
                export_formats=["md"],
                use_cache=False,
            )

            return {
                "image_path": image_path,
                "image": image_path.name,
                "reports": task_result["reports"],
                "errors": task_result["errors"],
                "success": task_result["success"],
                "failed": task_result["failed"],
                "report": task_result.get("report"),
            }
        except Exception as e:
            logger.error(f"❌ Erro processando {image_path.name}: {e}")
            return {
                "image_path": image_path,
                "image": image_path.name,
                "reports": [],
                "errors": [str(e)],
                "success": 0,
                "failed": failed_count,
                "report": None,
            }
    
    def run(self) -> dict:
        """Executa o pipeline completo de análise em lote."""
        logger.info("=" * 60)
        logger.info("🖼️  BATCH IMAGE ANALYZER v1.0")
        logger.info("=" * 60)
        
        # Mostrar modo de análise
        mode_info = ANALYSIS_PROMPTS[self.analysis_mode]
        logger.info(f"📋 Modo de Análise: {mode_info['name']}")
        logger.info(f"   {mode_info['description']}")
        
        # Mostrar OCR engine
        ocr_names = {"glm-ocr": "🔤 GLM OCR (glm-ocr:bf16)", "olmocr2": "📖 OLMoOCR2 (7b-q8)", "none": "❌ Desabilitado"}
        logger.info(f"🔍 OCR Engine: {ocr_names.get(self.ocr_engine, self.ocr_engine)}")
        
        # Mostrar workers
        logger.info(f"⚡ Workers paralelos: {self.workers}")
        
        # Verificar diretórios
        if not self.input_dir.exists():
            logger.info(f"📁 Criando diretório de entrada: {self.input_dir}")
            self.input_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Encontrar imagens
        images = list(self.processor.find_images(self.input_dir))
        self.stats["total_images"] = len(images)
        self.stats["pending_images"] = len(images)
        
        if not images:
            logger.warning(f"⚠️ Nenhuma imagem encontrada em: {self.input_dir}")
            logger.info("   Extensões suportadas: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
            return self.stats
        
        logger.info(f"📷 Encontradas {len(images)} imagens")
        logger.info("-" * 60)
        
        # Verificar modelos disponíveis
        available = self._get_available_models()
        logger.info(f"🤖 Modelos disponíveis: {len(available)}")
        for model, mtype in available:
            logger.info(f"   • {model} ({mtype})")

        checkpoint_manager = self._create_checkpoint_manager(available, ["md"])
        resume_state = checkpoint_manager.prepare_run(images, resume=self.resume_checkpoint)
        images = resume_state.pending_images
        self.stats["pending_images"] = len(images)
        self.stats["resumed_skipped"] = len(resume_state.skipped_entries)

        if self.resume_checkpoint:
            logger.info(f"♻️ Checkpoint: {checkpoint_manager.checkpoint_path}")
        if resume_state.reset_reason:
            logger.info(f"♻️ {resume_state.reset_reason}")
        if self.stats["resumed_skipped"] > 0:
            logger.info(f"♻️ {self.stats['resumed_skipped']} imagens já concluídas foram puladas")
        logger.info("-" * 60)

        if not images:
            logger.info("✅ Nenhuma imagem pendente para processar. O lote já está concluído para esta configuração.")
            self._log_final_summary()
            return self.stats
        
        # Processar imagens (sequencial ou paralelo)
        total = len(images)
        batch_reports: list = []

        if self.workers == 1:
            # Processamento sequencial
            logger.info("📝 Modo: Sequencial")
            logger.info("-" * 60)

            for i, image_path in enumerate(images, 1):
                logger.info(f"\n[{i}/{total}] {image_path.name}")

                task_result = self._process_single_image(image_path, i, total, available)
                self._record_checkpoint_result(checkpoint_manager, image_path, task_result)
                self._accumulate_stats(task_result, image_path)
                batch_reports.append(task_result.get("report"))
        else:
            # Processamento paralelo com ThreadPoolExecutor
            logger.info(f"🚀 Modo: Paralelo ({self.workers} workers)")
            logger.info("-" * 60)
            
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Um pipeline por worker, criado uma vez, em vez de um por imagem.
            worker_pipelines = [self._create_shared_pipeline() for _ in range(self.workers)]

            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Submeter todas as tarefas
                futures = {
                    executor.submit(
                        self._process_single_image,
                        img_path, i, total, available,
                        worker_pipelines[(i - 1) % self.workers],
                    ): img_path
                    for i, img_path in enumerate(images, 1)
                }

                # Processar resultados conforme completam
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    task_result = future.result()
                    image_path = task_result["image_path"]
                    self._record_checkpoint_result(checkpoint_manager, image_path, task_result)
                    self._accumulate_stats(task_result, image_path)
                    batch_reports.append(task_result.get("report"))

                    # Progresso
                    progress = (completed / total) * 100
                    logger.info(f"   📊 Progresso: {completed}/{total} ({progress:.1f}%)")

            for worker_pipeline in worker_pipelines:
                worker_pipeline.unload_models()

        self._log_final_summary()

        # Descarregar da VRAM os modelos que o pipeline sequencial deixou residentes.
        # O unload acontece aqui, uma vez, e não a cada imagem.
        unloaded = self.pipeline.unload_models()
        if unloaded:
            logger.info(f"♻️ Modelos liberados da VRAM: {', '.join(unloaded)}")


        # Deteccao de duplicatas
        duplicate_result = None
        if DUPLICATE_DETECTOR_AVAILABLE and len(images) >= 2:
            try:
                logger.info("-" * 60)
                logger.info("🔎 Detectando duplicatas e near-duplicates...")
                detector = DuplicateDetector()
                img_pairs = []
                for img_path in images:
                    try:
                        from PIL import Image as _PILImage
                        pil_img = _PILImage.open(img_path)
                        img_pairs.append((str(img_path), pil_img))
                    except Exception:
                        pass
                if len(img_pairs) >= 2:
                    duplicate_result = detector.analyze_batch(img_pairs)
                    logger.info(f"   Duplicatas exatas: {duplicate_result.exact_duplicates}")
                    logger.info(f"   Near-duplicates: {duplicate_result.near_duplicates}")
            except Exception as e:
                logger.warning(f"⚠️ Falha na deteccao de duplicatas: {e}")

        # Sumario consolidado do lote
        if BATCH_SUMMARY_AVAILABLE:
            try:
                logger.info("-" * 60)
                logger.info("📊 Gerando sumario consolidado do lote...")
                builder = BatchSummaryBuilder()
                # Usa os relatórios reais do pipeline. Antes, os campos iam vazios e o
                # "success" era adivinhado por um nome de .md que nem batia com o gerado,
                # então o resumo saía sem entidades, timeline, GPS nem objetos.
                batch_results = [
                    report_to_summary_entry(report)
                    for report in batch_reports
                    if report is not None
                ]
                summary = builder.build(batch_results, duplicate_result)
                md_report = summary.to_markdown()
                summary_path = self.output_dir / "resumo_lote.md"
                summary_path.write_text(md_report, encoding="utf-8")
                logger.info(f"   Sumario salvo: {summary_path}")
            except Exception as e:
                logger.warning(f"⚠️ Falha ao gerar sumario do lote: {e}")
        
        return self.stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Função principal."""
    import argparse
    available_modes = get_available_analysis_modes()
    
    # Construir descrição dos modos disponíveis
    modes_help = "Modos disponíveis:\n"
    for key, info in available_modes.items():
        modes_help += f"  {key}: {info['name']} - {info['description']}\n"
    
    parser = argparse.ArgumentParser(
        description="Batch Image Analyzer - Análise de imagens com múltiplos modelos de IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{modes_help}
OCR Engines disponíveis:
  glm-ocr:  GLM OCR via Ollama (glm-ocr:bf16 — padrão)
  olmocr2:  OLMoOCR2 via Ollama (richardyoung/olmocr2:7b-q8)
  none:     Desabilita OCR

Processamento Paralelo:
  Use --workers para processar múltiplas imagens simultaneamente.
  Recomendado: 2-4 workers para APIs, 1-2 para modelos locais pesados.

Exemplos de uso:
  python batch_image_analyzer.py
  python batch_image_analyzer.py --mode forense --ocr glm-ocr
    python batch_image_analyzer.py --resume
  python batch_image_analyzer.py --workers 4  # Processar 4 imagens em paralelo
    python batch_image_analyzer.py --model gpt-5.4-mini --model gemma3:12b-it-q8_0
  python batch_image_analyzer.py -i ./fotos -o ./analises -w 3 --mode geral
  python batch_image_analyzer.py -i ./evidencias -o ./laudos -m forense --ocr glm-ocr -w 2
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=INPUT_DIR,
        help=f"Diretório de entrada com imagens (padrão: {INPUT_DIR})"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=Path,
        default=OUTPUT_DIR,
        help=f"Diretório de saída para relatórios (padrão: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="geral",
        choices=list(available_modes.keys()),
        help="Modo de análise (ver lista de modos disponíveis acima)"
    )
    
    parser.add_argument(
        "--ocr",
        type=str,
        default="glm-ocr",
        choices=OCR_ENGINES,
        help="Motor de OCR: 'glm-ocr' (padrão), 'olmocr2' ou 'none'"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Número de workers paralelos (1-8, padrão: 1 = sequencial)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        default=None,
        help="Modelo(s) para usar. Repita para múltiplos. Ex: --model gpt-5.4-mini --model gemma3:12b-it-q8_0. Padrão: gpt-5.4-mini"
    )
    
    parser.add_argument(
        "--gpu-profile",
        type=str,
        default=None,
        choices=["4gb", "6gb", "8gb", "16gb", "24gb", "32gb", "auto"],
        help="Perfil de GPU: seleciona modelos Ollama recomendados para o tier de VRAM (4gb/6gb/8gb/16gb/24gb/32gb/auto)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma um lote anterior usando checkpoint por imagem no diretório de saída"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Modo verbose (DEBUG)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parsear modelos especificados pelo usuário
    models_list = None
    if args.model:
        models_list = []
        for m in args.model:
            if m == OPENAI_MODEL or m.startswith("gpt"):
                models_list.append((m, "openai"))
            else:
                models_list.append((m, "ollama"))
    
    # Aplicar perfil de GPU (se nenhum --model foi especificado)
    if not models_list and args.gpu_profile:
        profile_key = args.gpu_profile
        if profile_key == "auto":
            vram = detect_vram_gb()
            profile_key = get_recommended_gpu_profile(vram)
            if profile_key:
                logger.info(f"🎮 GPU detectada: {vram:.1f} GB VRAM → perfil '{profile_key}'")
            else:
                logger.warning("⚠️ GPU não detectada ou VRAM insuficiente. Usando somente GPT-5.4-mini.")
        
        if profile_key in GPU_MODEL_PROFILES:
            models_list = [
                (model_id, "ollama") for model_id, _ in GPU_MODEL_PROFILES[profile_key]["models"]
            ]
            logger.info(f"📋 Perfil GPU '{profile_key}': {[m[0] for m in models_list]}")
            if profile_key == "32gb":
                logger.info("💡 Em GPUs de 32 GB, os modelos do tier 24 GB podem rodar com OCR simultaneamente.")
    
    # Executar análise
    analyzer = BatchImageAnalyzer(
        input_dir=args.input,
        output_dir=args.output,
        analysis_mode=args.mode,
        ocr_engine=args.ocr,
        workers=args.workers,
        models=models_list,
        resume=args.resume,
    )
    
    stats = analyzer.run()
    
    # Retornar código de saída apropriado
    if stats["failed"] > 0 and stats["processed"] == 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
