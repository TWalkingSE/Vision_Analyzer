#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🗄️ Cache Manager - Sistema de Cache Inteligente para Análises
==============================================================
Evita reprocessar imagens já analisadas usando hash MD5.
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Diretório de cache
CACHE_DIR = Path("./.vision_cache")
CACHE_INDEX_FILE = CACHE_DIR / "cache_index.json"
STATS_FILE = CACHE_DIR / "stats_history.json"


@dataclass
class CacheEntry:
    """Entrada de cache para uma análise."""
    image_hash: str
    image_path: str
    image_name: str
    model: str
    analysis_mode: str
    ocr_engine: str
    report_path: str
    created_at: str
    file_size: int
    dimensions: tuple
    # Dados intermediários para re-análise sem reprocessar
    ocr_result: str = ""
    yolo_result: str = ""
    quality_result: str = ""
    exif_data: str = ""


class CacheManager:
    """Gerenciador de cache para análises de imagens."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_index_file = cache_dir / "cache_index.json"
        self.stats_file = cache_dir / "stats_history.json"
        self._ensure_cache_dir()
        self._load_index()
        self._load_stats()
    
    def _ensure_cache_dir(self):
        """Garante que o diretório de cache existe."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_index(self):
        """Carrega o índice de cache do disco."""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
                logger.info(f"📁 Cache carregado: {len(self.index)} entradas")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar cache: {e}")
                self.index = {}
        else:
            self.index = {}
    
    def _save_index(self):
        """Salva o índice de cache no disco."""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Erro ao salvar cache: {e}")
    
    @staticmethod
    def calculate_image_hash(image_path: Path) -> str:
        """Calcula hash MD5 de uma imagem."""
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def generate_cache_key(image_hash: str, model: str, analysis_mode: str, ocr_engine: str) -> str:
        """Gera chave única para entrada de cache."""
        return f"{image_hash}_{model}_{analysis_mode}_{ocr_engine}"
    
    def is_cached(
        self, 
        image_path: Path, 
        model: str, 
        analysis_mode: str, 
        ocr_engine: str
    ) -> tuple[bool, Optional[str]]:
        """
        Verifica se uma análise já está em cache.
        Retorna: (está_em_cache, caminho_do_relatório)
        """
        try:
            image_hash = self.calculate_image_hash(image_path)
            cache_key = self.generate_cache_key(image_hash, model, analysis_mode, ocr_engine)
            
            if cache_key in self.index:
                entry = self.index[cache_key]
                report_path = Path(entry["report_path"])
                
                # Verificar se o relatório ainda existe
                if report_path.exists():
                    logger.info(f"✅ Cache hit: {image_path.name} ({model})")
                    return True, str(report_path)
                else:
                    # Relatório foi deletado, remover do cache
                    del self.index[cache_key]
                    self._save_index()
            
            return False, None
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao verificar cache: {e}")
            return False, None
    
    def add_to_cache(
        self,
        image_path: Path,
        model: str,
        analysis_mode: str,
        ocr_engine: str,
        report_path: Path,
        dimensions: tuple = (0, 0),
        ocr_result: str = "",
        yolo_result: str = "",
        quality_result: str = "",
        exif_data: str = ""
    ):
        """Adiciona uma análise ao cache com dados intermediários."""
        try:
            image_hash = self.calculate_image_hash(image_path)
            cache_key = self.generate_cache_key(image_hash, model, analysis_mode, ocr_engine)
            
            entry = CacheEntry(
                image_hash=image_hash,
                image_path=str(image_path),
                image_name=image_path.name,
                model=model,
                analysis_mode=analysis_mode,
                ocr_engine=ocr_engine,
                report_path=str(report_path),
                created_at=datetime.now().isoformat(),
                file_size=image_path.stat().st_size,
                dimensions=dimensions,
                ocr_result=ocr_result,
                yolo_result=yolo_result,
                quality_result=quality_result,
                exif_data=exif_data
            )
            
            self.index[cache_key] = asdict(entry)
            self._save_index()
            
            logger.debug(f"📝 Adicionado ao cache: {image_path.name} ({model})")
            
        except Exception as e:
            logger.error(f"❌ Erro ao adicionar ao cache: {e}")
    
    def get_cached_intermediate(self, image_path: Path) -> Optional[Dict[str, str]]:
        """Retorna dados intermediários (OCR, YOLO, etc.) de qualquer análise
        em cache para esta imagem, para permitir re-análise sem reprocessar."""
        try:
            image_hash = self.calculate_image_hash(image_path)
            for key, entry in self.index.items():
                if entry.get("image_hash") == image_hash:
                    ocr = entry.get("ocr_result", "")
                    if ocr:
                        return {
                            "ocr_result": ocr,
                            "ocr_engine": entry.get("ocr_engine", ""),
                            "yolo_result": entry.get("yolo_result", ""),
                            "quality_result": entry.get("quality_result", ""),
                            "exif_data": entry.get("exif_data", ""),
                        }
        except Exception:
            pass
        return None
    
    def _load_stats(self):
        """Carrega histórico de estatísticas."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    self.stats_history = json.load(f)
            except Exception:
                self.stats_history = []
        else:
            self.stats_history = []
    
    def _save_stats(self):
        """Salva histórico de estatísticas."""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Erro ao salvar stats: {e}")
    
    def record_analysis(self, model: str, processing_time: float, cache_hit: bool = False):
        """Registra uma análise no histórico de métricas."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model": model,
            "processing_time": round(processing_time, 2),
            "cache_hit": cache_hit,
        }
        self.stats_history.append(entry)
        self._save_stats()
    
    def get_stats_history(self) -> list:
        """Retorna o histórico completo de métricas."""
        return self.stats_history
    
    def clear_cache(self):
        """Limpa todo o cache."""
        self.index = {}
        self._save_index()
        logger.info("🗑️ Cache limpo")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        total_entries = len(self.index)
        
        models = {}
        modes = {}
        total_size = 0
        
        for entry in self.index.values():
            model = entry.get("model", "unknown")
            mode = entry.get("analysis_mode", "unknown")
            size = entry.get("file_size", 0)
            
            models[model] = models.get(model, 0) + 1
            modes[mode] = modes.get(mode, 0) + 1
            total_size += size
        
        return {
            "total_entries": total_entries,
            "total_size_mb": total_size / (1024 * 1024),
            "by_model": models,
            "by_mode": modes,
            "cache_dir": str(self.cache_dir)
        }
    
    def remove_entry(self, image_path: Path, model: str, analysis_mode: str, ocr_engine: str):
        """Remove uma entrada específica do cache."""
        try:
            image_hash = self.calculate_image_hash(image_path)
            cache_key = self.generate_cache_key(image_hash, model, analysis_mode, ocr_engine)
            
            if cache_key in self.index:
                del self.index[cache_key]
                self._save_index()
                logger.info(f"🗑️ Removido do cache: {image_path.name} ({model})")
                
        except Exception as e:
            logger.error(f"❌ Erro ao remover do cache: {e}")


# Instância global para uso fácil
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Retorna a instância global do CacheManager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
