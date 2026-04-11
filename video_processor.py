#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎞️ Video Processor - Extrator Forense de Keyframes de Vídeo
============================================================
Converte vídeos (Câmera de Segurança, Celular, etc) em lotes de
imagens (keyframes) para serem processados pelo BatchImageAnalyzer.
"""

import cv2
import os
import logging
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    class _TqdmFallback:
        """Fallback simples quando tqdm não estiver instalado."""

        def __init__(self, total=None, desc=None):
            self.total = total
            self.desc = desc

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, exc_tb):
            return False

        def update(self, _value):
            return None

    def tqdm(*args, **kwargs):
        return _TqdmFallback(total=kwargs.get("total"), desc=kwargs.get("desc"))

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Ferramenta para decupar vídeos em frames analizáveis."""

    def __init__(self, output_dir: Path, frame_interval: int = 30):
        """
        Inicializa o processador de vídeo.

        Args:
            output_dir: Diretório onde serão salvas as imagens extraídas.
            frame_interval: A cada quantos frames extrair uma imagem (ex: 30 = 1 frame a cada ~1 segundo num video 30fps).
        """
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_keyframes(self, video_path: str) -> list[Path]:
        """
        Extrai os quadros-chave de um arquivo de vídeo.

        Args:
            video_path: Caminho completo para o arquivo de vídeo.
            
        Returns:
            Uma lista com os caminhos (Path) das imagens JPG extraídas.
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise Exception(f"Erro ao abrir stream de vídeo: {video_path}")

        # Metadata do Vídeo
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"🎞️ Vídeo: {video_path_obj.name} | FPS: {fps:.2f} | Duração: {duration:.2f}s | Total Frames: {total_frames}")

        frame_count = 0
        saved_count = 0
        extracted_files = []

        # Usar TQDM para progresso no terminal
        with tqdm(total=total_frames, desc=f"Descompactando {video_path_obj.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Captura frame apenas se bater o intervalo estipulado
                if frame_count % self.frame_interval == 0:
                    # Formatar timestamp para o nome do arquivo (ajuda no relatório LLM)
                    # ex: video_00m01s.jpg
                    seconds = frame_count / fps if fps > 0 else 0
                    mins = int(seconds // 60)
                    secs = int(seconds % 60)
                    timestamp_str = f"{mins:02d}m{secs:02d}s"

                    filename = f"{video_path_obj.stem}_{timestamp_str}.jpg"
                    out_path = self.output_dir / filename
                    
                    # Salva usando OpenCV
                    cv2.imwrite(str(out_path), frame)
                    extracted_files.append(out_path)
                    saved_count += 1

                frame_count += 1
                pbar.update(1)

        cap.release()
        logger.info(f"✅ {saved_count} frames extraídos com sucesso em {self.output_dir}")
        return extracted_files
