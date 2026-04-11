#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 ELA Analyzer - Error Level Analysis para detecção de manipulação
===================================================================
Detecta regiões potencialmente manipuladas em imagens JPEG comparando
os níveis de erro após recompressão.
"""

import logging
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageChops, ImageEnhance

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class ELAResult:
    """Resultado da análise ELA."""
    ela_image: Optional[Image.Image] = None
    max_error: float = 0.0
    mean_error: float = 0.0
    std_error: float = 0.0
    suspicious_percent: float = 0.0
    verdict: str = "indeterminado"
    detail: str = ""
    quality_used: int = 95
    scale_factor: int = 15
    hotspot_count: int = 0
    hotspots: list = field(default_factory=list)

    def get_summary(self) -> str:
        """Retorna resumo textual para injeção em prompts."""
        lines = [
            f"Veredicto ELA: {self.verdict}",
            f"Erro máximo: {self.max_error:.1f}/255",
            f"Erro médio: {self.mean_error:.2f}/255",
            f"Desvio padrão: {self.std_error:.2f}",
            f"Pixels suspeitos: {self.suspicious_percent:.2f}%",
            f"Hotspots detectados: {self.hotspot_count}",
        ]
        if self.detail:
            lines.append(f"Observação: {self.detail}")
        return "\n".join(lines)


class ELAAnalyzer:
    """
    Realiza Error Level Analysis (ELA) em imagens.

    ELA funciona re-salvando a imagem em uma qualidade JPEG conhecida e
    comparando pixel a pixel com o original. Regiões editadas ou coladas
    tipicamente apresentam níveis de erro diferentes do restante da imagem.
    """

    # Thresholds para classificação
    THRESHOLD_LOW = 5.0       # erro médio abaixo = "sem indícios"
    THRESHOLD_MEDIUM = 12.0   # erro médio entre LOW e MEDIUM = "inconclusivo"
    THRESHOLD_HIGH = 20.0     # acima = "suspeito"
    SUSPICIOUS_PIXEL_THRESH = 50  # pixel com erro > este valor é considerado suspeito

    def __init__(self, quality: int = 95, scale: int = 15):
        """
        Args:
            quality: Qualidade JPEG para recompressão (90-100). Padrão 95.
            scale: Fator de amplificação da diferença para visualização. Padrão 15.
        """
        self.quality = max(90, min(100, quality))
        self.scale = max(1, min(50, scale))

    def analyze(self, image: Image.Image) -> ELAResult:
        """
        Executa ELA na imagem.

        Args:
            image: Imagem PIL (qualquer modo, será convertida para RGB).

        Returns:
            ELAResult com imagem ELA, métricas e veredicto.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1. Recomprimir em JPEG com qualidade conhecida
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).copy()

        # 2. Calcular diferença absoluta pixel a pixel
        diff = ImageChops.difference(image, recompressed)

        # 3. Amplificar para visualização
        extrema = diff.getextrema()
        max_diff = max(ch[1] for ch in extrema)

        if max_diff == 0:
            # Imagens idênticas — provavelmente já salva nessa qualidade
            return ELAResult(
                ela_image=diff,
                max_error=0.0,
                mean_error=0.0,
                std_error=0.0,
                suspicious_percent=0.0,
                verdict="sem_indicios",
                detail="Imagem idêntica após recompressão — possivelmente já salva nesta qualidade JPEG.",
                quality_used=self.quality,
                scale_factor=self.scale,
            )

        # Amplificar diferença
        ela_image = ImageEnhance.Brightness(diff).enhance(self.scale)

        # 4. Métricas estatísticas
        result = ELAResult(
            ela_image=ela_image,
            quality_used=self.quality,
            scale_factor=self.scale,
        )

        if NUMPY_AVAILABLE:
            diff_array = np.array(diff, dtype=np.float64)
            gray = np.mean(diff_array, axis=2)  # média dos canais

            result.max_error = float(np.max(gray))
            result.mean_error = float(np.mean(gray))
            result.std_error = float(np.std(gray))

            # % de pixels com erro elevado
            suspicious_mask = gray > self.SUSPICIOUS_PIXEL_THRESH
            total_pixels = gray.size
            suspicious_count = int(np.sum(suspicious_mask))
            result.suspicious_percent = (suspicious_count / total_pixels) * 100.0

            # Detectar hotspots (regiões contíguas suspeitas)
            result.hotspot_count, result.hotspots = self._find_hotspots(
                suspicious_mask, image.size
            )
        else:
            # Fallback sem numpy — usar getextrema
            result.max_error = float(max_diff)
            # Estimativa grosseira via histogram
            hist = diff.convert("L").histogram()
            total = sum(hist)
            weighted = sum(i * h for i, h in enumerate(hist))
            result.mean_error = weighted / total if total > 0 else 0.0
            suspicious_count = sum(hist[self.SUSPICIOUS_PIXEL_THRESH:])
            result.suspicious_percent = (suspicious_count / total) * 100.0 if total > 0 else 0.0

        # 5. Classificar
        result.verdict, result.detail = self._classify(result)

        return result

    def _find_hotspots(self, suspicious_mask: "np.ndarray", image_size: tuple) -> tuple:
        """
        Encontra regiões contíguas de pixels suspeitos.
        Retorna (count, list_of_bbox) onde bbox = (x1, y1, x2, y2).
        """
        try:
            import cv2 as _cv2
        except ImportError:
            # Sem CV2, contamos apenas o total de pixels suspeitos
            count = int(np.sum(suspicious_mask))
            return (1 if count > 0 else 0, [])

        mask_uint8 = suspicious_mask.astype(np.uint8) * 255

        # Dilatar para agrupar pixels próximos
        kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (5, 5))
        dilated = _cv2.dilate(mask_uint8, kernel, iterations=2)

        contours, _ = _cv2.findContours(dilated, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos muito pequenos (< 0.1% da imagem)
        w, h = image_size
        min_area = (w * h) * 0.001
        hotspots = []
        for cnt in contours:
            area = _cv2.contourArea(cnt)
            if area >= min_area:
                x, y, cw, ch = _cv2.boundingRect(cnt)
                hotspots.append((x, y, x + cw, y + ch))

        return len(hotspots), hotspots

    def _classify(self, result: ELAResult) -> tuple:
        """Classifica o resultado em veredicto + detalhe."""
        mean = result.mean_error
        std = result.std_error
        pct = result.suspicious_percent
        hotspots = result.hotspot_count

        # Critério 1: erro médio muito baixo → sem indícios
        if mean < self.THRESHOLD_LOW and pct < 0.5:
            return (
                "sem_indicios",
                "Distribuição de erro uniforme e baixa. Sem evidências de manipulação localizada."
            )

        # Critério 2: erro alto E concentrado em hotspots → suspeito
        if mean > self.THRESHOLD_HIGH or (pct > 3.0 and hotspots >= 2):
            return (
                "suspeito",
                f"Erro médio elevado ({mean:.1f}) com {hotspots} região(ões) concentrada(s). "
                "Possível edição localizada (clone, colagem, inpainting)."
            )

        # Critério 3: desvio padrão alto relativo à média → padrão irregular
        if std > mean * 1.5 and hotspots >= 1:
            return (
                "suspeito",
                f"Desvio padrão ({std:.1f}) desproporcional à média ({mean:.1f}). "
                "Distribuição de erro heterogênea sugere manipulação parcial."
            )

        # Critério 4: zona intermediária
        if mean >= self.THRESHOLD_LOW:
            return (
                "inconclusivo",
                f"Erro médio moderado ({mean:.1f}). Pode indicar recompressões múltiplas, "
                "redimensionamento ou edições menores. Recomenda-se análise complementar."
            )

        return ("indeterminado", "Dados insuficientes para uma classificação confiável.")

    def generate_heatmap(self, ela_image: Image.Image) -> Optional[Image.Image]:
        """
        Gera um mapa de calor colorido a partir da imagem ELA.
        Requer numpy + PIL.

        Returns:
            Imagem PIL com mapa de calor (vermelho = mais erro) ou None se numpy indisponível.
        """
        if not NUMPY_AVAILABLE:
            return None

        arr = np.array(ela_image.convert("L"), dtype=np.float64)

        # Normalizar 0-1
        max_val = arr.max()
        if max_val == 0:
            return ela_image

        normalized = arr / max_val

        # Paleta: azul (0) → verde (0.33) → amarelo (0.66) → vermelho (1)
        r = np.clip(normalized * 3 - 1, 0, 1) * 255
        g = np.where(
            normalized < 0.5,
            normalized * 2 * 255,
            (1 - (normalized - 0.5) * 2) * 255,
        )
        b = np.clip(1 - normalized * 3, 0, 1) * 255

        heatmap = np.stack([r, g, b], axis=2).astype(np.uint8)
        return Image.fromarray(heatmap)

    def overlay(
        self, original: Image.Image, ela_image: Image.Image, alpha: float = 0.5
    ) -> Image.Image:
        """
        Sobrepõe a imagem ELA (ou heatmap) sobre a original.

        Args:
            original: Imagem original PIL.
            ela_image: Imagem ELA ou heatmap.
            alpha: Opacidade da sobreposição (0-1).

        Returns:
            Imagem PIL com overlay.
        """
        if original.mode != "RGB":
            original = original.convert("RGB")
        if ela_image.mode != "RGB":
            ela_image = ela_image.convert("RGB")

        # Garantir mesmo tamanho
        if ela_image.size != original.size:
            ela_image = ela_image.resize(original.size, Image.LANCZOS)

        return Image.blend(original, ela_image, alpha)
