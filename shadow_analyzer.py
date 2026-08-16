#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shadow Analyzer - Análise de consistência de sombras.

Detecta a direção de luz dominante na imagem e verifica se as sombras
são consistentes. Sombras apontando em direções diferentes podem indicar
composição/montagem de elementos de diferentes fontes.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class ShadowRegion:
    """Região de sombra detectada."""
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: int
    gradient_angle: float  # direção do gradiente (0-360)


@dataclass
class ShadowResult:
    """Resultado da análise de sombras."""
    light_direction: float = 0.0  # direção estimada da luz (0-360)
    shadow_regions: List[ShadowRegion] = field(default_factory=list)
    consistent: bool = True
    inconsistency_score: float = 0.0
    verdict: str = "consistente"  # consistente, inconclusivo, inconsistente
    detail: str = ""

    def get_summary(self) -> str:
        lines = [
            f"Análise de Sombras: {self.verdict}",
            f"Direção de luz estimada: {self.light_direction:.1f}°",
            f"Regiões de sombra: {len(self.shadow_regions)}",
            f"Score de inconsistência: {self.inconsistency_score:.2f}",
        ]
        if self.detail:
            lines.append(self.detail)
        return "\n".join(lines)


class ShadowAnalyzer:
    """Analisa consistência de sombras para detectar montagem."""

    def __init__(
        self,
        shadow_threshold: int = 60,
        min_region_area: int = 100,
        angle_tolerance: float = 25.0,
    ):
        """
        Args:
            shadow_threshold: Valor máximo de luminância para considerar sombra.
            min_region_area: Área mínima para considerar uma região de sombra.
            angle_tolerance: Tolerância angular (graus) para considerar consistente.
        """
        self.shadow_threshold = shadow_threshold
        self.min_region_area = min_region_area
        self.angle_tolerance = angle_tolerance

    def analyze(self, image: Image.Image) -> ShadowResult:
        """
        Analisa consistência de sombras na imagem.

        Args:
            image: Imagem PIL.

        Returns:
            ShadowResult com direção de luz e flag de consistência.
        """
        if not NUMPY_AVAILABLE or not PIL_AVAILABLE:
            return ShadowResult(verdict="inconclusivo", detail="numpy/PIL indisponível")

        result = ShadowResult()

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Redimensionar para análise (limite 512px)
            max_side = 512
            w, h = image.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                image = image.resize(
                    (int(w * scale), int(h * scale)),
                    Image.Resampling.LANCZOS,
                )

            gray = np.array(image.convert("L"), dtype=np.float64)
            h, w = gray.shape

            # 1. Detectar direção de luz dominante via gradiente global
            result.light_direction = self._estimate_light_direction(gray)

            # 2. Detectar regiões de sombra (pixels escuros)
            shadow_mask = gray < self.shadow_threshold

            # 3. Encontrar regiões contíguas de sombra
            regions = self._find_shadow_regions(shadow_mask, gray, w, h)

            # Filtrar regiões pequenas
            regions = [r for r in regions if r.area >= self.min_region_area]
            result.shadow_regions = regions

            if len(regions) < 2:
                result.verdict = "inconclusivo"
                result.detail = "Regiões de sombra insuficientes para análise."
                return result

            # 4. Verificar consistência dos gradientes das sombras
            angles = [r.gradient_angle for r in regions]
            angle_diffs = []
            for i in range(len(angles)):
                for j in range(i + 1, len(angles)):
                    diff = abs(angles[i] - angles[j])
                    diff = min(diff, 360 - diff)  # Diferença circular
                    angle_diffs.append(diff)

            if angle_diffs:
                max_diff = max(angle_diffs)
                avg_diff = sum(angle_diffs) / len(angle_diffs)
                result.inconsistency_score = avg_diff / 180.0

                if max_diff > self.angle_tolerance * 2:
                    result.consistent = False
                    result.verdict = "inconsistente"
                    result.detail = (
                        f"Sombras com direções divergentes detectadas "
                        f"(diff máx: {max_diff:.1f}°, média: {avg_diff:.1f}°). "
                        f"Possível montagem."
                    )
                elif max_diff > self.angle_tolerance:
                    result.consistent = True
                    result.verdict = "inconclusivo"
                    result.detail = (
                        f"Pequena divergência nas sombras "
                        f"(diff máx: {max_diff:.1f}°). Inconclusivo."
                    )
                else:
                    result.consistent = True
                    result.verdict = "consistente"
                    result.detail = "Sombras consistentes com fonte de luz única."

        except Exception as e:
            logger.warning(f"Erro no shadow analysis: {e}")
            result.verdict = "inconclusivo"
            result.detail = str(e)

        return result

    @staticmethod
    def _estimate_light_direction(gray: np.ndarray) -> float:
        """Estima direção de luz dominante via gradiente de luminância."""
        # Gradientes
        gy = np.gradient(gray, axis=0)
        gx = np.gradient(gray, axis=1)

        # Direção média do gradiente
        mean_gx = np.mean(gx)
        mean_gy = np.mean(gy)

        # Ângulo em graus (0 = direita, 90 = baixo)
        angle = math.degrees(math.atan2(mean_gy, mean_gx))
        if angle < 0:
            angle += 360

        return angle

    def _find_shadow_regions(
        self, mask: np.ndarray, gray: np.ndarray, w: int, h: int
    ) -> List[ShadowRegion]:
        """Encontra regiões contíguas de sombra usando flood fill simples."""
        if not NUMPY_AVAILABLE:
            return []

        visited = np.zeros_like(mask, dtype=bool)
        regions = []

        for y in range(h):
            for x in range(w):
                if mask[y, x] and not visited[y, x]:
                    # Flood fill
                    stack = [(x, y)]
                    pixels = []

                    while stack:
                        px, py = stack.pop()
                        if px < 0 or px >= w or py < 0 or py >= h:
                            continue
                        if visited[py, px] or not mask[py, px]:
                            continue
                        visited[py, px] = True
                        pixels.append((px, py))
                        stack.extend([
                            (px + 1, py), (px - 1, py),
                            (px, py + 1), (px, py - 1),
                        ])

                    if len(pixels) >= self.min_region_area:
                        xs = [p[0] for p in pixels]
                        ys = [p[1] for p in pixels]
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)
                        cx = sum(xs) / len(xs)
                        cy = sum(ys) / len(ys)

                        # Gradiente local na região
                        local = gray[max(0, y1 - 5):min(h, y2 + 6),
                                     max(0, x1 - 5):min(w, x2 + 6)]
                        if local.shape[0] > 1 and local.shape[1] > 1:
                            lgy = np.gradient(local, axis=0)
                            lgx = np.gradient(local, axis=1)
                            angle = math.degrees(
                                math.atan2(np.mean(lgy), np.mean(lgx))
                            )
                            if angle < 0:
                                angle += 360
                        else:
                            angle = 0.0

                        regions.append(ShadowRegion(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            centroid=(cx, cy),
                            area=len(pixels),
                            gradient_angle=angle,
                        ))

        return regions
