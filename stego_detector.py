#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stego Detector - Detecção de Esteganografia LSB.

Analisa os bits menos significativos (LSB) de cada canal RGB para
detectar possíveis dados escondidos. Usa análise de entropia e
chi-square attack para identificar padrões anômalos.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class StegoResult:
    """Resultado da detecção de esteganografia."""
    verdict: str = "nenhum"  # nenhum, suspeito, provavel
    lsb_entropy: float = 0.0
    expected_entropy: float = 0.0
    chi_square: float = 0.0
    chi_square_p: float = 0.0
    channel_entropies: dict = field(default_factory=dict)
    detail: str = ""

    def get_summary(self) -> str:
        lines = [
            f"Esteganografia LSB: {self.verdict}",
            f"Entropia LSB: {self.lsb_entropy:.4f} (esperado: {self.expected_entropy:.4f})",
            f"Chi-square: {self.chi_square:.2f} (p={self.chi_square_p:.4f})",
        ]
        if self.channel_entropies:
            for ch, ent in self.channel_entropies.items():
                lines.append(f"  Canal {ch}: {ent:.4f}")
        if self.detail:
            lines.append(self.detail)
        return "\n".join(lines)


class StegoDetector:
    """Detector de esteganografia LSB via entropia e chi-square."""

    # Entropia máxima esperada para LSBs de imagem natural
    NATURAL_LSB_ENTROPY_MAX = 0.95

    def analyze(self, image: Image.Image) -> StegoResult:
        """
        Analisa LSBs da imagem em busca de dados escondidos.

        Args:
            image: Imagem PIL.

        Returns:
            StegoResult com métricas e veredicto.
        """
        if not NUMPY_AVAILABLE or not PIL_AVAILABLE:
            return StegoResult(verdict="inconclusivo", detail="numpy/PIL indisponível")

        result = StegoResult()

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            img_array = np.array(image)
            h, w, c = img_array.shape

            # Limitar tamanho para performance
            max_pixels = 500_000
            if h * w > max_pixels:
                scale = math.sqrt(max_pixels / (h * w))
                new_h, new_w = int(h * scale), int(w * scale)
                img_array = np.array(
                    image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                )
                h, w, c = img_array.shape

            channel_entropies = {}
            all_lsbs = []

            for ch_idx in range(c):
                channel = img_array[:, :, ch_idx]
                lsbs = (channel & 1).flatten()
                all_lsbs.append(lsbs)
                ent = self._entropy(lsbs)
                channel_entropies[f"RGB"[ch_idx]] = ent

            result.channel_entropies = channel_entropies
            all_lsbs_flat = np.concatenate(all_lsbs)
            result.lsb_entropy = self._entropy(all_lsbs_flat)
            result.expected_entropy = 1.0  # Máximo para bits binários

            # Chi-square test
            result.chi_square, result.chi_square_p = self._chi_square_test(all_lsbs_flat)

            # Classificar
            high_entropy_channels = sum(
                1 for e in channel_entropies.values()
                if e > self.NATURAL_LSB_ENTROPY_MAX
            )

            if result.lsb_entropy > 0.99 and result.chi_square_p < 0.05:
                result.verdict = "provavel"
                result.detail = "Entropia LSB muito alta e chi-square significativo — dados escondidos prováveis."
            elif result.lsb_entropy > self.NATURAL_LSB_ENTROPY_MAX or high_entropy_channels >= 2:
                result.verdict = "suspeito"
                result.detail = "Entropia LSB elevada — possível esteganografia."
            else:
                result.verdict = "nenhum"
                result.detail = "Entropia LSB dentro do esperado para imagem natural."

        except Exception as e:
            logger.warning(f"Erro no stego detection: {e}")
            result.verdict = "inconclusivo"
            result.detail = str(e)

        return result

    @staticmethod
    def _entropy(bits: np.ndarray) -> float:
        """Calcula entropia de Shannon para array de bits (0/1)."""
        if len(bits) == 0:
            return 0.0
        p1 = np.mean(bits)
        p0 = 1.0 - p1
        ent = 0.0
        if p0 > 0:
            ent -= p0 * math.log2(p0)
        if p1 > 0:
            ent -= p1 * math.log2(p1)
        return ent

    @staticmethod
    def _chi_square_test(bits: np.ndarray) -> tuple:
        """
        Chi-square test para detectar padrões não-naturais nos LSBs.
        Compara frequência observada de pares de bits com esperada.
        """
        if len(bits) < 4:
            return 0.0, 1.0

        # Agrupar em pares
        pairs = bits.reshape(-1, 2) if len(bits) % 2 == 0 else bits[:-1].reshape(-1, 2)

        # Contar pares: (0,0), (0,1), (1,0), (1,1)
        observed = np.zeros(4)
        for pair in pairs:
            idx = int(pair[0]) * 2 + int(pair[1])
            observed[idx] += 1

        # Esperado: distribuição uniforme
        total = observed.sum()
        expected = np.full(4, total / 4)

        if total == 0:
            return 0.0, 1.0

        chi2 = np.sum((observed - expected) ** 2 / expected)

        # p-value aproximado (3 graus de liberdade)
        # Tabela simplificada: chi2 > 7.815 => p < 0.05
        if chi2 > 16.27:
            p = 0.001
        elif chi2 > 11.34:
            p = 0.01
        elif chi2 > 7.815:
            p = 0.05
        elif chi2 > 6.25:
            p = 0.10
        else:
            p = 0.50

        return float(chi2), float(p)
