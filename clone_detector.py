#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clone Detector - Detecção de Copy-Move Forgery.

Identifica regiões duplicadas dentro da mesma imagem que podem indicar
manipulação (clonagem de áreas para cobrir ou adicionar elementos).

Método: Divide a imagem em blocos sobrepostos, calcula hash perceptual
de cada bloco e identifica pares de blocos similares em posições diferentes.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageChops, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class CloneRegion:
    """Região suspeita de clonagem."""
    bbox_a: Tuple[int, int, int, int]  # x1, y1, x2, y2
    bbox_b: Tuple[int, int, int, int]
    similarity: float
    block_size: int


@dataclass
class CloneResult:
    """Resultado da detecção de clone."""
    regions: List[CloneRegion] = field(default_factory=list)
    verdict: str = "sem_indicios"  # sem_indicios, suspeito, inconclusivo
    block_size: int = 16
    total_pairs: int = 0
    heatmap: Optional[object] = None  # PIL Image

    def get_summary(self) -> str:
        if not self.regions:
            return "Copy-Move: Nenhuma região clonada detectada."

        lines = [
            f"Copy-Move: {len(self.regions)} região(ões) suspeita(s) detectada(s).",
            f"Veredicto: {self.verdict}",
            f"Pares analisados: {self.total_pairs}",
        ]
        for i, r in enumerate(self.regions[:5]):
            lines.append(
                f"  Região {i+1}: similaridade {r.similarity:.2%} "
                f"entre {r.bbox_a} e {r.bbox_b}"
            )
        if len(self.regions) > 5:
            lines.append(f"  ... e mais {len(self.regions) - 5} região(ões).")
        return "\n".join(lines)


class CloneDetector:
    """Detector de copy-move forgery via block-matching com hash perceptual."""

    def __init__(
        self,
        block_size: int = 16,
        stride: int = 8,
        threshold: float = 0.92,
        min_variance: float = 50.0,
        min_distance: int = 32,
    ):
        """
        Args:
            block_size: Tamanho do bloco em pixels.
            stride: Passo entre blocos (menor = mais denso, mais lento).
            threshold: Similaridade mínima para considerar clone (0-1).
            min_variance: Variância mínima do bloco (filtra áreas uniformes).
            min_distance: Distância mínima entre blocos similares (evita vizinhos).
        """
        self.block_size = block_size
        self.stride = stride
        self.threshold = threshold
        self.min_variance = min_variance
        self.min_distance = min_distance

    def analyze(self, image: Image.Image) -> CloneResult:
        """
        Executa detecção de copy-move na imagem.

        Args:
            image: Imagem PIL.

        Returns:
            CloneResult com regiões suspeitas.
        """
        if not NUMPY_AVAILABLE or not PIL_AVAILABLE:
            return CloneResult(verdict="inconclusivo")

        result = CloneResult(block_size=self.block_size)

        try:
            # Converter para grayscale e numpy
            gray = image.convert("L")
            img_array = np.array(gray, dtype=np.float64)

            h, w = img_array.shape
            bs = self.block_size

            # Redimensionar se imagem muito grande (limite 1024px no lado maior)
            max_side = 1024
            scale = 1.0
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img_array = np.array(
                    gray.resize((new_w, new_h), Image.Resampling.LANCZOS),
                    dtype=np.float64,
                )
                h, w = img_array.shape

            # Extrair blocos com hash DCT simplificado
            blocks = []
            positions = []

            for y in range(0, h - bs + 1, self.stride):
                for x in range(0, w - bs + 1, self.stride):
                    block = img_array[y:y + bs, x:x + bs]

                    # Filtrar áreas uniformes
                    if block.var() < self.min_variance:
                        continue

                    # Hash perceptual simplificado: DCT 8x8 dos valores médios
                    block_hash = self._compute_block_hash(block)
                    blocks.append(block_hash)
                    positions.append((x, y))

            if len(blocks) < 2:
                return result

            blocks_array = np.array(blocks)
            result.total_pairs = len(blocks)

            # Comparar todos os pares
            seen_pairs = set()
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    # Distância espacial
                    dx = abs(positions[i][0] - positions[j][0])
                    dy = abs(positions[i][1] - positions[j][1])
                    if dx < self.min_distance and dy < self.min_distance:
                        continue

                    # Similaridade via Hamming distance no hash binário
                    hamming = np.count_nonzero(blocks_array[i] != blocks_array[j])
                    similarity = 1.0 - (hamming / len(blocks_array[i]))

                    if similarity >= self.threshold:
                        # Deduplicar pares na mesma área
                        pair_key = (
                            min(positions[i], positions[j]),
                            max(positions[i], positions[j]),
                        )
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        # Converter coords de volta para escala original
                        s = 1.0 / scale
                        x1_a, y1_a = positions[i]
                        x1_b, y1_b = positions[j]
                        result.regions.append(CloneRegion(
                            bbox_a=(
                                int(x1_a * s), int(y1_a * s),
                                int((x1_a + bs) * s), int((y1_a + bs) * s),
                            ),
                            bbox_b=(
                                int(x1_b * s), int(y1_b * s),
                                int((x1_b + bs) * s), int((y1_b + bs) * s),
                            ),
                            similarity=float(similarity),
                            block_size=bs,
                        ))

            # Determinar veredicto
            if len(result.regions) >= 3:
                result.verdict = "suspeito"
            elif len(result.regions) >= 1:
                result.verdict = "inconclusivo"
            else:
                result.verdict = "sem_indicios"

        except Exception as e:
            logger.warning(f"Erro no clone detection: {e}")
            result.verdict = "inconclusivo"

        return result

    @staticmethod
    def _compute_block_hash(block: np.ndarray) -> np.ndarray:
        """Computa hash binário do bloco usando média + threshold."""
        # Redimensionar para 8x8 se necessário
        if block.shape[0] >= 8 and block.shape[1] >= 8:
            # Subamostrar para 8x8
            step_y = block.shape[0] // 8
            step_x = block.shape[1] // 8
            small = block[::step_y, ::step_x][:8, :8]
        else:
            small = block

        mean = small.mean()
        return (small > mean).flatten().astype(np.int8)
