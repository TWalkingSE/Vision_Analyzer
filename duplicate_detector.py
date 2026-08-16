#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duplicate Detector - Detecção de near-duplicates via perceptual hash.

Compara imagens de um lote usando pHash e dHash para identificar
duplicatas exatas (mesmo MD5) e near-duplicates (hash similar).
Útil para identificar se a mesma foto aparece múltiplas vezes
com edições diferentes.
"""

import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class DuplicatePair:
    """Par de imagens duplicadas ou near-duplicate."""
    image_a: str
    image_b: str
    similarity: float
    phash_distance: int
    dhash_distance: int
    same_md5: bool


@dataclass
class DuplicateResult:
    """Resultado da detecção de duplicatas."""
    pairs: List[DuplicatePair] = field(default_factory=list)
    exact_duplicates: int = 0
    near_duplicates: int = 0
    total_images: int = 0
    hashes: Dict[str, dict] = field(default_factory=dict)

    def get_summary(self) -> str:
        if not self.pairs:
            return f"Duplicatas: Nenhuma duplicata detectada em {self.total_images} imagem(ns)."

        lines = [
            f"Duplicatas: {self.exact_duplicates} exata(s), {self.near_duplicates} near-duplicate(s) em {self.total_images} imagem(ns).",
        ]
        for p in self.pairs[:10]:
            kind = "EXATA" if p.same_md5 else "NEAR"
            lines.append(
                f"  [{kind}] {Path(p.image_a).name} ↔ {Path(p.image_b).name} "
                f"(sim: {p.similarity:.1%}, pHash dist: {p.phash_distance})"
            )
        if len(self.pairs) > 10:
            lines.append(f"  ... e mais {len(self.pairs) - 10} par(es).")
        return "\n".join(lines)


class DuplicateDetector:
    """Detector de duplicatas e near-duplicates via perceptual hash."""

    def __init__(self, phash_threshold: int = 8, dhash_threshold: int = 8):
        """
        Args:
            phash_threshold: Distância máxima de pHash para near-duplicate.
            dhash_threshold: Distância máxima de dHash para near-duplicate.
        """
        self.phash_threshold = phash_threshold
        self.dhash_threshold = dhash_threshold

    def compute_hashes(self, image: Image.Image) -> dict:
        """
        Computa hashes de uma imagem.

        Returns:
            Dict com phash, dhash, e md5.
        """
        result = {"phash": None, "dhash": None, "md5": ""}

        if not PIL_AVAILABLE:
            return result

        try:
            # pHash: DCT-based
            result["phash"] = self._phash(image)
            # dHash: difference hash
            result["dhash"] = self._dhash(image)
        except Exception as e:
            logger.debug(f"Erro ao computar hash: {e}")

        return result

    def analyze_batch(self, images: List[Tuple[str, Image.Image]]) -> DuplicateResult:
        """
        Compara todas as imagens de um lote.

        Args:
            images: Lista de (caminho, imagem PIL).

        Returns:
            DuplicateResult com pares de duplicatas.
        """
        result = DuplicateResult(total_images=len(images))

        if not PIL_AVAILABLE or len(images) < 2:
            return result

        # Computar hashes de todas as imagens
        hashes_list = []
        for path, img in images:
            h = self.compute_hashes(img)
            try:
                buf = img.tobytes()
                h["md5"] = hashlib.md5(buf).hexdigest()
            except Exception:
                pass
            h["path"] = str(path)
            hashes_list.append(h)
            result.hashes[str(path)] = h

        # Comparar todos os pares
        for i in range(len(hashes_list)):
            for j in range(i + 1, len(hashes_list)):
                a = hashes_list[i]
                b = hashes_list[j]

                same_md5 = bool(a.get("md5") and a.get("md5") == b.get("md5"))

                ph_dist = self._hamming_distance(a.get("phash"), b.get("phash"))
                dh_dist = self._hamming_distance(a.get("dhash"), b.get("dhash"))

                # Similaridade baseada em pHash
                if a.get("phash") is not None:
                    hash_len = len(a["phash"])
                    similarity = 1.0 - (ph_dist / max(hash_len, 1))
                else:
                    similarity = 0.0

                is_near = (
                    ph_dist <= self.phash_threshold
                    and dh_dist <= self.dhash_threshold
                )

                if same_md5 or is_near:
                    pair = DuplicatePair(
                        image_a=a["path"],
                        image_b=b["path"],
                        similarity=similarity,
                        phash_distance=ph_dist,
                        dhash_distance=dh_dist,
                        same_md5=same_md5,
                    )
                    result.pairs.append(pair)

                    if same_md5:
                        result.exact_duplicates += 1
                    else:
                        result.near_duplicates += 1

        return result

    @staticmethod
    def _phash(image: Image.Image, hash_size: int = 8) -> Optional[List[int]]:
        """Computa perceptual hash (pHash) usando DCT via multiplicação matricial."""
        if not NUMPY_AVAILABLE:
            return None

        try:
            size = 32
            img = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float64)

            # Construir matriz DCT 1D (size x hash_size) — separável
            # D[u, x] = cos(pi * (2*x+1) * u / (2*size))
            x = np.arange(size).reshape(1, -1)
            u = np.arange(hash_size).reshape(-1, 1)
            dct_matrix = np.cos(np.pi * (2 * x + 1) * u / (2 * size))

            # DCT 2D via duas multiplicações matriciais: D @ arr @ D^T
            dct_block = dct_matrix @ arr @ dct_matrix.T

            # Mediana excluindo o componente DC (canto [0,0])
            flat = dct_block.flatten()
            median = np.median(flat[1:])

            return (dct_block > median).flatten().astype(int).tolist()
        except Exception as e:
            logger.debug(f"Erro no pHash: {e}")
            return None

    @staticmethod
    def _dhash(image: Image.Image, hash_size: int = 8) -> Optional[List[int]]:
        """Computa difference hash (dHash)."""
        try:
            img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())

            width = hash_size + 1
            diff = []
            for row in range(hash_size):
                for col in range(hash_size):
                    left = pixels[row * width + col]
                    right = pixels[row * width + col + 1]
                    diff.append(1 if left > right else 0)
            return diff
        except Exception as e:
            logger.debug(f"Erro no dHash: {e}")
            return None

    @staticmethod
    def _hamming_distance(a: Optional[List[int]], b: Optional[List[int]]) -> int:
        """Distância de Hamming entre dois hashes binários."""
        if a is None or b is None:
            return 64  # Máximo
        if len(a) != len(b):
            return max(len(a), len(b))
        return sum(1 for x, y in zip(a, b) if x != y)
