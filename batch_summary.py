#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Summary - Sumário consolidado de lote de imagens.

Agrega resultados de múltiplas imagens em um relatório consolidado:
- Total de imagens, sucessos, falhas
- Entidades extraídas (deduplicadas, com contagem)
- Timeline consolidada (cross-image, ordenada)
- Distribuição de tipos de documento/ameaça
- Top objetos detectados (YOLO agregado)
- Coordenadas GPS agregadas
- Duplicatas detectadas
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class BatchSummary:
    """Sumário consolidado de um lote de imagens."""
    total_images: int = 0
    successful: int = 0
    failed: int = 0
    entities_by_type: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    entity_counts: Dict[str, int] = field(default_factory=dict)
    document_types: Dict[str, int] = field(default_factory=dict)
    threat_types: Dict[str, int] = field(default_factory=dict)
    consolidated_timeline: List[Dict[str, Any]] = field(default_factory=list)
    gps_coordinates: List[Dict[str, Any]] = field(default_factory=list)
    top_objects: Dict[str, int] = field(default_factory=dict)
    duplicates: List[Dict[str, Any]] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    forensic_verdicts: Dict[str, int] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Gera relatório Markdown consolidado."""
        lines = [
            "# 📊 Relatório Consolidado de Lote",
            "",
            f"**Total de imagens:** {self.total_images}",
            f"**Análises bem-sucedidas:** {self.successful}",
            f"**Falhas:** {self.failed}",
            "",
        ]

        # Entidades
        if self.entity_counts:
            lines.append("## 📌 Entidades Extraídas (Agregadas)")
            lines.append("")
            lines.append("| Tipo | Contagem | Valores (top 5) |")
            lines.append("|------|----------|-----------------|")
            for etype, count in sorted(self.entity_counts.items(), key=lambda x: -x[1]):
                values = self.entities_by_type.get(etype, [])
                top_vals = [v.get("value", "") for v in values[:5]]
                lines.append(f"| {etype} | {count} | {', '.join(top_vals)} |")
            lines.append("")

        # Classificação
        if self.document_types:
            lines.append("## 🏷️ Tipos de Documento Detectados")
            lines.append("")
            for dtype, count in sorted(self.document_types.items(), key=lambda x: -x[1]):
                lines.append(f"- **{dtype}:** {count}")
            lines.append("")

        if self.threat_types:
            lines.append("## ⚠️ Tipos de Ameaça Detectados")
            lines.append("")
            for ttype, count in sorted(self.threat_types.items(), key=lambda x: -x[1]):
                lines.append(f"- **{ttype}:** {count}")
            lines.append("")

        # Timeline consolidada
        if self.consolidated_timeline:
            lines.append("## 📅 Timeline Consolidada (Cross-Image)")
            lines.append("")
            lines.append("| Timestamp | Imagem | Descrição |")
            lines.append("|-----------|--------|-----------|")
            for event in self.consolidated_timeline[:20]:
                lines.append(
                    f"| {event.get('timestamp', '')} "
                    f"| {event.get('image', '')} "
                    f"| {event.get('description', '')[:80]} |"
                )
            if len(self.consolidated_timeline) > 20:
                lines.append(f"| ... | ... | ... e mais {len(self.consolidated_timeline) - 20} eventos |")
            lines.append("")

        # GPS
        if self.gps_coordinates:
            lines.append("## 📍 Coordenadas GPS Agregadas")
            lines.append("")
            lines.append("| Imagem | Latitude | Longitude | Mapa |")
            lines.append("|--------|----------|-----------|------|")
            for gps in self.gps_coordinates:
                lat = gps.get("lat", "")
                lon = gps.get("lon", "")
                img = gps.get("image", "")
                url = gps.get("osm_url", "")
                lines.append(f"| {img} | {lat} | {lon} | [Ver mapa]({url}) |")
            lines.append("")

        # Objetos YOLO
        if self.top_objects:
            lines.append("## 🤖 Objetos Detectados (YOLO Agregado)")
            lines.append("")
            for obj, count in sorted(self.top_objects.items(), key=lambda x: -x[1]):
                lines.append(f"- **{obj}:** {count}")
            lines.append("")

        # Duplicatas
        if self.duplicates:
            lines.append("## 🔄 Duplicatas Detectadas")
            lines.append("")
            for dup in self.duplicates:
                kind = "EXATA" if dup.get("same_md5") else "NEAR"
                lines.append(
                    f"- [{kind}] {dup.get('image_a', '')} ↔ {dup.get('image_b', '')} "
                    f"(sim: {dup.get('similarity', 0):.1%})"
                )
            lines.append("")

        # Forensic verdicts
        if self.forensic_verdicts:
            lines.append("## 🔬 Veredictos Forenses Agregados")
            lines.append("")
            for verdict, count in sorted(self.forensic_verdicts.items(), key=lambda x: -x[1]):
                lines.append(f"- **{verdict}:** {count}")
            lines.append("")

        # Quality scores
        if self.quality_scores:
            avg_quality = sum(self.quality_scores) / len(self.quality_scores)
            lines.append("## 📈 Qualidade das Imagens")
            lines.append("")
            lines.append(f"- **Score médio:** {avg_quality:.0f}/100")
            lines.append(f"- **Melhor:** {max(self.quality_scores):.0f}/100")
            lines.append(f"- **Pior:** {min(self.quality_scores):.0f}/100")
            lines.append("")

        lines.append("---")
        lines.append(f"*Relatório consolidado gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Retorna representação em dicionário."""
        return {
            "total_images": self.total_images,
            "successful": self.successful,
            "failed": self.failed,
            "entity_counts": self.entity_counts,
            "entities_by_type": self.entities_by_type,
            "document_types": self.document_types,
            "threat_types": self.threat_types,
            "consolidated_timeline": self.consolidated_timeline,
            "gps_coordinates": self.gps_coordinates,
            "top_objects": self.top_objects,
            "duplicates": self.duplicates,
            "quality_scores": self.quality_scores,
            "forensic_verdicts": self.forensic_verdicts,
        }


def _parse_gps_from_quality(quality_result: str) -> tuple:
    """Extrai (lat, lon) da linha 'GPS: <lat>, <lon>' do bloco de qualidade."""
    for line in quality_result.splitlines():
        line = line.strip()
        if not line.startswith("GPS:"):
            continue
        try:
            lat_str, lon_str = line[4:].split(",", 1)
            return float(lat_str.strip()), float(lon_str.strip())
        except (ValueError, IndexError):
            return None, None
    return None, None


def _parse_quality_score(quality_result: str) -> Optional[float]:
    """Extrai o score composto da linha 'Score de qualidade (0-100): NN'."""
    for line in quality_result.splitlines():
        if "Score de qualidade" in line and ":" in line:
            try:
                return float(line.rsplit(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def report_to_summary_entry(report, analysis=None) -> Dict[str, Any]:
    """Converte um ImageAnalysisReport do pipeline no dict esperado por `build`.

    Antes, a CLI montava esse dict com campos vazios e adivinhava o nome do .md, então o
    resumo saía sem entidades, timeline, GPS nem objetos. Com o relatório real em mãos,
    UI e CLI passam a usar exatamente a mesma conversão.
    """
    if analysis is None:
        analysis = next((a for a in report.analyses if a.success), None)

    quality_result = report.quality_result or ""
    gps_lat, gps_lon = _parse_gps_from_quality(quality_result)

    return {
        "image_name": report.image.name,
        "success": bool(analysis and analysis.success),
        "post_processing": (analysis.post_processing if analysis else {}) or {},
        "quality_result": quality_result,
        "yolo_result": report.yolo_result or "",
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "quality_score": _parse_quality_score(quality_result),
    }


class BatchSummaryBuilder:
    """Constrói um BatchSummary a partir de resultados individuais."""

    def build(
        self,
        results: List[Dict[str, Any]],
        duplicate_result: Optional[Any] = None,
    ) -> BatchSummary:
        """
        Constrói sumário consolidado.

        Args:
            results: Lista de dicionários com resultados por imagem.
                Cada dict deve ter: image_name, success, post_processing,
                quality_result, yolo_result, gps_lat, gps_lon, quality_score.
            duplicate_result: Resultado opcional do DuplicateDetector.

        Returns:
            BatchSummary consolidado.
        """
        summary = BatchSummary()
        summary.total_images = len(results)

        all_entities = []
        all_timeline = []
        obj_counter = Counter()
        doc_counter = Counter()
        threat_counter = Counter()
        forensic_counter = Counter()

        for r in results:
            image_name = r.get("image_name", "unknown")

            if r.get("success", False):
                summary.successful += 1
            else:
                summary.failed += 1

            # Entidades
            pp = r.get("post_processing", {})
            if pp:
                entities = pp.get("entities", [])
                for ent in entities:
                    ent_with_img = {**ent, "image": image_name}
                    all_entities.append(ent_with_img)

                # Classificação
                classification = pp.get("classification", {})
                if classification:
                    doc_type = classification.get("document_type", "")
                    if doc_type and doc_type != "nao_identificado":
                        doc_counter[doc_type] += 1
                    threat_type = classification.get("threat_type", "")
                    if threat_type and threat_type != "nenhum":
                        threat_counter[threat_type] += 1

                # Timeline
                timeline = pp.get("timeline", [])
                for event in timeline:
                    event_dict = {
                        "timestamp": event.get("timestamp", ""),
                        "image": image_name,
                        "description": event.get("description", ""),
                        "source": event.get("source", ""),
                    }
                    all_timeline.append(event_dict)

            # GPS
            gps_lat = r.get("gps_lat")
            gps_lon = r.get("gps_lon")
            if gps_lat is not None and gps_lon is not None:
                summary.gps_coordinates.append({
                    "image": image_name,
                    "lat": gps_lat,
                    "lon": gps_lon,
                    "osm_url": f"https://www.openstreetmap.org/?mlat={gps_lat:.6f}&mlon={gps_lon:.6f}#map=17/{gps_lat:.6f}/{gps_lon:.6f}",
                })

            # Quality score
            qs = r.get("quality_score")
            if qs is not None:
                summary.quality_scores.append(qs)

            # YOLO objects
            yolo_text = r.get("yolo_result", "")
            if yolo_text and "YOLO não executado" not in yolo_text:
                for line in yolo_text.split("\n"):
                    line = line.strip()
                    if line.startswith("- ") and "x" in line:
                        # Formato: "- 2x person"
                        parts = line[2:].split("x", 1)
                        if len(parts) == 2:
                            try:
                                count = int(parts[0].strip())
                                obj_name = parts[1].strip()
                                obj_counter[obj_name] += count
                            except ValueError:
                                pass

            # Forensic verdicts
            quality_text = r.get("quality_result", "")
            if quality_text and "DADOS FORENSES AUTOMATIZADOS" in quality_text:
                for line in quality_text.split("\n"):
                    line = line.strip()
                    if "Veredicto ELA:" in line:
                        verdict = line.replace("Veredicto ELA:", "").strip()
                        forensic_counter[f"ELA:{verdict}"] += 1
                    elif "JPEG Ghost:" in line:
                        verdict = line.replace("JPEG Ghost:", "").strip()
                        forensic_counter[f"Ghost:{verdict}"] += 1
                    elif "Copy-Move:" in line:
                        verdict = line.replace("Copy-Move:", "").strip()
                        forensic_counter[f"Clone:{verdict}"] += 1

        # Agregar entidades
        entity_by_type = {}
        entity_counts = Counter()
        for ent in all_entities:
            etype = ent.get("type", "unknown")
            entity_by_type.setdefault(etype, []).append(ent)
            entity_counts[etype] += 1

        summary.entities_by_type = {k: v for k, v in entity_by_type.items()}
        summary.entity_counts = dict(entity_counts)
        summary.document_types = dict(doc_counter)
        summary.threat_types = dict(threat_counter)
        summary.top_objects = dict(obj_counter)
        summary.forensic_verdicts = dict(forensic_counter)

        # Ordenar timeline
        all_timeline.sort(key=lambda e: e.get("timestamp", ""))
        summary.consolidated_timeline = all_timeline

        # Duplicatas
        if duplicate_result and hasattr(duplicate_result, "pairs"):
            for pair in duplicate_result.pairs:
                summary.duplicates.append({
                    "image_a": pair.image_a,
                    "image_b": pair.image_b,
                    "similarity": pair.similarity,
                    "same_md5": pair.same_md5,
                    "phash_distance": pair.phash_distance,
                })

        return summary
