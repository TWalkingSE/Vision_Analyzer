#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXIF Analyzer - Análise de consistência de metadados EXIF.

Verifica anomalias que podem indicar manipulação de metadados:
- Software indicando editor de imagem
- Timestamps inconsistentes
- Ausência de EXIF em foto aparentemente legítima
- GPS sem Make/Model
- Orientação inconsistente
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ExifTags
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False


@dataclass
class EXIFAnomaly:
    """Anomalia detectada nos metadados."""
    type: str
    severity: str  # "info", "warning", "critical"
    description: str
    field: str = ""


@dataclass
class EXIFAnalysisResult:
    """Resultado da análise de EXIF."""
    anomalies: List[EXIFAnomaly] = field(default_factory=list)
    verdict: str = "consistente"  # consistente, suspeito, stripado, inconclusivo
    has_exif: bool = False
    has_gps: bool = False
    software: str = ""
    camera_make: str = ""
    camera_model: str = ""
    datetime_original: str = ""
    datetime_modified: str = ""
    datetime_digitized: str = ""

    @property
    def critical_count(self) -> int:
        return sum(1 for a in self.anomalies if a.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for a in self.anomalies if a.severity == "warning")

    def get_summary(self) -> str:
        if not self.has_exif:
            return "EXIF ausente — metadados podem ter sido removidos."

        lines = [f"Veredicto EXIF: {self.verdict}"]
        if self.camera_make or self.camera_model:
            lines.append(f"Câmera: {self.camera_make} {self.camera_model}".strip())
        if self.software:
            lines.append(f"Software: {self.software}")
        if self.has_gps:
            lines.append("GPS: presente")
        if self.datetime_original:
            lines.append(f"Data original: {self.datetime_original}")

        if self.anomalies:
            lines.append(f"Anomalias: {self.critical_count} críticas, {self.warning_count} avisos")
            for a in self.anomalies:
                icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(a.severity, "ℹ️")
                lines.append(f"  {icon} [{a.type}] {a.description}")

        return "\n".join(lines)


# Softwares conhecidos que indicam edição
EDITING_SOFTWARE = {
    "photoshop": "Adobe Photoshop",
    "gimp": "GIMP",
    "lightroom": "Adobe Lightroom",
    "snapseed": "Snapseed",
    "canva": "Canva",
    "pixlr": "Pixlr",
    "affinity": "Affinity Photo",
    "paint.net": "Paint.NET",
    "picsart": "PicsArt",
    "instagram": "Instagram",
    "whatsapp": "WhatsApp",
    "telegram": "Telegram",
}


class EXIFAnalyzer:
    """Analisa consistência de metadados EXIF para detectar manipulação."""

    def analyze(self, image: Image.Image, exif_data: Dict[str, Any] = None) -> EXIFAnalysisResult:
        """
        Analia EXIF em busca de inconsistências.

        Args:
            image: Imagem PIL.
            exif_data: Dicionário EXIF já extraído (opcional).

        Returns:
            EXIFAnalysisResult com anomalias e veredicto.
        """
        result = EXIFAnalysisResult()

        if not PIL_AVAILABLE:
            return result

        # Usar exif_data passado ou extrair
        if exif_data is None:
            exif_data = self._extract_exif_dict(image)

        if not exif_data:
            result.has_exif = False
            result.anomalies.append(EXIFAnomaly(
                type="no_exif",
                severity="warning",
                description="Nenhum metadado EXIF encontrado. Pode indicar strip intencional ou exportação por editor.",
            ))
            result.verdict = "stripado"
            return result

        result.has_exif = True

        # Extrair campos chave
        result.camera_make = str(exif_data.get("Make", ""))
        result.camera_model = str(exif_data.get("Model", ""))
        result.software = str(exif_data.get("Software", ""))
        result.datetime_original = str(exif_data.get("DateTimeOriginal", ""))
        result.datetime_modified = str(exif_data.get("DateTime", ""))
        result.datetime_digitized = str(exif_data.get("DateTimeDigitized", ""))

        # Verificar GPS
        gps_keys = [k for k in exif_data if "GPS" in k or "gps" in k.lower()]
        result.has_gps = bool(gps_keys)

        # 1. Verificar software de edição
        self._check_software(result)

        # 2. Verificar consistência de timestamps
        self._check_timestamps(result)

        # 3. Verificar GPS sem câmera
        self._check_gps_without_camera(result)

        # 4. Verificar orientação
        self._check_orientation(result, exif_data)

        # 5. Verificar ausência de campos básicos
        self._check_missing_fields(result)

        # Determinar veredicto
        if result.critical_count > 0:
            result.verdict = "suspeito"
        elif result.warning_count > 0:
            result.verdict = "inconclusivo"
        else:
            result.verdict = "consistente"

        return result

    def _extract_exif_dict(self, image: Image.Image) -> Dict[str, Any]:
        """Extrai EXIF da imagem como dicionário."""
        exif_data = {}
        try:
            exif = image._getexif()
            if not exif:
                return {}
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='ignore')
                    except Exception:
                        value = str(value)
                exif_data[tag_name] = value
        except Exception as e:
            logger.debug(f"Erro ao extrair EXIF: {e}")
        return exif_data

    def _check_software(self, result: EXIFAnalysisResult):
        """Verifica se o software indica editor de imagem."""
        if not result.software:
            return

        sw_lower = result.software.lower()
        for key, name in EDITING_SOFTWARE.items():
            if key in sw_lower:
                severity = "critical" if key in ("photoshop", "gimp", "affinity", "pixlr") else "warning"
                result.anomalies.append(EXIFAnomaly(
                    type="editing_software",
                    severity=severity,
                    description=f"Software de edição detectado: {result.software} ({name}).",
                    field="Software",
                ))
                return

    def _check_timestamps(self, result: EXIFAnalysisResult):
        """Verifica inconsistências entre timestamps."""
        dt_orig = result.datetime_original
        dt_mod = result.datetime_modified
        dt_dig = result.datetime_digitized

        if not dt_orig and not dt_mod:
            return

        # Parsear timestamps
        parsed_orig = self._parse_exif_time(dt_orig) if dt_orig else None
        parsed_mod = self._parse_exif_time(dt_mod) if dt_mod else None
        parsed_dig = self._parse_exif_time(dt_dig) if dt_dig else None

        # Comparar original vs modificado
        if parsed_orig and parsed_mod:
            diff = abs((parsed_mod - parsed_orig).total_seconds())
            if diff > 3600:  # Mais de 1 hora de diferença
                result.anomalies.append(EXIFAnomaly(
                    type="timestamp_mismatch",
                    severity="warning",
                    description=f"DateTimeOriginal ({dt_orig}) difere de DateTime ({dt_mod}) em {diff/3600:.1f}h.",
                    field="DateTime/DateTimeOriginal",
                ))

        # Comparar original vs digitalizado
        if parsed_orig and parsed_dig:
            diff = abs((parsed_dig - parsed_orig).total_seconds())
            if diff > 3600:
                result.anomalies.append(EXIFAnomaly(
                    type="timestamp_mismatch",
                    severity="warning",
                    description=f"DateTimeOriginal ({dt_orig}) difere de DateTimeDigitized ({dt_dig}) em {diff/3600:.1f}h.",
                    field="DateTimeOriginal/DateTimeDigitized",
                ))

        # Verificar data no futuro
        now = datetime.now()
        for dt_str, parsed, field_name in [
            (dt_orig, parsed_orig, "DateTimeOriginal"),
            (dt_mod, parsed_mod, "DateTime"),
            (dt_dig, parsed_dig, "DateTimeDigitized"),
        ]:
            if parsed and parsed > now:
                result.anomalies.append(EXIFAnomaly(
                    type="future_timestamp",
                    severity="critical",
                    description=f"{field_name} ({dt_str}) está no futuro.",
                    field=field_name,
                ))

    def _check_gps_without_camera(self, result: EXIFAnalysisResult):
        """GPS presente sem Make/Model é incomum."""
        if result.has_gps and not result.camera_make and not result.camera_model:
            result.anomalies.append(EXIFAnomaly(
                type="gps_no_camera",
                severity="warning",
                description="GPS presente mas sem Make/Model da câmera. Incomum em fotos legítimas.",
                field="GPS/Make/Model",
            ))

    def _check_orientation(self, result: EXIFAnalysisResult, exif_data: Dict[str, Any]):
        """Verifica inconsistência de orientação."""
        orientation = exif_data.get("Orientation", 1)
        if isinstance(orientation, bytes):
            try:
                orientation = int.from_bytes(orientation, 'little')
            except Exception:
                orientation = 1

        if orientation and orientation > 1:
            # Se orientação não é 1 mas a imagem aparenta estar correta,
            # pode indicar que foi rotacionada por editor mas EXIF não atualizado
            result.anomalies.append(EXIFAnomaly(
                type="orientation_flag",
                severity="info",
                description=f"Orientation flag = {orientation}. Imagem pode ter sido rotacionada manualmente.",
                field="Orientation",
            ))

    def _check_missing_fields(self, result: EXIFAnalysisResult):
        """Verifica ausência de campos básicos em foto com EXIF."""
        if result.has_exif and not result.camera_make and not result.camera_model and not result.software:
            result.anomalies.append(EXIFAnomaly(
                type="missing_basic_fields",
                severity="warning",
                description="EXIF presente mas sem Make, Model ou Software. Pode indicar edição seletiva de metadados.",
                field="Make/Model/Software",
            ))

    @staticmethod
    def _parse_exif_time(ts: str) -> Optional[datetime]:
        """Tenta parsear timestamp EXIF (formato 'YYYY:MM:DD HH:MM:SS')."""
        formats = [
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y:%m:%d %H:%M",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(ts.strip(), fmt)
            except (ValueError, AttributeError):
                continue
        return None
