#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Post Processor - Pós-Processamento Estruturado de Análises
==============================================================
Extração de dados estruturados, normalização, classificação,
validação OCR vs LLM e geração de relatórios enriquecidos.
"""

import re
import logging
import unicodedata
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class ExtractedEntity:
    """Uma entidade extraída do texto."""
    type: str          # cpf, phone, email, ip, url
    value: str         # valor original
    normalized: str    # valor normalizado
    valid: bool        # se passou na validação
    source: str        # "ocr" ou "llm"
    context: str = ""  # texto ao redor


@dataclass
class ClassificationResult:
    """Resultado da classificação automática."""
    document_type: str = "indefinido"
    threat_type: str = "nenhuma"
    context_tags: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ValidationResult:
    """Resultado da validação OCR vs LLM."""
    ocr_text: str = ""
    llm_text: str = ""
    consistency_score: float = 0.0
    inconsistencies: List[str] = field(default_factory=list)
    verdict: str = "nao_avaliado"


@dataclass
class TimelineEvent:
    """Evento na linha do tempo."""
    timestamp: str
    parsed_dt: Optional[datetime] = None
    description: str = ""
    source: str = ""


@dataclass
class PostProcessingResult:
    """Resultado completo do pós-processamento."""
    entities: List[ExtractedEntity] = field(default_factory=list)
    classification: Optional[ClassificationResult] = None
    validation: Optional[ValidationResult] = None
    timeline: List[TimelineEvent] = field(default_factory=list)
    normalized_text: str = ""
    summary: str = ""


# ============================================================================
# REGEX PATTERNS
# ============================================================================

# CPF: 000.000.000-00 ou 00000000000
CPF_PATTERN = re.compile(
    r'\b(\d{3}[.\s]?\d{3}[.\s]?\d{3}[-.\s]?\d{2})\b'
)

# Telefone BR: (11) 99999-9999, 11999999999, +5511999999999
PHONE_PATTERN = re.compile(
    r'(?:\+?55\s?)?'
    r'(?:\(?\d{2}\)?[\s.-]?)'
    r'(?:9\d{4}|\d{4})[\s.-]?\d{4}\b'
)

# E-mail
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
)

# IPv4
IPV4_PATTERN = re.compile(
    r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
)

# IPv6 (simplificado)
IPV6_PATTERN = re.compile(
    r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
    r'|'
    r'\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b'
    r'|'
    r'\b::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}\b'
)

# URLs
URL_PATTERN = re.compile(
    r'https?://[^\s<>"\'\])}]+|'
    r'www\.[^\s<>"\'\])}]+'
)

# Timestamps diversos
TIMESTAMP_PATTERNS = [
    # 2024-01-15 14:30:00
    re.compile(r'\b(\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}(?::\d{2})?)\b'),
    # 15/01/2024 14:30
    re.compile(r'\b(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}(?::\d{2})?)\b'),
    # 14:30:00 ou 14:30
    re.compile(r'\b(\d{2}:\d{2}(?::\d{2})?)\b'),
    # Jan 15, 2024
    re.compile(r'\b([A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4})\b'),
]

# Placa BR (Mercosul e antiga)
PLATE_PATTERN = re.compile(
    r'\b([A-Z]{3}[-\s]?\d[A-Z0-9]\d{2})\b', re.IGNORECASE
)


# ============================================================================
# VALIDAÇÃO
# ============================================================================

def validate_cpf(cpf: str) -> bool:
    """Valida CPF usando dígitos verificadores."""
    digits = re.sub(r'\D', '', cpf)
    if len(digits) != 11:
        return False
    if digits == digits[0] * 11:
        return False

    # Primeiro dígito verificador
    total = sum(int(digits[i]) * (10 - i) for i in range(9))
    remainder = total % 11
    d1 = 0 if remainder < 2 else 11 - remainder
    if int(digits[9]) != d1:
        return False

    # Segundo dígito verificador
    total = sum(int(digits[i]) * (11 - i) for i in range(10))
    remainder = total % 11
    d2 = 0 if remainder < 2 else 11 - remainder
    return int(digits[10]) == d2


def validate_email(email: str) -> bool:
    """Validação básica de e-mail."""
    return bool(re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$', email))


def validate_ip(ip: str) -> bool:
    """Valida endereço IPv4."""
    parts = ip.split('.')
    if len(parts) != 4:
        return False
    return all(0 <= int(p) <= 255 for p in parts if p.isdigit())


def validate_phone(phone: str) -> bool:
    """Validação básica de telefone BR."""
    digits = re.sub(r'\D', '', phone)
    return 10 <= len(digits) <= 13


# ============================================================================
# POST-PROCESSOR
# ============================================================================

class PostProcessor:
    """Pós-processador de análises de imagem."""

    # Classificação de documentos
    DOC_TYPE_KEYWORDS = {
        "rg": ["registro geral", "identidade", "rg ", "carteira de identidade"],
        "cpf": ["cadastro de pessoa", "cpf"],
        "cnh": ["carteira de habilitação", "cnh", "habilitação"],
        "comprovante": ["comprovante", "recibo", "nota fiscal", "nf-e", "boleto"],
        "contrato": ["contrato", "acordo", "termo de"],
        "screenshot": ["captura de tela", "screenshot", "print", "whatsapp", "telegram", "instagram"],
        "foto_documento": ["documento", "certidão", "alvará"],
        "foto_veiculo": ["placa", "veículo", "carro", "moto"],
        "foto_pessoa": ["pessoa", "indivíduo", "suspeito", "rosto"],
        "foto_local": ["local", "cena", "ambiente", "fachada"],
    }

    # Classificação de ameaças/golpes
    THREAT_KEYWORDS = {
        "phishing": ["phishing", "falso", "clique aqui", "senha", "atualizar cadastro", "urgente"],
        "fraude_financeira": ["pix", "transferência", "boleto falso", "código de barras", "qr code"],
        "ameaca": ["ameaça", "matar", "morrer", "vou te", "cuidado", "aviso"],
        "extorsao": ["extorsão", "pagar", "resgate", "vazamento", "nude", "íntim"],
        "falsificacao": ["falsificação", "adulterado", "manipulado", "editado"],
        "estelionato": ["estelionato", "golpe", "pirâmide", "investimento falso", "lucro garantido"],
        "cyberbullying": ["bullying", "humilhação", "ofensa", "xingamento"],
        "drogas": ["droga", "maconha", "cocaína", "tráfico", "substância"],
        "armas": ["arma", "pistola", "revólver", "fuzil", "munição"],
    }

    def __init__(self):
        pass

    def process(
        self,
        ocr_text: str = "",
        llm_analysis: str = "",
        exif_data: str = "",
        yolo_result: str = "",
        quality_result: str = ""
    ) -> PostProcessingResult:
        """
        Executa pipeline completo de pós-processamento.

        Args:
            ocr_text: Texto bruto do OCR
            llm_analysis: Análise gerada pelo LLM
            exif_data: Metadados EXIF extraídos
            yolo_result: Resultado do YOLO
            quality_result: Resultado de qualidade da imagem

        Returns:
            PostProcessingResult com todos os dados estruturados
        """
        result = PostProcessingResult()

        # 1. Normalizar texto OCR
        result.normalized_text = self.normalize_text(ocr_text)

        # 2. Extrair entidades estruturadas
        combined_text = f"{ocr_text}\n{llm_analysis}"
        result.entities = self.extract_entities(combined_text)

        # 3. Classificar documento e ameaça
        result.classification = self.classify(combined_text)

        # 4. Validar OCR vs LLM
        result.validation = self.validate_ocr_vs_llm(ocr_text, llm_analysis)

        # 5. Extrair timeline
        all_text = f"{ocr_text}\n{llm_analysis}\n{exif_data}"
        result.timeline = self.extract_timeline(all_text)

        # 6. Gerar resumo
        result.summary = self.generate_summary(result, yolo_result, quality_result)

        return result

    # ========================================================================
    # EXTRAÇÃO DE ENTIDADES
    # ========================================================================

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extrai todas as entidades estruturadas do texto."""
        entities = []

        # CPFs
        for match in CPF_PATTERN.finditer(text):
            value = match.group(1)
            normalized = re.sub(r'\D', '', value)
            if len(normalized) == 11:
                entities.append(ExtractedEntity(
                    type="cpf",
                    value=value,
                    normalized=f"{normalized[:3]}.{normalized[3:6]}.{normalized[6:9]}-{normalized[9:]}",
                    valid=validate_cpf(value),
                    source="mixed",
                    context=text[max(0, match.start()-30):match.end()+30]
                ))

        # Telefones
        for match in PHONE_PATTERN.finditer(text):
            value = match.group()
            digits = re.sub(r'\D', '', value)
            entities.append(ExtractedEntity(
                type="phone",
                value=value,
                normalized=digits,
                valid=validate_phone(value),
                source="mixed",
                context=text[max(0, match.start()-30):match.end()+30]
            ))

        # E-mails
        for match in EMAIL_PATTERN.finditer(text):
            value = match.group()
            entities.append(ExtractedEntity(
                type="email",
                value=value,
                normalized=value.lower(),
                valid=validate_email(value),
                source="mixed",
                context=text[max(0, match.start()-30):match.end()+30]
            ))

        # IPs (v4)
        for match in IPV4_PATTERN.finditer(text):
            value = match.group()
            entities.append(ExtractedEntity(
                type="ipv4",
                value=value,
                normalized=value,
                valid=validate_ip(value),
                source="mixed",
                context=text[max(0, match.start()-30):match.end()+30]
            ))

        # IPs (v6)
        for match in IPV6_PATTERN.finditer(text):
            value = match.group()
            entities.append(ExtractedEntity(
                type="ipv6",
                value=value,
                normalized=value.lower(),
                valid=True,
                source="mixed"
            ))

        # URLs
        for match in URL_PATTERN.finditer(text):
            value = match.group()
            entities.append(ExtractedEntity(
                type="url",
                value=value,
                normalized=value,
                valid=True,
                source="mixed",
                context=text[max(0, match.start()-30):match.end()+30]
            ))

        # Placas veiculares BR
        for match in PLATE_PATTERN.finditer(text):
            value = match.group(1).upper()
            normalized = re.sub(r'[-\s]', '', value)
            entities.append(ExtractedEntity(
                type="plate",
                value=value,
                normalized=normalized,
                valid=len(normalized) == 7,
                source="mixed",
                context=text[max(0, match.start()-30):match.end()+30]
            ))

        return entities

    # ========================================================================
    # NORMALIZAÇÃO DE TEXTO
    # ========================================================================

    def normalize_text(self, text: str) -> str:
        """Normaliza texto OCR: corrige encoding, remove lixo, limpa."""
        if not text or text.startswith("["):
            return text

        # Corrigir encoding comum
        replacements = {
            'Ã£': 'ã', 'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó',
            'Ãº': 'ú', 'Ã§': 'ç', 'Ã¢': 'â', 'Ãª': 'ê', 'Ã´': 'ô',
            'Ã': 'à', 'Ã€': 'À', 'Ã‰': 'É', 'Ã"': 'Ó',
            '\x00': '', '\ufffd': '', '\u200b': '', '\u200c': '',
            '\ufeff': '', '\xa0': ' ',
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)

        # Normalizar Unicode
        text = unicodedata.normalize('NFC', text)

        # Remover caracteres de controle (exceto newline e tab)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # Remover lixo OCR comum (sequências repetidas de caracteres especiais)
        text = re.sub(r'[|]{3,}', '', text)
        text = re.sub(r'[-]{5,}', '---', text)
        text = re.sub(r'[_]{5,}', '___', text)
        text = re.sub(r'[=]{5,}', '===', text)
        text = re.sub(r'[~]{3,}', '', text)

        # Remover linhas que são apenas pontuação/símbolos
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not re.match(r'^[^a-zA-Z0-9À-ÿ]{3,}$', stripped):
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # Normalizar espaços em branco
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    # ========================================================================
    # CLASSIFICAÇÃO
    # ========================================================================

    def classify(self, text: str) -> ClassificationResult:
        """Classifica o tipo de documento e tipo de ameaça."""
        result = ClassificationResult()
        text_lower = text.lower()

        # Classificar tipo de documento
        best_doc_score = 0
        for doc_type, keywords in self.DOC_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_doc_score:
                best_doc_score = score
                result.document_type = doc_type

        # Classificar tipo de ameaça
        best_threat_score = 0
        for threat_type, keywords in self.THREAT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_threat_score:
                best_threat_score = score
                result.threat_type = threat_type

        if best_threat_score == 0:
            result.threat_type = "nenhuma"

        # Tags de contexto
        context_keywords = {
            "financeiro": ["banco", "pix", "transferência", "dinheiro", "r$", "real", "pagamento"],
            "violencia": ["arma", "faca", "sangue", "ferimento", "agressão"],
            "digital": ["computador", "celular", "tela", "app", "site", "link"],
            "veicular": ["carro", "moto", "placa", "veículo", "trânsito"],
            "noturno": ["noite", "escuro", "madrugada", "iluminação artificial"],
            "residencial": ["casa", "apartamento", "residência", "condomínio"],
            "comercial": ["loja", "empresa", "escritório", "comércio"],
            "via_publica": ["rua", "avenida", "estrada", "calçada"],
        }

        for tag, keywords in context_keywords.items():
            if any(kw in text_lower for kw in keywords):
                result.context_tags.append(tag)

        # Confiança baseada em quantas keywords foram encontradas
        total_hits = best_doc_score + best_threat_score + len(result.context_tags)
        result.confidence = min(1.0, total_hits / 5.0)

        return result

    # ========================================================================
    # VALIDAÇÃO OCR vs LLM
    # ========================================================================

    def validate_ocr_vs_llm(self, ocr_text: str, llm_text: str) -> ValidationResult:
        """Compara texto OCR com análise LLM para detectar inconsistências."""
        result = ValidationResult(ocr_text=ocr_text, llm_text=llm_text)

        if not ocr_text or ocr_text.startswith("[") or not llm_text:
            result.verdict = "nao_avaliado"
            return result

        ocr_lower = ocr_text.lower()
        llm_lower = llm_text.lower()

        # Extrair "palavras significativas" do OCR (>3 chars, não stopwords)
        stopwords_pt = {
            "que", "para", "com", "por", "uma", "dos", "das", "nos",
            "nas", "não", "mais", "como", "mas", "foi", "são", "tem",
            "ser", "ter", "está", "esta", "esse", "isso",
        }

        ocr_words = set(
            w for w in re.findall(r'\b[a-záàâãéèêíïóôõúüç]{4,}\b', ocr_lower)
            if w not in stopwords_pt
        )

        if not ocr_words:
            result.verdict = "nao_avaliado"
            return result

        # Contar quantas palavras do OCR aparecem no LLM
        found = sum(1 for w in ocr_words if w in llm_lower)
        total = len(ocr_words)
        result.consistency_score = found / total if total > 0 else 0

        # Detectar inconsistências específicas
        # 1. Números/códigos no OCR que não aparecem no LLM
        ocr_numbers = set(re.findall(r'\b\d{3,}\b', ocr_text))
        for num in ocr_numbers:
            if num not in llm_text:
                result.inconsistencies.append(
                    f"Número '{num}' presente no OCR mas ausente na análise LLM"
                )

        # 2. Entidades no OCR que deveriam estar no LLM
        ocr_entities = self.extract_entities(ocr_text)
        llm_entities = self.extract_entities(llm_text)

        ocr_entity_values = {e.normalized for e in ocr_entities}
        llm_entity_values = {e.normalized for e in llm_entities}

        missing = ocr_entity_values - llm_entity_values
        for val in missing:
            result.inconsistencies.append(
                f"Entidade '{val}' detectada no OCR mas não mencionada pelo LLM"
            )

        # Veredicto
        if result.consistency_score >= 0.7 and len(result.inconsistencies) <= 1:
            result.verdict = "consistente"
        elif result.consistency_score >= 0.4:
            result.verdict = "parcialmente_consistente"
        else:
            result.verdict = "inconsistente"

        return result

    # ========================================================================
    # TIMELINE
    # ========================================================================

    def extract_timeline(self, text: str) -> List[TimelineEvent]:
        """Extrai eventos ordenados cronologicamente do texto."""
        events = []

        for pattern in TIMESTAMP_PATTERNS:
            for match in pattern.finditer(text):
                ts_str = match.group(1) if match.lastindex else match.group()
                parsed = self._parse_timestamp(ts_str)

                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 100)
                context = text[context_start:context_end].strip()
                # Limpar para uma linha
                context = re.sub(r'\s+', ' ', context)

                events.append(TimelineEvent(
                    timestamp=ts_str,
                    parsed_dt=parsed,
                    description=context[:150],
                    source="text"
                ))

        # Ordenar por timestamp quando possível
        events.sort(key=lambda e: e.parsed_dt or datetime.max)

        # Deduplicar (mesmo timestamp)
        seen = set()
        unique = []
        for e in events:
            if e.timestamp not in seen:
                seen.add(e.timestamp)
                unique.append(e)

        return unique

    def _parse_timestamp(self, ts: str) -> Optional[datetime]:
        """Tenta parsear um timestamp em vários formatos."""
        formats = [
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
            "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M",
            "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(ts.strip(), fmt)
            except ValueError:
                continue
        return None

    # ========================================================================
    # GERAÇÃO DE RESUMO
    # ========================================================================

    def generate_summary(
        self,
        result: PostProcessingResult,
        yolo_result: str = "",
        quality_result: str = ""
    ) -> str:
        """Gera resumo estruturado do pós-processamento."""
        lines = []

        # Entidades
        if result.entities:
            lines.append("## 📌 Entidades Extraídas")
            lines.append("")
            lines.append("| Tipo | Valor | Válido |")
            lines.append("|------|-------|--------|")
            for ent in result.entities:
                icon = {
                    "cpf": "🪪", "phone": "📞", "email": "📧",
                    "ipv4": "🌐", "ipv6": "🌐", "url": "🔗", "plate": "🚗"
                }.get(ent.type, "📎")
                valid_str = "✅" if ent.valid else "❌"
                lines.append(f"| {icon} {ent.type.upper()} | `{ent.normalized}` | {valid_str} |")
            lines.append("")

        # Classificação
        if result.classification:
            cls = result.classification
            lines.append("## 🏷️ Classificação Automática")
            lines.append("")
            lines.append(f"- **Tipo de documento:** {cls.document_type}")
            lines.append(f"- **Tipo de ameaça:** {cls.threat_type}")
            if cls.context_tags:
                lines.append(f"- **Contexto:** {', '.join(cls.context_tags)}")
            lines.append(f"- **Confiança:** {cls.confidence:.0%}")
            lines.append("")

        # Validação OCR vs LLM
        if result.validation and result.validation.verdict != "nao_avaliado":
            val = result.validation
            verdict_icon = {
                "consistente": "✅", "parcialmente_consistente": "⚠️",
                "inconsistente": "❌"
            }.get(val.verdict, "❓")
            lines.append("## ✅ Validação OCR vs LLM")
            lines.append("")
            lines.append(f"- **Veredicto:** {verdict_icon} {val.verdict.replace('_', ' ').title()}")
            lines.append(f"- **Score de consistência:** {val.consistency_score:.0%}")
            if val.inconsistencies:
                lines.append("- **Inconsistências detectadas:**")
                for inc in val.inconsistencies[:5]:
                    lines.append(f"  - ⚠️ {inc}")
            lines.append("")

        # Timeline
        if result.timeline:
            lines.append("## 🕐 Linha do Tempo")
            lines.append("")
            for event in result.timeline[:20]:
                lines.append(f"- **{event.timestamp}** — {event.description}")
            lines.append("")

        return "\n".join(lines) if lines else ""

    def to_dict(self, result: PostProcessingResult) -> Dict[str, Any]:
        """Converte o resultado em um dict serializável para exportação."""
        return {
            "summary": result.summary,
            "normalized_text": result.normalized_text,
            "entities": [
                {
                    "type": entity.type,
                    "value": entity.value,
                    "normalized": entity.normalized,
                    "valid": entity.valid,
                    "source": entity.source,
                    "context": entity.context,
                }
                for entity in result.entities
            ],
            "classification": (
                {
                    "document_type": result.classification.document_type,
                    "threat_type": result.classification.threat_type,
                    "context_tags": result.classification.context_tags,
                    "confidence": result.classification.confidence,
                }
                if result.classification
                else None
            ),
            "validation": (
                {
                    "ocr_text": result.validation.ocr_text,
                    "llm_text": result.validation.llm_text,
                    "consistency_score": result.validation.consistency_score,
                    "inconsistencies": result.validation.inconsistencies,
                    "verdict": result.validation.verdict,
                }
                if result.validation
                else None
            ),
            "timeline": [
                {
                    "timestamp": event.timestamp,
                    "parsed_dt": event.parsed_dt.isoformat() if event.parsed_dt else None,
                    "description": event.description,
                    "source": event.source,
                }
                for event in result.timeline
            ],
        }

    def format_report_section(self, result: PostProcessingResult) -> str:
        """Formata seção para inclusão em relatório final."""
        sections = []
        
        sections.append("---")
        sections.append("")
        sections.append("## 🔧 Pós-Processamento Automático")
        sections.append("")

        if result.summary:
            sections.append(result.summary)

        should_include_normalized = bool(result.normalized_text) and (
            not result.validation or result.normalized_text != result.validation.ocr_text
        )
        if should_include_normalized:
            sections.append("### 📝 Texto Normalizado")
            sections.append("```")
            sections.append(result.normalized_text[:2000])
            sections.append("```")
            sections.append("")

        return "\n".join(sections)
