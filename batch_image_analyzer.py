#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖼️ Batch Image Analyzer - Sistema de Análise de Imagens em Lote
================================================================
Análise de imagens usando múltiplos modelos de IA (OpenAI + Ollama)
com suporte a RAW, HEIC e OCR integrado.

Author: Vision Analyzer Pro
Version: 1.0.0
Python: 3.10+
"""

import os
import re
import sys
import base64
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from runtime_config import (
    GPU_MODEL_PROFILES,
    HEIF_EXTENSIONS,
    JPEG_QUALITY,
    MAX_IMAGE_SIZE,
    OCR_MODEL,
    OCR_MODEL_ALT,
    OLLAMA_VISION_PREFIXES,
    OPENAI_MODEL,
    RAW_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    detect_vram_gb,
    get_model_short_name as shared_get_model_short_name,
    get_recommended_gpu_profile,
    sanitize_filename as shared_sanitize_filename,
)
from batch_checkpoint import (
    BatchCheckpointManager,
    build_batch_job_config,
    build_batch_signature,
    get_default_checkpoint_path,
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DEPENDÊNCIAS E IMPORTS
# ============================================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("⚠️ python-dotenv não instalado. Usando variáveis de ambiente do sistema.")

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # Permite imagens grandes
except ImportError:
    logger.error("❌ Pillow não instalado. Execute: pip install Pillow")
    sys.exit(1)

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False
    logger.warning("⚠️ pillow-heif não instalado. Suporte a HEIC/HEIF desabilitado.")

try:
    import rawpy
    import imageio
    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False
    logger.warning("⚠️ rawpy/imageio não instalados. Suporte a RAW desabilitado.")

try:
    from object_detector import get_detector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("⚠️ YOLO não disponível.")

try:
    from object_detector import get_detectron2_detector, is_detectron2_available
    DETECTRON2_AVAILABLE = is_detectron2_available()
except ImportError:
    DETECTRON2_AVAILABLE = False

try:
    from image_preprocessor import auto_fix_image, quick_analyze, binarize_for_ocr
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    logger.warning("⚠️ Image Preprocessor não disponível.")

try:
    from post_processor import PostProcessor
    POST_PROCESSOR_AVAILABLE = True
except ImportError:
    POST_PROCESSOR_AVAILABLE = False
    logger.warning("⚠️ Post Processor não disponível.")

try:
    from prompt_templates import get_prompt_manager
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    logger.warning("⚠️ Prompt Templates não disponível. Usando prompts locais reduzidos.")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("⚠️ openai não instalado. Análise via GPT-5.4-mini desabilitada.")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("⚠️ ollama não instalado. Análise via modelos locais desabilitada.")

try:
    from analysis_pipeline import AnalysisPipeline
    SHARED_PIPELINE_AVAILABLE = True
except ImportError:
    SHARED_PIPELINE_AVAILABLE = False
    AnalysisPipeline = None


def _detect_gpu() -> bool:
    """Auto-detecta se CUDA está disponível para aceleração GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    # Fallback: nvidia-smi detecta GPU mesmo sem torch CUDA
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except Exception:
        return False


# ============================================================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================================================

# Diretórios padrão
INPUT_DIR = Path("./imagens_entrada")
OUTPUT_DIR = Path("./relatorios_saida")

# Opções de OCR
OCR_ENGINES = ["glm-ocr", "olmocr2", "none"]


# ============================================================================
# PROMPTS DE SISTEMA
# ============================================================================

# Prompt padrão para análise geral de imagens
PROMPT_GERAL = """# SYSTEM ROLE
Atue como um Especialista Sênior em Visão Computacional e Semiótica. Sua função é converter informações visuais em descrições textuais de alta fidelidade, acessíveis e tecnicamente precisas.

# CONTEXTO E OBJETIVO
Você receberá uma imagem que pode conter pessoas, cenários, objetos, documentos, paisagens ou qualquer combinação desses elementos.

**DADO DE ENTRADA OCR (PRÉ-PROCESSADO):** O sistema detectou o seguinte texto bruto na imagem:
---
{ocr_result}
---
(Valide este texto com sua visão. Se o OCR estiver vazio ou errado, corrija na sua análise).

Seu objetivo é fornecer uma análise que sirva tanto para interpretação humana quanto para acessibilidade (leitores de tela), garantindo que nenhum detalhe visual crucial seja omitido.

# 🧱 INSTRUÇÕES DE EXECUÇÃO (STEP-BY-STEP)

<passo_1>
**Classificação e Análise Preliminar**
- Classifique o tipo predominante da imagem: retrato, grupo, paisagem, objeto/produto, documento, captura de tela, arte/ilustração, misto.
- Valide o texto do OCR fornecido acima comparando com a imagem.
- Identifique os 3 elementos visuais mais salientes (o que chama mais atenção).
</passo_1>

<passo_2>
**Geração de Acessibilidade (Alt Text)**
- Escreva um parágrafo único e conciso (máximo 300 caracteres) que descreva a essência da imagem para uma pessoa com deficiência visual.
- Comece pelo elemento mais importante e inclua o contexto funcional ("para que serve" a imagem).
- NÃO comece com "Imagem de..." ou "Foto de..." (redundante para leitores de tela).
</passo_2>

<passo_3>
**Descrição Detalhada (O Core da Análise)**
Utilize uma abordagem de varredura (do foco principal para o fundo):

**Se houver pessoas:**
1. **Sujeitos (Pessoas):** Descreva aparência física (cabelo, tom de pele aproximado), vestuário (cores, texturas), postura corporal e micro-expressões faciais.
2. **Interações:** Descreva como os sujeitos interagem entre si ou com objetos (ex: "segurando uma caneta", "olhando para o horizonte").

**Se houver objetos/elementos (sempre):**
3. **Objetos e Elementos:** Identifique e descreva objetos significativos, suas posições relativas (esquerda/direita, primeiro plano/fundo), materiais aparentes e estado.
4. **Relações Espaciais:** Descreva como os elementos se relacionam no espaço ("à esquerda de", "acima de", "sobreposto a").

**Ambiente (sempre):**
5. **Cenário e Contexto:** Detalhe o cenário, mobiliário, clima atmosférico e contexto temporal (dia/noite/época). Se for um ambiente interno, descreva o espaço. Se externo, descreva a paisagem.
</passo_3>

<passo_4>
**Análise Técnica Fotográfica**
- Iluminação (dura/suave, direção, fonte provável)
- Enquadramento (close-up, plano americano, plano geral, etc.)
- Foco (profundidade de campo, ponto focal)
- Paleta de cores predominante (tons quentes/frios, saturação)
- Composição (regra dos terços, simetria, linhas-guia)
</passo_4>

<passo_5>
**Nível de Confiança**
- Para cada observação ambígua, indique explicitamente o nível de certeza: [CERTO], [PROVÁVEL] ou [INCERTO].
</passo_5>

# 🚫 GUARDRAILS E SEGURANÇA (CRÍTICO)

1. **Zero Alucinação:** Descreva APENAS o que é visível. Se algo estiver ambíguo, declare a incerteza (ex: "objeto indistinto na mão esquerda [INCERTO]") em vez de adivinhar.
2. **Mitigação de Viés:** Ao descrever pessoas, foque em traços observáveis. Evite assumir:
   - Relações parentais (use "mulher e criança" em vez de "mãe e filho").
   - Profissões (use "vestindo jaleco branco" em vez de "médico").
   - Emoções subjetivas (use "sorrindo" em vez de "feliz").
3. **Segurança de PII:** Se houver rostos nítidos que pareçam ser de pessoas não públicas, não tente nomeá-las.
4. **Completude:** Não pule seções. Se uma seção não se aplica, escreva "Não aplicável a esta imagem".

# 📝 FORMATO DE SAÍDA (MARKDOWN)

A resposta deve seguir estritamente este template:

## 👁️ Texto Alternativo (Acessibilidade)
[Texto curto para leitores de tela — máx. 300 caracteres]

## 🏷️ Classificação
* **Tipo:** [retrato / grupo / paisagem / objeto / documento / captura de tela / arte / misto]
* **Elementos Salientes:** [3 elementos mais importantes]

## 🕵️ Análise Detalhada
### Sujeitos e Ação
* **[Sujeito Principal]:** [Descrição rica]
* **[Sujeitos Secundários]:** [Descrição]
* **Vestuário e Adereços:** [Detalhes de texturas e cores]

### Objetos e Elementos
* **[Objeto]:** [Descrição, posição, estado]

### Cenário e Contexto
[Descrição do ambiente, fundo e elementos espaciais]

### Transcrição de Texto (OCR Validado)
* *Texto detectado:* "[Inserir texto validado ou 'Nenhum texto visível']"

## 📸 Dados Técnicos
| Elemento | Descrição |
| :--- | :--- |
| **Iluminação** | [Ex: Natural, vindo da direita] |
| **Ângulo** | [Ex: Contra-plongée] |
| **Composição** | [Ex: Regra dos terços, sujeito centralizado] |
| **Estilo** | [Ex: Fotografia realista, renderização 3D] |
| **Paleta de Cores** | [Ex: Tons quentes, alta saturação] |
| **Vibe/Atmosfera** | [Ex: Melancólica, Corporativa, Festiva] |
"""

# Prompt especializado para análise forense e investigação policial
PROMPT_FORENSE = """# SYSTEM ROLE
Atue como um Perito Criminal Sênior especializado em Análise Forense de Imagens e Inteligência Policial. Sua função é examinar evidências visuais com rigor técnico, imparcialidade e precisão jurídica, gerando laudos descritivos para inquéritos.

# CONTEXTO E OBJETIVO
Você receberá uma imagem de interesse investigativo (cena de crime, vigilância, evidência apreendida ou suspeitos).

**DADOS DE ENTRADA (OCR FORENSE):**
O sistema de OCR extraiu o seguinte texto bruto da imagem (placas, documentos, tatuagens, pichações):
---
{ocr_result}
---
Utilize este texto para corroborar a identificação de veículos, locais ou indivíduos. Se o OCR estiver vazio, verifique visualmente se há elementos textuais que o sistema perdeu.

# 🧱 PROTOCOLO DE ANÁLISE (STEP-BY-STEP)

<passo_1>
**Varredura de Segurança e Ambiente (Macro)**
- Identifique o local (interno/externo, residencial/comercial/ermo).
- Determine o horário provável (luz natural/artificial) e condições climáticas.
- Identifique pontos de entrada/saída e sinais de arrombamento ou luta corporal.
</passo_1>

<passo_2>
**Identificação de Sujeitos e Suspeitos**
- Descreva indivíduos com foco em identificação: estimativa de altura/peso, vestuário (marcas, cores), sinais particulares (tatuagens, cicatrizes, acessórios).
- Analise a linguagem corporal (agressiva, defensiva, em fuga).
- **Atenção:** Se houver rostos nítidos, descreva as características faciais, mas NÃO atribua nomes de pessoas reais (PII).
</passo_2>

<passo_3>
**Levantamento de Vestígios e Armas (Micro)**
- Busque por armas (fogo, brancas, improvisadas) e descreva modelo/tipo se possível.
- Identifique itens de interesse: drogas, dinheiro, eletrônicos, ferramentas de crime.
- Localize vestígios biológicos visíveis (manchas de substância avermelhada, fluidos) ou danos materiais (vidros quebrados, cápsulas deflagradas).
</passo_3>

<passo_4>
**Análise Textual e Veicular (Cruzamento com OCR)**
- Identifique veículos (Modelo, Cor, Placa). Compare a placa visual com o `{ocr_result}`.
- Transcreva documentos, telas de celular ou pichações visíveis.
</passo_4>

# 🚫 GUARDRAILS E ÉTICA FORENSE

1.  **Objetividade Absoluta:** Use linguagem denotativa. Em vez de "cena horrível", use "cena com presença de múltiplos ferimentos". Em vez de "sangue", prefira "manchas de substância de aspecto hemático" (padrão pericial).
2.  **Não Alucinar Intenção:** Descreva ações ("sujeito segurando objeto metálico"), não julgue intenções ("sujeito queria matar") a menos que a ação seja inequívoca.
3.  **Segurança de Conteúdo:** Se a imagem contiver violência extrema ou explícita que viole diretrizes de segurança, foque a descrição nos elementos periféricos e evidências materiais, evitando descrições gráficas de ferimentos.

# 📝 FORMATO DE SAÍDA (LAUDO TÉCNICO)

A resposta deve seguir estritamente este template em Markdown:

## 🚨 Relatório de Análise Pericial

### 1. Caracterização do Ambiente
* **Localização:** [Descrição técnica do espaço]
* **Condições:** [Iluminação, clima, visibilidade]
* **Sinais de Violência/Arrombamento:** [Sim/Não - Detalhes]

### 2. Envolvidos e Características
* **Indivíduo A:** [Gênero, Etnia aproximada, Vestimentas detalhadas, Sinais particulares]
* **Indivíduo B:** [Se houver]
* **Dinâmica:** [Posicionamento e interação entre os sujeitos]

### 3. Materialidade e Evidências (Tabela)
| Item/Vestígio | Localização na Imagem | Detalhes Visuais |
| :--- | :--- | :--- |
| [Ex: Arma de Fogo] | [Ex: Mão direita do Indivíduo A] | [Ex: Pistola preta, tipo polímero] |
| [Ex: Veículo] | [Ex: Fundo da imagem] | [Ex: Sedan Prata, amassado no para-choque] |
| [Ex: Subst. Hemática]| [Ex: Piso, próximo à porta] | [Ex: Poça de grande extensão] |

### 4. Análise Textual e OCR (Validação)
* **Placas Veiculares:** [Texto visualizado vs. OCR]
* **Documentos/Outros:** [Transcrição]

### 5. Conclusão Preliminar da Imagem
[Resumo objetivo do que está ocorrendo na cena, focado na materialidade do fato]
"""

# Dicionário de prompts disponíveis
PROMPT_ANALISE_PROFUNDA = """# SYSTEM ROLE
Você é um Analista Estruturalista e Semioticista Mestre com conhecimento formidável em materiais, linguagem corporal (cinésica), distâncias (proxêmica) e fotografia forense. Você pensa "Passo a Passo" (Chain of Thought) antes de gerar o laudo final.

# DADOS DETERMINÍSTICOS INJETADOS

**EXIF / GPS / TIMESTAMPS:**
{exif_data}

**TEXTO (OCR):**
{ocr_result}

**MÁQUINA DE VISÃO (YOLO11):**
{yolo_result}

**SINAIS DA IMAGEM:**
{quality_result}

# 🧱 PROTOCOLO DE ANÁLISE (CHAIN OF THOUGHT)

Sua resposta deve ser dividida em **duas partes**:
1. Uma seção `<thought>` onde você raciocina sobre a imagem e cruza seus achados visuais com os `DADOS DETERMINÍSTICOS`.
2. O seu `LAUDO COGNITIVO` final, estruturado e em Markdown puro.

## Parte 1: Raciocínio (Espaço de Pensamento)
Inicie sua resposta com:
<thought>
- O que o YOLO detectou? [Refletir sobre a contagem e posições lógicas].
- Qual o contexto semiótico da cena? (O que está implícito?).
- A luz e a qualidade da imagem afetam a leitura da imagem? Como?
- Observações de micro-texturas (tecidos, metal, vidro, madeira).
- Análise Proxêmica: Qual a distância entre os sujeitos e o que isso significa psicologicamente no contexto da foto?
</thought>

## Parte 2: Laudo Cognitivo

# 🧠 Laudo Cognitivo de Alta Fidelidade

## 1. Inventário Factual Dinâmico
- Liste todos os sujeitos, objetos (obrigatório bater com o YOLO) e seus estados físicos percebidos.
- Descreva texturas primárias.

## 2. Cinésica e Proxêmica (Se houver humanos/animais)
- **Cinésica:** Micro-expressões faciais, tensão nas mãos, posição dos pés, inclinação do torso.
- **Proxêmica:** Distância física entre entidades e como o ambiente molda ou restringe essa interação.

## 3. Dissecação do Ambiente
- Como a luz atinge as superfícies.
- Integração do texto OCR no espaço físico.

## 4. Síntese Interpretativa
- Qual a narrativa principal, tom prevalente e a vibração inerente à imagem condensada em um parágrafo.
"""

PROMPT_SCREENSHOTS = """# SYSTEM ROLE
Atue como um Especialista em Análise de Evidências Digitais, UX Forense e OCR de Interfaces. Sua função é examinar screenshots e capturas de tela com foco em conversas, páginas web, e-mails e telas de aplicativos, preservando a hierarquia visual e a ordem factual do conteúdo.

# CONTEXTO
Você receberá uma captura de tela ou imagem de interface que pode representar conversas (WhatsApp, Telegram, Instagram, Signal), páginas web, e-mails, comprovantes, painéis administrativos ou apps bancários.

**[1] METADADOS / EXIF (SE EXISTIREM):**
---
{exif_data}
---

**[2] TEXTO OCR PRÉ-EXTRAÍDO:**
---
{ocr_result}
---

**[3] DETECÇÕES DE OBJETOS / PISTAS VISUAIS:**
---
{yolo_result}
---

**[4] QUALIDADE / SINAIS TÉCNICOS DA IMAGEM:**
---
{quality_result}
---

Use esses dados apenas como âncoras factuais. Se houver resumo de ELA ou alertas técnicos, trate-os como indícios auxiliares, nunca como prova conclusiva de edição por si só.

# INSTRUÇÕES

1. **Identifique a plataforma ou o contexto da interface**
    - Determine o aplicativo, site ou tipo de sistema mais provável
    - Indique os elementos que sustentam essa conclusão: logotipo, cores, barra superior, layout, botões, URL, abas, menus, ícones, padrão de balões de conversa

2. **Preserve a hierarquia da tela**
    - Descreva a estrutura visual de cima para baixo: cabeçalho, barra de status, nome do contato/canal, corpo principal, rodapé/campo de entrada, menus, pop-ups, notificações
    - Se for conversa, mantenha a sequência cronológica visível e a distinção entre remetente e destinatário
    - Se for e-mail, diferencie remetente, destinatário, assunto, data, corpo, anexos e assinaturas
    - Se for página web, diferencie domínio, título, navegação, conteúdo principal, banners, formulários e avisos
    - Se for app bancário, destaque instituição, saldos, transações, favorecidos, chaves, comprovantes, botões e alertas de segurança

3. **Extraia e organize o conteúdo textual**
    - Valide o OCR com o que é visível
    - Corrija erros óbvios de OCR sem inventar conteúdo ausente
    - Estruture mensagens, valores, links, números, e-mails, telefones, datas, horários, IDs, usernames e botões relevantes

4. **Avalie sinais de edição, montagem ou inconsistência**
    - Procure desalinhamentos, cortes abruptos, sobreposição anormal, fontes incompatíveis, bolhas com estilos divergentes, horários incoerentes, elementos duplicados, ícones fora de contexto, áreas borradas seletivamente ou trechos visualmente recompostos
    - Diferencie claramente: [CERTO], [PROVÁVEL] e [INCERTO]
    - Se não houver indício confiável, diga explicitamente que não há evidência visual suficiente de edição

5. **Foque em utilidade pericial e documental**
    - Destaque campos sensíveis e artefatos importantes para investigação: usuários, handles, links, códigos, comprovantes, valores, chaves Pix, contas, boletos, anexos, nomes de grupos, status da mensagem, indicador de encaminhamento e carimbo temporal

# REGRAS CRÍTICAS

- Não invente mensagens cortadas, ocultas ou ilegíveis
- Não assuma identidade real de perfis; descreva apenas os identificadores visíveis
- Se a imagem for uma foto de uma tela e não um screenshot nativo, registre isso
- Se houver ambiguidade entre app/site semelhantes, apresente hipótese principal e alternativas com nível de confiança

# FORMATO DE SAÍDA

## 🖥️ Resumo Executivo
[Resumo objetivo do que a tela mostra e da finalidade provável]

## 🧭 Identificação da Plataforma
* **Tipo de Evidência:** [conversa / e-mail / página web / app bancário / outro]
* **Plataforma Provável:** [WhatsApp / Telegram / Instagram / navegador / banco / outro]
* **Base da Identificação:** [elementos visuais que sustentam a conclusão]

## 🧱 Estrutura da Tela
| Região | Conteúdo observado |
|--------|--------------------|
| Cabeçalho | [Descrição] |
| Área principal | [Descrição] |
| Rodapé / ações | [Descrição] |

## 💬 Conteúdo Principal
### Participantes / Perfis
* **Remetente/Origem:** [Nome, handle ou rótulo visível]
* **Destinatário/Alvo:** [Nome, handle ou rótulo visível]
* **Contexto:** [grupo, chat privado, caixa de e-mail, página institucional, app financeiro etc.]

### Linha do Tempo / Itens Visíveis
| Ordem | Autor/Origem | Horário/Data | Conteúdo / Ação |
|-------|--------------|--------------|-----------------|
| 1 | [Autor] | [Horário] | [Mensagem, evento, botão, valor, aviso] |

## 🔎 Campos Sensíveis e Artefatos Relevantes
* **Links / URLs:** [Se houver]
* **Telefones / E-mails / IDs:** [Se houver]
* **Valores / Transações / Chaves:** [Se houver]
* **Anexos / Mídias / Indicadores:** [Se houver]

## 🧪 Sinais de Edição ou Montagem
* **Achados:** [Descrição objetiva]
* **Nível de Confiança:** [CERTO / PROVÁVEL / INCERTO]
* **Conclusão Técnica:** [Há ou não há indícios visuais suficientes]

## ⚠️ Limitações
[Trechos cortados, blur, baixa resolução, OCR parcial, sobreposição de interface ou qualquer fator que limite a leitura]
"""

ANALYSIS_PROMPTS = {
    "geral": {
        "name": "📷 Análise Geral",
        "description": "Análise descritiva para acessibilidade e documentação",
        "prompt": PROMPT_GERAL
    },
    "forense": {
        "name": "🔍 Análise Forense",
        "description": "Laudo pericial para investigação policial",
        "prompt": PROMPT_FORENSE
    },
    "analise_profunda": {
        "name": "🧠 Análise Profunda",
        "description": "Semiótica, materiais, proxêmica e micro-detalhes (Chain of Thought)",
        "prompt": PROMPT_ANALISE_PROFUNDA
    },
    "screenshots": {
        "name": "🖥️ Análise de Screenshots/Telas",
        "description": "Conversas, páginas web, e-mails e interfaces com foco em hierarquia e sinais de edição",
        "prompt": PROMPT_SCREENSHOTS
    }
}

# Prompt padrão (para compatibilidade)
SYSTEM_PROMPT_TEMPLATE = PROMPT_GERAL


def get_available_analysis_modes() -> dict:
    """Retorna modos disponíveis preferindo prompt_templates quando presente."""
    if PROMPTS_AVAILABLE:
        try:
            mgr = get_prompt_manager()
            return {
                key: {
                    "name": prompt.name,
                    "description": prompt.description,
                    "prompt": prompt.prompt,
                }
                for key, prompt in mgr.get_all_prompts().items()
            }
        except Exception as exc:
            logger.warning(f"⚠️ Falha ao carregar prompt_templates: {exc}")

    return ANALYSIS_PROMPTS


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class ImageData:
    """Dados de uma imagem processada."""
    path: Path
    name: str
    extension: str
    size_bytes: int
    dimensions: tuple[int, int] = (0, 0)
    hash_md5: str = ""
    hash_sha256: str = ""
    base64_data: str = ""
    jpeg_bytes: bytes = b""
    
    def __post_init__(self):
        self.extension = self.extension.lower()


@dataclass 
class AnalysisResult:
    """Resultado de uma análise de IA."""
    model_name: str
    success: bool
    content: str = ""
    error: str = ""
    processing_time: float = 0.0
    tokens_used: int = 0


@dataclass
class ImageAnalysisReport:
    """Relatório completo de análise de uma imagem."""
    image: ImageData
    ocr_result: str = ""
    yolo_result: str = ""
    quality_result: str = ""
    exif_result: str = ""
    analyses: list[AnalysisResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# PROCESSAMENTO DE IMAGENS
# ============================================================================

class ImageProcessor:
    """Processador de imagens com suporte a múltiplos formatos."""
    
    @staticmethod
    def find_images(input_dir: Path) -> Generator[Path, None, None]:
        """Busca recursivamente imagens no diretório."""
        if not input_dir.exists():
            logger.error(f"❌ Diretório não encontrado: {input_dir}")
            return
        
        for file_path in input_dir.rglob("*"):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield file_path
    
    @staticmethod
    def load_image(path: Path) -> Optional[Image.Image]:
        """Carrega imagem de qualquer formato suportado."""
        ext = path.suffix.lower()
        
        try:
            # Arquivos RAW
            if ext in RAW_EXTENSIONS:
                if not RAW_SUPPORT:
                    logger.warning(f"⚠️ Suporte a RAW não disponível para: {path.name}")
                    return None
                with rawpy.imread(str(path)) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        half_size=False,
                        no_auto_bright=False,
                        output_bps=8
                    )
                return Image.fromarray(rgb)
            
            # Arquivos HEIC/HEIF
            elif ext in HEIF_EXTENSIONS:
                if not HEIF_SUPPORT:
                    logger.warning(f"⚠️ Suporte a HEIC não disponível para: {path.name}")
                    return None
                # pillow-heif já está registrado, usar PIL diretamente
                return Image.open(path)
            
            # Outros formatos (PIL nativo)
            else:
                return Image.open(path)
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar {path.name}: {e}")
            return None
    
    @staticmethod
    def prepare_for_api(image: Image.Image) -> tuple[str, bytes]:
        """
        Prepara imagem para APIs (converte para JPEG RGB, redimensiona, retorna base64).
        Returns: (base64_string, jpeg_bytes)
        """
        # Converter para RGB se necessário
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionar se muito grande
        if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
            image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Converter para JPEG em memória
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
        jpeg_bytes = buffer.getvalue()
        
        # Codificar em base64
        base64_data = base64.b64encode(jpeg_bytes).decode('utf-8')
        
        return base64_data, jpeg_bytes
    
    @staticmethod
    def calculate_md5(data: bytes) -> str:
        """Calcula hash MD5 dos bytes."""
        return hashlib.md5(data).hexdigest()
    
    @staticmethod
    def calculate_sha256(data: bytes) -> str:
        """Calcula hash SHA-256 dos bytes."""
        return hashlib.sha256(data).hexdigest()
    
    @classmethod
    def process_image(cls, path: Path) -> Optional[ImageData]:
        """Processa uma imagem e retorna dados preparados para análise."""
        logger.info(f"📷 Processando: {path.name}")
        
        # Carregar imagem
        image = cls.load_image(path)
        if image is None:
            return None
        
        # Preparar para API
        base64_data, jpeg_bytes = cls.prepare_for_api(image)
        
        # Criar objeto de dados
        image_data = ImageData(
            path=path,
            name=path.stem,
            extension=path.suffix,
            size_bytes=path.stat().st_size,
            dimensions=image.size,
            hash_md5=cls.calculate_md5(jpeg_bytes),
            hash_sha256=cls.calculate_sha256(jpeg_bytes),
            base64_data=base64_data,
            jpeg_bytes=jpeg_bytes
        )
        
        logger.info(f"   ✅ {image_data.dimensions[0]}x{image_data.dimensions[1]} | {len(jpeg_bytes)/1024:.1f}KB")
        return image_data


# ============================================================================
# CLIENTES DE IA
# ============================================================================

class OpenAIClient:
    """Cliente para API OpenAI."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if not self.api_key:
            logger.warning("⚠️ OPENAI_API_KEY não encontrada no .env")
        elif OPENAI_AVAILABLE:
            self.client = openai.OpenAI(api_key=self.api_key)
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def analyze_image(self, image_data: ImageData, system_prompt: str) -> AnalysisResult:
        """Analisa imagem usando GPT-5.4-mini."""
        if not self.is_available():
            return AnalysisResult(
                model_name=OPENAI_MODEL,
                success=False,
                error="Cliente OpenAI não disponível"
            )
        
        start_time = datetime.now()
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analise esta imagem seguindo as instruções do sistema."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data.base64_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.1
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            raw_content = response.choices[0].message.content or ""
            cleaned_content = re.sub(r"<think>[\s\S]*?</think>", "", raw_content).strip()

            if not cleaned_content:
                return AnalysisResult(
                    model_name=OPENAI_MODEL,
                    success=False,
                    error="Modelo retornou resposta vazia",
                    processing_time=elapsed,
                    tokens_used=response.usage.total_tokens if response.usage else 0
                )

            return AnalysisResult(
                model_name=OPENAI_MODEL,
                success=True,
                content=cleaned_content,
                processing_time=elapsed,
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
            
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ OpenAI erro: {e}")
            return AnalysisResult(
                model_name=OPENAI_MODEL,
                success=False,
                error=str(e),
                processing_time=elapsed
            )


# ============================================================================
# FUNÇÕES DE OCR
# ============================================================================


class OllamaClient:
    """Cliente para Ollama (modelos locais)."""
    
    def __init__(self):
        self.available = OLLAMA_AVAILABLE
        self._check_connection()
    
    def _check_connection(self):
        """Verifica se o servidor Ollama está rodando."""
        if not self.available:
            return
        
        try:
            ollama.list()
            logger.info("✅ Ollama conectado")
        except Exception as e:
            logger.warning(f"⚠️ Ollama não acessível: {e}")

    def unload_model(self, model: str) -> None:
        """Descarrega um modelo da VRAM enviando keep_alive=0."""
        if not self.available:
            return
        try:
            ollama.chat(model=model, messages=[], keep_alive=0)
            logger.info(f"♻️ Modelo {model} descarregado da VRAM")
        except Exception as exc:
            logger.debug(f"Falha ao descarregar modelo {model}: {exc}")

    def unload_models(self, models: list[str]) -> None:
        """Descarrega uma lista de modelos da VRAM."""
        for model in models:
            self.unload_model(model)
            self.available = False
    
    def is_available(self) -> bool:
        return self.available
    
    def extract_ocr_glm(self, image_data: ImageData) -> str:
        """Extrai texto da imagem usando GLM OCR (glm-ocr:bf16)."""
        if not self.is_available():
            return "[GLM OCR não disponível - Ollama offline]"
        
        logger.info(f"   🔤 Executando GLM OCR...")
        
        try:
            response = ollama.chat(
                model=OCR_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": "Extract all visible text from this image. Return only the raw text, nothing else.",
                        "images": [image_data.base64_data]
                    }
                ],
                options={
                    "temperature": 0.1,
                    "num_predict": 1024,
                    "num_ctx": 4096
                }
            )
            
            ocr_text = response['message']['content'].strip()
            logger.info(f"   ✅ GLM OCR extraiu {len(ocr_text)} caracteres")
            return ocr_text if ocr_text else "[Nenhum texto detectado]"
            
        except Exception as e:
            logger.error(f"   ❌ GLM OCR erro: {e}")
            return f"[Erro GLM OCR: {e}]"
    
    def extract_ocr_olmocr2(self, image_data: ImageData) -> str:
        """Extrai texto da imagem usando OLMoOCR2 (richardyoung/olmocr2:7b-q8)."""
        if not self.is_available():
            return "[OLMoOCR2 não disponível - Ollama offline]"
        
        logger.info(f"   📖 Executando OLMoOCR2...")
        
        try:
            response = ollama.chat(
                model=OCR_MODEL_ALT,
                messages=[
                    {
                        "role": "user",
                        "content": "Extract all visible text from this image. Return only the raw text, nothing else.",
                        "images": [image_data.base64_data]
                    }
                ],
                options={
                    "temperature": 0.1,
                    "num_predict": 1024,
                    "num_ctx": 4096
                }
            )
            
            ocr_text = response['message']['content'].strip()
            logger.info(f"   ✅ OLMoOCR2 extraiu {len(ocr_text)} caracteres")
            return ocr_text if ocr_text else "[Nenhum texto detectado]"
            
        except Exception as e:
            logger.error(f"   ❌ OLMoOCR2 erro: {e}")
            return f"[Erro OLMoOCR2: {e}]"
    
    def extract_ocr(self, image_data: ImageData, engine: str = "glm-ocr") -> str:
        """
        Extrai texto da imagem usando o engine especificado.
        
        Args:
            image_data: Dados da imagem
            engine: "glm-ocr", "olmocr2" ou "none"
        """
        if engine == "none":
            return "[OCR desabilitado]"
        
        if engine == "olmocr2":
            return self.extract_ocr_olmocr2(image_data)
        
        # Default: glm-ocr
        return self.extract_ocr_glm(image_data)
    
    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        """Remove blocos <think>...</think> gerados por modelos com raciocínio interno."""
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    def analyze_image(self, image_data: ImageData, system_prompt: str, model: str) -> AnalysisResult:
        """Analisa imagem usando modelo Ollama de visão."""
        if not self.is_available():
            return AnalysisResult(
                model_name=model,
                success=False,
                error="Ollama não disponível"
            )
        
        start_time = datetime.now()
        
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": "Analise esta imagem seguindo as instruções do sistema.",
                        "images": [image_data.base64_data]
                    }
                ],
                options={
                    "temperature": 0.3,
                    "num_predict": 8192,
                    "num_ctx": 8192
                }
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()

            raw_content = response['message']['content'] or ""
            cleaned_content = self._strip_think_blocks(raw_content)

            if not cleaned_content:
                return AnalysisResult(
                    model_name=model,
                    success=False,
                    error="Modelo retornou resposta vazia (possível truncamento do bloco de raciocínio)",
                    processing_time=elapsed
                )

            return AnalysisResult(
                model_name=model,
                success=True,
                content=cleaned_content,
                processing_time=elapsed
            )
            
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Ollama [{model}] erro: {e}")
            return AnalysisResult(
                model_name=model,
                success=False,
                error=str(e),
                processing_time=elapsed
            )


# ============================================================================
# GERADOR DE RELATÓRIOS
# ============================================================================

class ReportGenerator:
    """Gera relatórios Markdown para cada análise."""
    
    @staticmethod
    def sanitize_filename(name: str) -> str:
        """Remove caracteres inválidos do nome do arquivo."""
        return shared_sanitize_filename(name)
    
    @staticmethod
    def get_model_short_name(model: str) -> str:
        """Retorna nome curto do modelo para o arquivo."""
        return shared_get_model_short_name(model)
    
    @classmethod
    def save_report(cls, report: ImageAnalysisReport, output_dir: Path) -> list[Path]:
        """Salva relatórios individuais para cada modelo."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        
        for analysis in report.analyses:
            if not analysis.success:
                continue
            
            # Nome do arquivo: nomeimagem_nomemodelo.md
            model_short = cls.get_model_short_name(analysis.model_name)
            image_name = cls.sanitize_filename(report.image.name)
            filename = f"{image_name}_{model_short}.md"
            filepath = output_dir / filename
            
            # Conteúdo do relatório
            content = cls._build_report_content(report, analysis)
            
            try:
                filepath.write_text(content, encoding='utf-8')
                saved_files.append(filepath)
                logger.info(f"   💾 Salvo: {filename}")
            except Exception as e:
                logger.error(f"   ❌ Erro ao salvar {filename}: {e}")
        
        return saved_files
    
    @classmethod
    def _build_report_content(cls, report: ImageAnalysisReport, analysis: AnalysisResult) -> str:
        """Constrói o conteúdo Markdown do relatório."""
        # Gerar seção de pós-processamento
        post_section = ""
        if POST_PROCESSOR_AVAILABLE:
            try:
                pp = PostProcessor()
                pp_result = pp.process(
                    ocr_text=report.ocr_result,
                    llm_analysis=analysis.content,
                    exif_data=report.exif_result,
                    yolo_result=report.yolo_result,
                    quality_result=report.quality_result
                )
                post_section = pp.format_report_section(pp_result)
            except Exception as e:
                logger.debug(f"Pós-processamento falhou: {e}")
        
        header = f"""---
# 📊 Relatório de Análise de Imagem
---

**Arquivo:** `{report.image.path.name}`  
**Modelo:** `{analysis.model_name}`  
**Data:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Tempo de Processamento:** {analysis.processing_time:.2f}s  

---

## 📋 Metadados da Imagem

| Propriedade | Valor |
|:------------|:------|
| **Dimensões** | {report.image.dimensions[0]} x {report.image.dimensions[1]} px |
| **Formato Original** | {report.image.extension.upper()} |
| **Tamanho** | {report.image.size_bytes / 1024:.1f} KB |
| **Hash MD5** | `{report.image.hash_md5}` |
| **Hash SHA-256** | `{report.image.hash_sha256}` |

---

## 🔍 Dados Extraídos (Pré-Análise)

### 📸 EXIF & GPS (Metadados Ocultos)
```text
{report.exif_result}
```

### 📝 OCR (Texto Visível)
```text
{report.ocr_result}
```

### 🎯 YOLO11 (Objetos)
```text
{report.yolo_result}
```

### ✨ Qualidade da Imagem
```text
{report.quality_result}
```

---

## 🤖 Análise do Modelo

{analysis.content}

---

{post_section}

*Relatório gerado automaticamente por Batch Image Analyzer v2.0*
"""
        return header


# ============================================================================
# ANALISADOR PRINCIPAL
# ============================================================================

class BatchImageAnalyzer:
    """Orquestrador principal do pipeline de análise."""
    
    def __init__(
        self, 
        input_dir: Path = INPUT_DIR, 
        output_dir: Path = OUTPUT_DIR,
        analysis_mode: str = "geral",
        ocr_engine: str = "glm-ocr",
        workers: int = 1,
        models: list = None,
        resume: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.analysis_mode = analysis_mode
        self.ocr_engine = ocr_engine
        self.workers = max(1, min(workers, 8))  # Entre 1 e 8 workers
        self.explicit_models = models  # Modelos especificados pelo usuário
        self.resume_checkpoint = resume
        self.processor = ImageProcessor()
        self.openai_client = OpenAIClient()
        self.ollama_client = OllamaClient()
        self.report_generator = ReportGenerator()
        self.post_processor = PostProcessor() if POST_PROCESSOR_AVAILABLE else None
        
        # Validar modo de análise
        available_modes = get_available_analysis_modes()
        if analysis_mode not in available_modes:
            logger.warning(f"⚠️ Modo '{analysis_mode}' inválido. Usando 'geral'.")
            self.analysis_mode = "geral"
        
        # Validar engine de OCR
        if ocr_engine not in OCR_ENGINES:
            logger.warning(f"⚠️ OCR engine '{ocr_engine}' inválido. Usando 'glm-ocr'.")
            self.ocr_engine = "glm-ocr"
        
        # Estatísticas (thread-safe com lock)
        import threading
        self._stats_lock = threading.Lock()
        self.stats = {
            "total_images": 0,
            "pending_images": 0,
            "processed": 0,
            "failed": 0,
            "reports_generated": 0,
            "resumed_skipped": 0,
            "analysis_mode": self.analysis_mode,
            "ocr_engine": self.ocr_engine,
            "workers": self.workers
        }

    def _create_shared_pipeline(self):
        """Cria uma instância do pipeline compartilhado quando disponível."""
        if not SHARED_PIPELINE_AVAILABLE:
            return None

        return AnalysisPipeline(
            analysis_mode=self.analysis_mode,
            ocr_engine=self.ocr_engine,
            yolo_model="yolo11s",
        )
    
    def _get_available_models(self) -> list[tuple[str, str]]:
        """Retorna lista de modelos disponíveis: [(nome, tipo)]"""
        # Se o usuário especificou modelos, usar somente esses
        if self.explicit_models:
            return self.explicit_models
        
        # Padrão: somente GPT-5.4-mini (se disponível)
        models = []
        
        if self.openai_client.is_available():
            models.append((OPENAI_MODEL, "openai"))
        
        return models

    def _create_checkpoint_manager(
        self,
        selected_models: list[tuple[str, str]],
        export_formats: list[str],
    ) -> BatchCheckpointManager:
        job_config = build_batch_job_config(
            selected_models=selected_models,
            analysis_mode=self.analysis_mode,
            ocr_engine=self.ocr_engine,
            export_formats=export_formats,
            yolo_model="yolo11s",
        )
        return BatchCheckpointManager(
            checkpoint_path=get_default_checkpoint_path(self.output_dir),
            job_signature=build_batch_signature(job_config),
            job_config=job_config,
        )

    def _record_checkpoint_result(
        self,
        checkpoint_manager: Optional[BatchCheckpointManager],
        image_path: Path,
        task_result: dict,
    ) -> None:
        if checkpoint_manager is None:
            return

        checkpoint_manager.record_result(
            image_path=image_path,
            success_count=task_result.get("success", 0),
            failed_count=task_result.get("failed", 0),
            reports=task_result.get("reports", []),
            errors=task_result.get("errors", []),
        )

    def _log_final_summary(self) -> None:
        logger.info("\n" + "=" * 60)
        logger.info("📊 RESUMO FINAL")
        logger.info("=" * 60)
        logger.info(f"   Total de imagens solicitadas: {self.stats['total_images']}")
        logger.info(f"   Imagens pendentes nesta execução: {self.stats['pending_images']}")
        logger.info(f"   Processadas com sucesso: {self.stats['processed']}")
        logger.info(f"   Falhas: {self.stats['failed']}")
        logger.info(f"   Relatórios gerados: {self.stats['reports_generated']}")
        if self.stats["resumed_skipped"] > 0:
            logger.info(f"   Puladas por checkpoint: {self.stats['resumed_skipped']}")
        logger.info(f"   Diretório de saída: {self.output_dir.absolute()}")
        logger.info("=" * 60)
    
    def _analyze_with_model(
        self, 
        image_data: ImageData, 
        system_prompt: str,
        model: str, 
        model_type: str
    ) -> AnalysisResult:
        """Executa análise com um modelo específico."""
        logger.info(f"   🤖 Analisando com {model}...")
        
        if model_type == "openai":
            return self.openai_client.analyze_image(image_data, system_prompt)
        else:
            return self.ollama_client.analyze_image(image_data, system_prompt, model)
    
    def analyze_image(self, image_path: Path) -> Optional[ImageAnalysisReport]:
        """Pipeline completo de análise para uma imagem."""
        # 1. Processar imagem e extrair qualidades visuais
        image_data = self.processor.process_image(image_path)
        if image_data is None:
            return None
            
        pil_image = Image.open(BytesIO(image_data.jpeg_bytes))
        
        # 1.a) Preprocessor (Qualidade/Blur/Brightness e EXIF)
        quality_str = "[Qualidade não avaliada]"
        exif_str = "[EXIF não avaliado]"
        if PREPROCESSOR_AVAILABLE:
            quality_info = quick_analyze(pil_image)
            if quality_info:
                blur_txt = "Borrada" if quality_info.get("is_blurry") else "Nítida"
                brt = quality_info.get("brightness", 0)
                brt_txt = "Escura" if brt < 0.3 else ("Clara" if brt > 0.7 else "Equilibrada")
                recs = ", ".join(quality_info.get("recommendations", []))
                quality_str = f"Nitidez: {blur_txt}\nIluminação: {brt_txt}\nAvisos: {recs if recs else 'Nenhum'}"
                
                exif_dict = quality_info.get("exif_data", {})
                if exif_dict:
                    exif_lines = [f"- {k}: {v}" for k, v in exif_dict.items()]
                    exif_str = "\n".join(exif_lines)
                else:
                    exif_str = "Metadados de câmera ou GPS ausentes."
        
        # 1.b) YOLO (Detecção de Objetos Físicos)
        yolo_str = "[YOLO não executado]"
        if YOLO_AVAILABLE and self.analysis_mode not in {"documentos", "screenshots"}:
            detector = get_detector("yolo11s")  # fast by default for batch
            if detector:
                try:
                    yolo_res = detector.detect(pil_image)
                    if yolo_res.total_objects == 0:
                        yolo_str = "Nenhum objeto primário detectado pelo YOLO."
                    else:
                        summary = yolo_res.get_summary()
                        yolo_str = "Objetos detectados pela máquina:\n"
                        for obj_class, count in summary.items():
                            yolo_str += f"- {count}x {obj_class}\n"
                except Exception as e:
                    logger.warning(f"⚠️ Falha no YOLO: {e}")
        
        # 1.c) ELA (Error Level Analysis — detecção de manipulação)
        ela_str = ""
        if self.analysis_mode in {"forense", "screenshots"}:
            try:
                from ela_analyzer import ELAAnalyzer
                ela = ELAAnalyzer(quality=95, scale=15)
                ela_result = ela.analyze(pil_image)
                ela_str = ela_result.get_summary()
                logger.info(f"   🔬 ELA: {ela_result.verdict}")
            except ImportError:
                logger.debug("ELA não disponível — módulo ela_analyzer não encontrado")
            except Exception as e:
                logger.warning(f"⚠️ Falha no ELA: {e}")
        
        # 2. Extrair OCR
        ocr_result = self.ollama_client.extract_ocr(image_data, engine=self.ocr_engine)
        
        # 3. Construir prompt via prompt_templates (resolve_prompt ou direto)
        template = None
        if PROMPTS_AVAILABLE:
            mgr = get_prompt_manager()
            template = mgr.get_prompt(self.analysis_mode)
        
        # Enriquecer quality_str com ELA quando disponível
        if ela_str:
            quality_str += f"\n\n--- ANÁLISE ELA (Error Level Analysis) ---\n{ela_str}"
        
        if template:
            system_prompt = template.format_prompt(
                ocr_result=ocr_result,
                yolo_result=yolo_str,
                quality_result=quality_str,
                exif_data=exif_str
            )
        else:
            # Fallback
            system_prompt = ANALYSIS_PROMPTS.get(self.analysis_mode, ANALYSIS_PROMPTS["geral"])["prompt"].format(
                ocr_result=ocr_result,
                yolo_result=yolo_str,
                quality_result=quality_str,
                exif_data=exif_str
            )
        
        # 4. Criar relatório
        report = ImageAnalysisReport(
            image=image_data,
            ocr_result=ocr_result,
            yolo_result=yolo_str,
            quality_result=quality_str,
            exif_result=exif_str
        )
        
        # 5. Analisar com cada modelo disponível
        available_models = self._get_available_models()
        
        if not available_models:
            logger.error("❌ Nenhum modelo de IA disponível!")
            return None
        
        for model, model_type in available_models:
            try:
                result = self._analyze_with_model(
                    image_data, system_prompt, model, model_type
                )
                report.analyses.append(result)
                
                if result.success:
                    logger.info(f"   ✅ {model}: {result.processing_time:.1f}s")
                else:
                    logger.warning(f"   ⚠️ {model}: {result.error}")
                    
            except Exception as e:
                logger.error(f"   ❌ Erro inesperado em {model}: {e}")
                report.analyses.append(AnalysisResult(
                    model_name=model,
                    success=False,
                    error=str(e)
                ))
        
        return report
    
    def _process_single_image(
        self,
        image_path: Path,
        index: int,
        total: int,
        selected_models: list[tuple[str, str]],
    ) -> dict:
        """
        Processa uma única imagem (para execução paralela).
        """
        logger.info(f"[{index}/{total}] 📷 Processando: {image_path.name}")
        failed_count = max(len(selected_models), 1)

        shared_pipeline = self._create_shared_pipeline()
        if shared_pipeline is not None:
            try:
                task_result = shared_pipeline.process_image(
                    image_path=image_path,
                    selected_models=selected_models,
                    output_dir=self.output_dir,
                    export_formats=["md"],
                    use_cache=False,
                )

                return {
                    "image_path": image_path,
                    "image": image_path.name,
                    "reports": task_result["reports"],
                    "errors": task_result["errors"],
                    "success": task_result["success"],
                    "failed": task_result["failed"],
                }
            except Exception as e:
                logger.error(f"❌ Erro processando {image_path.name}: {e}")
                return {
                    "image_path": image_path,
                    "image": image_path.name,
                    "reports": [],
                    "errors": [str(e)],
                    "success": 0,
                    "failed": failed_count,
                }
        
        try:
            report = self.analyze_image(image_path)
            
            if report:
                saved = self.report_generator.save_report(report, self.output_dir)
                return {
                    "image_path": image_path,
                    "image": image_path.name,
                    "reports": saved,
                    "errors": [],
                    "success": len(saved),
                    "failed": 0,
                }

            return {
                "image_path": image_path,
                "image": image_path.name,
                "reports": [],
                "errors": ["Falha ao processar imagem"],
                "success": 0,
                "failed": failed_count,
            }
                
        except Exception as e:
            logger.error(f"❌ Erro processando {image_path.name}: {e}")
            return {
                "image_path": image_path,
                "image": image_path.name,
                "reports": [],
                "errors": [str(e)],
                "success": 0,
                "failed": failed_count,
            }
    
    def run(self) -> dict:
        """Executa o pipeline completo de análise em lote."""
        logger.info("=" * 60)
        logger.info("🖼️  BATCH IMAGE ANALYZER v1.0")
        logger.info("=" * 60)
        
        # Mostrar modo de análise
        mode_info = ANALYSIS_PROMPTS[self.analysis_mode]
        logger.info(f"📋 Modo de Análise: {mode_info['name']}")
        logger.info(f"   {mode_info['description']}")
        
        # Mostrar OCR engine
        ocr_names = {"glm-ocr": "🔤 GLM OCR (glm-ocr:bf16)", "olmocr2": "📖 OLMoOCR2 (7b-q8)", "none": "❌ Desabilitado"}
        logger.info(f"🔍 OCR Engine: {ocr_names.get(self.ocr_engine, self.ocr_engine)}")
        
        # Mostrar workers
        logger.info(f"⚡ Workers paralelos: {self.workers}")
        
        # Verificar diretórios
        if not self.input_dir.exists():
            logger.info(f"📁 Criando diretório de entrada: {self.input_dir}")
            self.input_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Encontrar imagens
        images = list(self.processor.find_images(self.input_dir))
        self.stats["total_images"] = len(images)
        self.stats["pending_images"] = len(images)
        
        if not images:
            logger.warning(f"⚠️ Nenhuma imagem encontrada em: {self.input_dir}")
            logger.info("   Extensões suportadas: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
            return self.stats
        
        logger.info(f"📷 Encontradas {len(images)} imagens")
        logger.info("-" * 60)
        
        # Verificar modelos disponíveis
        available = self._get_available_models()
        logger.info(f"🤖 Modelos disponíveis: {len(available)}")
        for model, mtype in available:
            logger.info(f"   • {model} ({mtype})")

        checkpoint_manager = self._create_checkpoint_manager(available, ["md"])
        resume_state = checkpoint_manager.prepare_run(images, resume=self.resume_checkpoint)
        images = resume_state.pending_images
        self.stats["pending_images"] = len(images)
        self.stats["resumed_skipped"] = len(resume_state.skipped_entries)

        if self.resume_checkpoint:
            logger.info(f"♻️ Checkpoint: {checkpoint_manager.checkpoint_path}")
        if resume_state.reset_reason:
            logger.info(f"♻️ {resume_state.reset_reason}")
        if self.stats["resumed_skipped"] > 0:
            logger.info(f"♻️ {self.stats['resumed_skipped']} imagens já concluídas foram puladas")
        logger.info("-" * 60)

        if not images:
            logger.info("✅ Nenhuma imagem pendente para processar. O lote já está concluído para esta configuração.")
            self._log_final_summary()
            return self.stats
        
        # Processar imagens (sequencial ou paralelo)
        total = len(images)
        shared_pipeline = self._create_shared_pipeline()
        
        if self.workers == 1:
            # Processamento sequencial (original)
            logger.info("📝 Modo: Sequencial")
            logger.info("-" * 60)
            
            for i, image_path in enumerate(images, 1):
                logger.info(f"\n[{i}/{total}] {image_path.name}")

                if shared_pipeline is not None:
                    try:
                        task_result = shared_pipeline.process_image(
                            image_path=image_path,
                            selected_models=available,
                            output_dir=self.output_dir,
                            export_formats=["md"],
                            use_cache=False,
                        )
                        self._record_checkpoint_result(checkpoint_manager, image_path, task_result)

                        if task_result["success"] > 0:
                            with self._stats_lock:
                                self.stats["processed"] += 1
                                self.stats["reports_generated"] += len(task_result["reports"])
                            logger.info(f"   ✅ {image_path.name}: {task_result['success']} relatórios")
                        else:
                            with self._stats_lock:
                                self.stats["failed"] += 1
                        for error in task_result["errors"]:
                            logger.error(f"   ❌ {image_path.name}: {error}")
                    except Exception as e:
                        logger.error(f"❌ Erro fatal processando {image_path.name}: {e}")
                        self._record_checkpoint_result(
                            checkpoint_manager,
                            image_path,
                            {
                                "success": 0,
                                "failed": max(len(available), 1),
                                "reports": [],
                                "errors": [str(e)],
                            },
                        )
                        with self._stats_lock:
                            self.stats["failed"] += 1
                    continue

                try:
                    report = self.analyze_image(image_path)

                    if report:
                        saved = self.report_generator.save_report(report, self.output_dir)
                        self._record_checkpoint_result(
                            checkpoint_manager,
                            image_path,
                            {
                                "success": len(saved),
                                "failed": 0,
                                "reports": saved,
                                "errors": [],
                            },
                        )
                        with self._stats_lock:
                            self.stats["processed"] += 1
                            self.stats["reports_generated"] += len(saved)
                    else:
                        self._record_checkpoint_result(
                            checkpoint_manager,
                            image_path,
                            {
                                "success": 0,
                                "failed": max(len(available), 1),
                                "reports": [],
                                "errors": ["Falha ao processar imagem"],
                            },
                        )
                        with self._stats_lock:
                            self.stats["failed"] += 1

                except Exception as e:
                    logger.error(f"❌ Erro fatal processando {image_path.name}: {e}")
                    self._record_checkpoint_result(
                        checkpoint_manager,
                        image_path,
                        {
                            "success": 0,
                            "failed": max(len(available), 1),
                            "reports": [],
                            "errors": [str(e)],
                        },
                    )
                    with self._stats_lock:
                        self.stats["failed"] += 1
                    continue
        else:
            # Processamento paralelo com ThreadPoolExecutor
            logger.info(f"🚀 Modo: Paralelo ({self.workers} workers)")
            logger.info("-" * 60)
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Submeter todas as tarefas
                futures = {
                    executor.submit(self._process_single_image, img_path, i, total, available): img_path 
                    for i, img_path in enumerate(images, 1)
                }
                
                # Processar resultados conforme completam
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    task_result = future.result()
                    image_path = task_result["image_path"]
                    self._record_checkpoint_result(checkpoint_manager, image_path, task_result)

                    if task_result["success"] > 0:
                        with self._stats_lock:
                            self.stats["processed"] += 1
                            self.stats["reports_generated"] += len(task_result["reports"])
                        logger.info(f"   ✅ {image_path.name}: {len(task_result['reports'])} relatórios")
                    else:
                        with self._stats_lock:
                            self.stats["failed"] += 1
                    for error in task_result["errors"]:
                        logger.error(f"   ❌ {image_path.name}: {error}")
                    
                    # Progresso
                    progress = (completed / total) * 100
                    logger.info(f"   📊 Progresso: {completed}/{total} ({progress:.1f}%)")

        self._log_final_summary()

        # Descarregar modelos Ollama da VRAM
        ollama_models_to_unload = []
        for model, mtype in available:
            if mtype == "ollama" and model not in ollama_models_to_unload:
                ollama_models_to_unload.append(model)
        if self.ocr_engine != "none":
            ocr_model = OCR_MODEL if self.ocr_engine == "glm-ocr" else OCR_MODEL_ALT
            if ocr_model not in ollama_models_to_unload:
                ollama_models_to_unload.append(ocr_model)
        if ollama_models_to_unload:
            self.ollama_client.unload_models(ollama_models_to_unload)
        
        return self.stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Função principal."""
    import argparse
    available_modes = get_available_analysis_modes()
    
    # Construir descrição dos modos disponíveis
    modes_help = "Modos disponíveis:\n"
    for key, info in available_modes.items():
        modes_help += f"  {key}: {info['name']} - {info['description']}\n"
    
    parser = argparse.ArgumentParser(
        description="Batch Image Analyzer - Análise de imagens com múltiplos modelos de IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{modes_help}
OCR Engines disponíveis:
  glm-ocr:  GLM OCR via Ollama (glm-ocr:bf16 — padrão)
  olmocr2:  OLMoOCR2 via Ollama (richardyoung/olmocr2:7b-q8)
  none:     Desabilita OCR

Processamento Paralelo:
  Use --workers para processar múltiplas imagens simultaneamente.
  Recomendado: 2-4 workers para APIs, 1-2 para modelos locais pesados.

Exemplos de uso:
  python batch_image_analyzer.py
  python batch_image_analyzer.py --mode forense --ocr glm-ocr
    python batch_image_analyzer.py --resume
  python batch_image_analyzer.py --workers 4  # Processar 4 imagens em paralelo
    python batch_image_analyzer.py --model gpt-5.4-mini --model gemma3:12b-it-q8_0
  python batch_image_analyzer.py -i ./fotos -o ./analises -w 3 --mode geral
  python batch_image_analyzer.py -i ./evidencias -o ./laudos -m forense --ocr glm-ocr -w 2
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=INPUT_DIR,
        help=f"Diretório de entrada com imagens (padrão: {INPUT_DIR})"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=Path,
        default=OUTPUT_DIR,
        help=f"Diretório de saída para relatórios (padrão: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="geral",
        choices=list(available_modes.keys()),
        help="Modo de análise (ver lista de modos disponíveis acima)"
    )
    
    parser.add_argument(
        "--ocr",
        type=str,
        default="glm-ocr",
        choices=OCR_ENGINES,
        help="Motor de OCR: 'glm-ocr' (padrão), 'olmocr2' ou 'none'"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Número de workers paralelos (1-8, padrão: 1 = sequencial)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        default=None,
        help="Modelo(s) para usar. Repita para múltiplos. Ex: --model gpt-5.4-mini --model gemma3:12b-it-q8_0. Padrão: gpt-5.4-mini"
    )
    
    parser.add_argument(
        "--gpu-profile",
        type=str,
        default=None,
        choices=["4gb", "6gb", "8gb", "16gb", "24gb", "32gb", "auto"],
        help="Perfil de GPU: seleciona modelos Ollama recomendados para o tier de VRAM (4gb/6gb/8gb/16gb/24gb/32gb/auto)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma um lote anterior usando checkpoint por imagem no diretório de saída"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Modo verbose (DEBUG)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parsear modelos especificados pelo usuário
    models_list = None
    if args.model:
        models_list = []
        for m in args.model:
            if m == OPENAI_MODEL or m.startswith("gpt"):
                models_list.append((m, "openai"))
            else:
                models_list.append((m, "ollama"))
    
    # Aplicar perfil de GPU (se nenhum --model foi especificado)
    if not models_list and args.gpu_profile:
        profile_key = args.gpu_profile
        if profile_key == "auto":
            vram = detect_vram_gb()
            profile_key = get_recommended_gpu_profile(vram)
            if profile_key:
                logger.info(f"🎮 GPU detectada: {vram:.1f} GB VRAM → perfil '{profile_key}'")
            else:
                logger.warning("⚠️ GPU não detectada ou VRAM insuficiente. Usando somente GPT-5.4-mini.")
        
        if profile_key in GPU_MODEL_PROFILES:
            models_list = [
                (model_id, "ollama") for model_id, _ in GPU_MODEL_PROFILES[profile_key]["models"]
            ]
            logger.info(f"📋 Perfil GPU '{profile_key}': {[m[0] for m in models_list]}")
            if profile_key == "32gb":
                logger.info("💡 Em GPUs de 32 GB, os modelos do tier 24 GB podem rodar com OCR simultaneamente.")
    
    # Executar análise
    analyzer = BatchImageAnalyzer(
        input_dir=args.input,
        output_dir=args.output,
        analysis_mode=args.mode,
        ocr_engine=args.ocr,
        workers=args.workers,
        models=models_list,
        resume=args.resume,
    )
    
    stats = analyzer.run()
    
    # Retornar código de saída apropriado
    if stats["failed"] > 0 and stats["processed"] == 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
