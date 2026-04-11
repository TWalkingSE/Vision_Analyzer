# 👁️ Vision Analyzer v4.0

Sistema avançado de análise individual e em lote com pipeline de 5 estágios: **Pré-processamento → Detecção de Objetos → OCR → LLM → Pós-processamento**. Combina OpenAI GPT-5.4-mini + Ollama local, detecção YOLO11/Detectron2, OCR via GLM/OLMoOCR2, e pós-processamento estruturado com extração de entidades, classificação e validação. A interface Streamlit e a CLI compartilham o mesmo pipeline de execução e exportação, com preflight, telemetria por etapa e reaproveitamento de dados intermediários do cache.

> **Interface gráfica Streamlit** com análise de imagem única e em lote, chat interativo com IA, detecção de objetos com auto-crop, detecção de manipulação (ELA), comparador A/B, dashboard de métricas, extração de frames de vídeo, upload drag-and-drop e visualizador de relatórios integrado.

## ✨ Funcionalidades

- **Pipeline de 5 estágios**: Pré-processamento → Detecção → OCR → LLM → Pós-processamento
- **Análise individual dedicada na UI**: permite executar apenas uma imagem selecionada sem usar o fluxo de lote
- **Pipeline compartilhado UI + CLI**: o lote da interface e a linha de comando usam a mesma orquestração de análise, cache e exportação
- **Preflight de execução**: valida OpenAI, Ollama, saída e dependências críticas antes de processar
- **Telemetria por etapa**: validação, OCR, prompt e demais estágios agora registram tempo e status no relatório
- **Checkpoint por imagem**: UI e CLI podem retomar lotes interrompidos e pular arquivos já concluídos com a mesma configuração
- **16 Formatos de Imagem**: JPG, PNG, HEIC, RAW (CR2, NEF, ARW, ORF, RW2), TIFF, WebP, AVIF, GIF, BMP, ICO
- **2 Engines de OCR via Ollama**: GLM OCR (glm-ocr:bf16) e OLMoOCR2 (richardyoung/olmocr2:7b-q8)
- **Pré-processamento avançado**: Deskew, remoção de ruído, upscale, binarização para OCR, auto-rotação EXIF
- **Detecção de Objetos**: YOLO11 (5 tamanhos) + Detectron2 (Faster R-CNN / Mask R-CNN) como alternativa
- **Pipeline sensível ao modo**: modos documentais e de telas podem pular YOLO para reduzir custo sem perder contexto útil
- **Pós-processamento estruturado**: Extração de entidades (CPF, telefone, e-mail, IP, URLs, placas), normalização, classificação automática, validação OCR vs LLM, timeline e relatório
- **Perfis de GPU**: Seleção automática de modelos por tier de VRAM (4GB / 6GB / 8GB / 16GB / 24GB / 32GB)
- **10+ Modelos de Visão IA**: OpenAI GPT-5.4-mini + modelos Ollama otimizados por GPU (Qwen3-VL, Gemma4, Qwen3.5)
- **8 Modos de Análise**: Geral, Profunda (Chain of Thought), Forense, Documentos, Screenshots/Telas, Veículos, Pessoas, Acessibilidade
- **5 Formatos de Exportação**: Markdown, JSON, HTML, PDF (com Markdown renderizado) e DOCX
- **Relatórios estruturados**: Markdown, JSON e visualização na UI incluem pré-análise, avisos/preflight e pós-processamento estruturado; HTML e PDF embutem a imagem analisada
- **Detecção de Manipulação (ELA)**: Error Level Analysis com mapa de calor e veredicto
- **Comparador A/B**: Análise lado-a-lado com 2 modelos
- **Dashboard de Métricas**: Tempo por modelo, cache hits, análises por dia
- **Re-análise Inteligente**: Re-execute LLM sem reprocessar OCR/YOLO
- **Liberação automática de VRAM**: Modelos Ollama são descarregados da GPU ao término de cada análise
- **Suporte a modelos de raciocínio**: Blocos `<think>` de modelos como Qwen3-VL são removidos automaticamente da resposta final
- **Processamento Paralelo**: Até 8 workers simultâneos
- **Cache Inteligente**: Hash MD5 + SHA-256 com reaproveitamento de OCR, YOLO, qualidade e EXIF quando aplicável
- **Busca Semântica Local**: Indexação vetorial com ChromaDB + embeddings locais via Ollama
- **Extração de Vídeo**: Decupagem automática de keyframes
- **API Utils**: Rate limiting, retry com backoff exponencial

---

## 📁 Estrutura do Projeto

```
Vision_Analyzer/
├── app.py                    # Interface gráfica principal (Streamlit)
├── streamlit_app.py          # Wrapper legado de compatibilidade para a UI
├── batch_image_analyzer.py   # Engine de análise em lote + CLI
├── batch_checkpoint.py       # Checkpoint por imagem e retomada de lotes
├── analysis_pipeline.py      # Pipeline compartilhado entre Streamlit e CLI
├── runtime_config.py         # Constantes e utilitários compartilhados de runtime
├── prompt_templates.py       # Sistema de templates de prompt
├── image_preprocessor.py     # Pré-processamento (deskew, denoise, upscale, binarização)
├── object_detector.py        # Detecção YOLO11 + Detectron2
├── post_processor.py         # Pós-processamento (entidades, classificação, validação)
├── ela_analyzer.py           # Detecção de manipulação (ELA)
├── cache_manager.py          # Cache inteligente por MD5
├── chat_assistant.py         # Chat interativo com IA
├── export_manager.py         # Exportação multi-formato
├── semantic_search.py        # Busca semântica vetorial (ChromaDB)
├── video_processor.py        # Extração de keyframes de vídeo
├── api_utils.py              # Rate limiting, retry, validação
├── requirements.txt          # Dependências Python
├── tests/                    # Testes automatizados com pytest
├── .streamlit/
│   └── config.toml           # Configuração Streamlit (tema dark)
├── imagens_entrada/          # Coloque suas imagens aqui
├── relatorios_saida/         # Relatórios gerados aqui
└── yolo11*.pt                # Modelos YOLO11 (N/S/M/L/X)
```

---

## 📦 Instalação

### Pré-requisitos

- **Python 3.10+**
- **[Ollama](https://ollama.ai)** instalado e em execução (para modelos locais)
- **OpenAI API Key** (opcional, para GPT-5.4-mini)

### 1. Clonar / baixar o projeto

```bash
git clone <repositório>
cd Vision_Analyzer
```

### 2. Criar ambiente virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

O arquivo `requirements.txt` agora traz pisos de versão para as dependências principais, deixando o ambiente mais previsível sem introduzir novas dependências obrigatórias neste ciclo de hardening.

### 3.1. PyTorch com CUDA (Aceleração GPU)

O `pip install -r requirements.txt` instala o PyTorch **CPU-only** por padrão. Para habilitar aceleração GPU (necessário para detecção de VRAM e perfis de GPU), reinstale o PyTorch com CUDA:

```bash
# Desinstalar versão CPU
pip uninstall torch torchvision torchaudio -y

# Instalar com CUDA 12.6 (ajuste conforme sua GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> Verifique sua versão do CUDA com `nvidia-smi`. Para outras versões de CUDA, consulte [pytorch.org/get-started](https://pytorch.org/get-started/locally/).
> Mesmo sem PyTorch CUDA, o sistema faz fallback via `nvidia-smi` para detectar GPU/VRAM.

> **Dependências opcionais** — consulte `requirements.txt`:
> - `detectron2` → detecção avançada Detectron2 (Faster R-CNN / Mask R-CNN) — veja [Instalação do Detectron2](#-instalação-do-detectron2-opcional)
> - `chromadb` → busca semântica local

### 4. Configurar variáveis de ambiente

```bash
# Copiar template
cp .env.example .env

# Editar com sua chave OpenAI (Windows)
notepad .env

# Editar com sua chave OpenAI (Linux/macOS)
nano .env
```

Conteúdo do `.env`:

```env
# Obrigatório apenas se usar GPT-5.4-mini
OPENAI_API_KEY=sk-sua-chave-aqui

# Opcional: Ollama em servidor remoto
# OLLAMA_HOST=http://192.168.1.100:11434

# Opcional: busca semântica local (padrão = Ollama local)
# VISION_SEMANTIC_SEARCH_OLLAMA_URL=http://localhost:11434
# VISION_SEMANTIC_SEARCH_MODEL=nomic-embed-text
```

### 5. Baixar modelos Ollama

```bash
# === Perfil 4 GB VRAM ===
ollama pull qwen3.5:2b
ollama pull qwen3-vl:2b

# === Perfil 6 GB VRAM ===
ollama pull qwen3.5:4b
ollama pull qwen3-vl:4b

# === Perfil 8 GB VRAM ===
ollama pull qwen3-vl:8b
ollama pull gemma4:e2b

# === Perfil 16 GB VRAM ===
ollama pull qwen3.5:9b-q8_0
ollama pull gemma4:e4b-it-q8_0

# === Perfil 24 GB VRAM ===
ollama pull qwen3-vl:32b
ollama pull gemma4:31b

# === Perfil 32 GB VRAM ===
# Mesmo conjunto do perfil 24 GB, com margem para rodar OCR simultaneamente
ollama pull qwen3-vl:32b
ollama pull gemma4:31b

# OCR via Ollama (pelo menos um)
ollama pull glm-ocr:bf16
ollama pull richardyoung/olmocr2:7b-q8

# Busca semântica local
ollama pull nomic-embed-text
```

> Baixe apenas os modelos do tier compatível com sua GPU. Em GPUs com 32 GB, os modelos do perfil 24 GB podem ser usados junto com OCR simultaneamente. Os modelos YOLO11 (`.pt`) já estão incluídos no projeto.

---

## � Uso

### 🎈 Interface Gráfica — Streamlit (recomendado)

```bash
streamlit run app.py
```

`streamlit_app.py` continua no projeto apenas como wrapper de compatibilidade. O entrypoint recomendado e suportado é `app.py`.

Acesse `http://localhost:8501` no navegador. A sidebar permite configurar:

- ✅ **Perfil de GPU** com auto-detecção de VRAM (4GB / 6GB / 8GB / 16GB / 24GB / 32GB)
- ✅ Seleção de modelos de IA com status de disponibilidade
- ✅ Engine de OCR
- ✅ Modo de análise
- ✅ Workers paralelos (1–8)
- ✅ Modelo YOLO para batch
- ✅ Formatos de exportação (MD, JSON, HTML, PDF, DOCX)
- ✅ Ativar/desativar cache
- ✅ Retomar lote interrompido por checkpoint

Na aba **Análise**, a interface oferece dois modos de execução:

- **Imagem única**: seleciona uma imagem específica e executa o pipeline dedicado de análise individual, com preview imediato do relatório e telemetria
- **Lote completo**: processa todas as imagens disponíveis com suporte a workers paralelos e checkpoint por imagem

Na aba **Relatórios**, cada laudo pode ser aberto em duas visões: Markdown bruto e painel estruturado alimentado pelo `.json` correspondente, com pré-análise e pós-processamento.

**10 abas** disponíveis: Análise, Chat, Detecção, ELA, Comparar, Dashboard, Vídeo HD, Relatórios, Configurações, Sobre

### 💻 Linha de Comando (CLI)

```bash
# Análise básica (usa GLM OCR, modo geral, 1 worker)
python batch_image_analyzer.py

# Com diretórios customizados
python batch_image_analyzer.py --input ./minhas_fotos --output ./analises

# Análise forense com GLM OCR e 3 workers em paralelo
python batch_image_analyzer.py --mode forense --ocr glm-ocr --workers 3

# Análise de documentos com OLMoOCR2
python batch_image_analyzer.py --mode documentos --ocr olmocr2

# Análise de screenshots, conversas e interfaces
python batch_image_analyzer.py --mode screenshots --ocr glm-ocr

# Modo debug (verbose)
python batch_image_analyzer.py --verbose

# Retomar lote interrompido no mesmo diretório de saída
python batch_image_analyzer.py --resume
```

### Argumentos CLI completos

| Argumento | Abrev. | Padrão | Descrição |
|-----------|--------|--------|-----------|
| `--input` | `-i` | `./imagens_entrada` | Diretório de entrada com imagens |
| `--output` | `-o` | `./relatorios_saida` | Diretório de saída dos relatórios |
| `--mode` | `-m` | `geral` | Modo de análise (ver tabela abaixo) |
| `--ocr` | — | `glm-ocr` | Engine OCR (ver tabela abaixo) |
| `--workers` | `-w` | `1` | Workers paralelos (1–8) |
| `--gpu-profile` | — | `auto` | Perfil de GPU: `4gb`, `6gb`, `8gb`, `16gb`, `24gb`, `32gb` ou `auto` (detecta VRAM) |
| `--resume` | — | — | Retoma o lote usando checkpoint por imagem em `.vision_batch_checkpoint.json` no diretório de saída |
| `--verbose` | `-v` | — | Ativa logs de debug |

Quando `--resume` ou a opção equivalente da UI estão ativos, o sistema compara a configuração do lote (modelos, modo, OCR, formatos de exportação e YOLO) com o checkpoint salvo. Se a configuração mudou, o checkpoint é reiniciado automaticamente para evitar pular imagens com relatórios incompatíveis.

---

## 🧠 Modos de Análise

O sistema inclui **8 modos pré-definidos** em `prompt_templates.py`, cada um com instruções detalhadas, guardrails éticos e formato de saída estruturado.

| Modo | Chave CLI | Ícone | Descrição |
|------|-----------|-------|-----------|
| **Análise Geral** | `geral` | 📷 | Descrição completa para acessibilidade e documentação. Gera alt-text WCAG, análise de sujeitos/objetos/ambiente e dados técnicos fotográficos. Inclui níveis de confiança [CERTO/PROVÁVEL/INCERTO]. |
| **Análise Profunda** | `analise_profunda` | 🧠 | Chain of Thought com semiótica, proxêmica (distâncias) e cinésica (linguagem corporal). Raciocina explicitamente antes do laudo, analisando micro-texturas e dissecando o ambiente. |
| **Análise Forense** | `forense` | 🔍 | Laudo pericial para investigação policial. Identifica suspeitos, vestígios, armas, veículos (com placa via OCR) e substâncias. Usa linguagem pericial objetiva. |
| **Documentos** | `documentos` | 📄 | Extração estruturada de documentos: identifica tipo, extrai campos (nome, CPF, CNPJ, datas, valores), valida OCR e avalia autenticidade. |
| **Screenshots/Telas** | `screenshots` | 🖥️ | Focado em capturas de tela de conversas, páginas web, e-mails e apps. Identifica a plataforma, preserva a hierarquia visual, organiza mensagens por ordem/remetente/horário e aponta sinais de edição ou montagem. |
| **Veículos** | `veiculos` | 🚗 | Identificação técnica de veículos: marca, modelo, cor, ano aproximado, estado de conservação e leitura de placa cruzada com OCR. |
| **Pessoas** | `pessoas` | 👥 | Descrição para reconhecimento: características físicas, vestuário detalhado, sinais particulares (tatuagens, cicatrizes) e análise de linguagem corporal. |
| **Acessibilidade** | `acessibilidade` | ♿ | Otimizado para geração de conteúdo acessível (WCAG 2.1 AA). Produz alt-text conciso, descrição para audiodescrição e identificação de elementos de interface. |

> **Prompts customizados** podem ser criados na aba **Configurações** da interface Streamlit e são salvos em `.vision_prompts/custom_prompts.json`.

---

## 🔍 Engines de OCR

| Motor | Chave | Tecnologia | Melhor para | Requisito |
|-------|-------|-----------|-------------|-----------|
| **🔤 GLM OCR** | `glm-ocr` | Ollama (LLM) | Documentos, contratos, texto geral (padrão) | `ollama pull glm-ocr:bf16` |
| **📖 OLMoOCR2** | `olmocr2` | Ollama (LLM) | Documentos, texto denso, alternativa | `ollama pull richardyoung/olmocr2:7b-q8` |
| **❌ Nenhum** | `none` | — | Quando OCR não é necessário | — |

> Ambos os engines utilizam **Ollama** para inferência local, sem dependências Python adicionais.

---

## 🤖 Modelos de Visão

### Modelo Padrão (API)

| Modelo | Tipo | RAM estimada | Melhor para |
|--------|------|-------------|-------------|
| `gpt-5.4-mini` | API OpenAI | — (nuvem) | Alta precisão, suporte oficial |

### Perfis de GPU — Modelos Ollama por Tier de VRAM

O sistema auto-detecta a VRAM da sua GPU e recomenda os modelos ideais:

| VRAM | Modelo 1 | Tamanho | Modelo 2 | Tamanho |
|------|----------|---------|----------|---------|
| **4 GB** | `qwen3.5:2b` | 2.7 GB | `qwen3-vl:2b` | 1.9 GB |
| **6 GB** | `qwen3.5:4b` | 3.4 GB | `qwen3-vl:4b` | 3.3 GB |
| **8 GB** | `qwen3-vl:8b` | 6.1 GB | `gemma4:e2b` | 7.2 GB |
| **16 GB** | `qwen3.5:9b-q8_0` | 11 GB | `gemma4:e4b-it-q8_0` | 12 GB |
| **24 GB** | `qwen3-vl:32b` | 21 GB | `gemma4:31b` | 20 GB |
| **32 GB** | `qwen3-vl:32b` | 21 GB | `gemma4:31b` | 20 GB |

**CLI com perfil de GPU:**
```bash
# Auto-detectar VRAM e selecionar modelos
python batch_image_analyzer.py --gpu-profile auto

# Forçar perfil específico
python batch_image_analyzer.py --gpu-profile 16gb

# Ou especificar modelos manualmente
python batch_image_analyzer.py --model qwen3-vl:8b --model gemma4:e2b
```

> O sistema usa **todos os modelos selecionados** em sequência, gerando um relatório `.md` por modelo por imagem.
>
> Em GPUs com **32 GB de VRAM**, os modelos do tier **24 GB** deixam margem para executar o OCR simultaneamente.

---

## 🎯 Detecção de Objetos

### YOLO11

O Vision Analyzer inclui os **5 tamanhos** do YOLO11 pré-baixados:

| Modelo | Tamanho | Velocidade | Precisão | Uso recomendado |
|--------|---------|-----------|---------|-----------------|
| `yolo11n` | 5.4 MB | ⚡⚡⚡⚡⚡ | ★★☆☆☆ | Preview rápido, muitas imagens |
| `yolo11s` | 18 MB | ⚡⚡⚡⚡ | ★★★☆☆ | Padrão para batch |
| `yolo11m` | 38 MB | ⚡⚡⚡ | ★★★★☆ | Equilíbrio geral |
| `yolo11l` | 49 MB | ⚡⚡ | ★★★★☆ | Alta precisão |
| `yolo11x` | 109 MB | ⚡ | ★★★★★ | Máxima precisão |

### Detectron2 (Opcional)

Alternativa para cenas complexas, com modelos Faster R-CNN e Mask R-CNN:

| Modelo | Backbone | Uso recomendado |
|--------|----------|-----------------|
| `faster_rcnn_R_50_FPN` | ResNet-50 | Detecção rápida |
| `faster_rcnn_R_101_FPN` | ResNet-101 | Alta precisão |
| `mask_rcnn_R_50_FPN` | ResNet-50 | Segmentação de instâncias |

**Instalação:**
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**Classes de interesse detectadas** (17 classes forenses):
`person`, `car`, `truck`, `motorcycle`, `bicycle`, `bus`, `cell phone`, `knife`, `scissors`, `backpack`, `handbag`, `suitcase`, `bottle`, `laptop`, `tv`, `dog`, `cat`

Os resultados YOLO são **injetados no prompt** como dados determinísticos, forçando o modelo de IA a confirmar e detalhar os objetos detectados.

---

## 🔬 Detecção de Manipulação (ELA)

O módulo `ela_analyzer.py` implementa **Error Level Analysis (ELA)**, uma técnica forense que detecta regiões potencialmente manipuladas em imagens JPEG.

### Como funciona

1. A imagem é **re-salva** em JPEG com qualidade conhecida (padrão: 95%)
2. Calcula a **diferença absoluta** pixel a pixel entre original e recompressão
3. A diferença é **amplificada** para visualização (padrão: ×15)
4. Regiões editadas (clone, colagem, inpainting) apresentam **níveis de erro diferentes** do restante

### Modos de Visualização (aba 🔬 ELA)

| Modo | Descrição |
|------|-----------|
| **ELA Amplificado** | Imagem de diferença amplificada (padrão) |
| **Mapa de Calor** | Paleta colorida: azul (baixo erro) → vermelho (alto erro) |
| **Overlay** | Sobreposição do mapa de calor sobre a imagem original |

### Classificação Automática

| Veredicto | Critério |
|-----------|----------|
| ✅ **Sem indícios** | Erro médio < 5 e pixels suspeitos < 0.5% |
| ⚠️ **Inconclusivo** | Erro médio entre 5 e 20 — possível recompressão múltipla |
| 🚨 **Suspeito** | Erro médio > 20 ou hotspots concentrados (≥2 regiões) |

### Integração com modo Forense

No `batch_image_analyzer.py`, quando o modo de análise é **forense**, o ELA é executado automaticamente e o resultado é injetado no prompt do perito IA como dado adicional em `{quality_result}`.

### Parâmetros Configuráveis

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| Qualidade JPEG | 95 | Qualidade de recompressão (90-100) |
| Fator de amplificação | 15 | Multiplicador da diferença (1-50) |
| Opacidade do overlay | 50% | Transparência da sobreposição (10-90%) |

> **Limitações:** ELA funciona melhor em JPEGs originais. Imagens PNG, screenshots ou já recomprimidas várias vezes podem gerar falsos positivos. Use sempre em conjunto com outros métodos de análise.

---

## ⚡ Processamento Paralelo

| Workers | Modo | Recomendação |
|---------|------|--------------|
| `1` | Sequencial | Padrão — seguro para todos os casos |
| `2–4` | Paralelo | Ideal para API OpenAI ou imagens leves |
| `1–2` | Paralelo | Para modelos Ollama locais pesados (14B+) |
| `4–8` | Paralelo | Apenas para hardware com muita RAM/VRAM |

No modo paralelo, o progresso exibe:
- Imagens concluídas: `X/Y`
- Análises individuais: `A/B (XX%)`
- Cache hits (análises recuperadas do cache)

---

## 🗄️ Cache Inteligente

O `CacheManager` gera um **hash MD5** de cada imagem e armazena a configuração usada (modelo, modo, OCR). Em execuções futuras:

1. Calcula o MD5 da imagem
2. Verifica o índice em `.vision_cache/cache_index.json`
3. Se encontrado com a mesma configuração e com os artefatos completos → retorna o relatório já existente
4. Se não encontrado por completo, mas houver uma análise anterior da mesma imagem → reaproveita OCR, YOLO, qualidade e EXIF quando compatíveis
5. Se não houver nada reutilizável → executa análise completa e armazena no cache

Para ativar no Streamlit, marque **"Usar cache"** nas Configurações.

---

## � Busca Semântica

O módulo `semantic_search.py` usa **ChromaDB** + **embeddings locais via Ollama** para indexar todos os relatórios `.md` gerados e permitir buscas em linguagem natural, sem depender do Hugging Face Hub em runtime.

**Dependências adicionais:**
```bash
pip install chromadb
ollama pull nomic-embed-text
```

**Como usar (Streamlit → aba Relatórios → Busca Semântica):**
1. Clique em **"Indexar Relatórios"** para vetorizar os `.md` existentes
2. Digite uma busca em linguagem natural, ex: `"pessoa com mochila azul à noite"`
3. O sistema retorna os relatórios mais similares com score de distância
4. Clique em **Abrir relatório** para carregar diretamente o laudo correspondente no visualizador da aba ao lado

Por padrão, o modelo de embedding usado é `nomic-embed-text` no Ollama local. Se quiser trocar, defina `VISION_SEMANTIC_SEARCH_MODEL` no `.env`.

---

## 🎞️ Extração de Vídeo

O `video_processor.py` converte vídeos em lotes de imagens para análise:

1. Faça upload do vídeo na aba **Vídeo** da interface
2. Configure o intervalo de frames (ex: `30` = 1 frame por segundo em vídeo 30 FPS)
3. Os frames são salvos em `./imagens_entrada/` com timestamp no nome (`video_00m01s.jpg`)
4. Execute a análise em lote normalmente

**Formatos suportados:** MP4, AVI, MOV, MKV

---

## 📤 Exportação de Relatórios

| Formato | Extensão | Requisito | Descrição |
|---------|----------|-----------|-----------|
| **Markdown** | `.md` | — | Sempre gerado, formato padrão, com seção de execução da pipeline |
| **JSON** | `.json` | — | Estruturado com `pre_analysis`, telemetria e `post_processing` para integração com outros sistemas |
| **HTML** | `.html` | — | Página web com estilo, imagem analisada incorporada e resumo da execução |
| **PDF** | `.pdf` | `reportlab` | Documento profissional com capa, índice e imagem analisada |
| **DOCX** | `.docx` | `python-docx` | Editável no Microsoft Word |

**PDF Consolidado**: Na aba Relatórios, o botão **"📑 Gerar PDF Consolidado"** agrupa todos os relatórios em um único PDF com capa, índice, tabelas de metadados e o conteúdo de cada análise.

Os arquivos `.md`, `.json`, `.html`, `.pdf` e `.docx` saem do mesmo `ExportManager`, então os campos de OCR, EXIF, YOLO, qualidade e pós-processamento permanecem alinhados entre UI e CLI.

Nos relatórios individuais, os formatos `.html` e `.pdf` agora incluem a imagem analisada incorporada no próprio arquivo. Os formatos `.md`, `.json` e `.html` também passam a expor avisos de preflight e telemetria da pipeline quando disponíveis. O `.docx` permanece focado no conteúdo textual e estruturado.

---

## ✅ Testes

```bash
pytest
```

Testes adicionados nesta versão:

- `tests/test_export_manager.py`: valida exportação Markdown/JSON e HTML/PDF com pré-análise, telemetria, pós-processamento e imagens incorporadas
- `tests/test_analysis_pipeline.py`: valida o pipeline compartilhado, preflight, short-circuit de cache e reutilização de intermediários
- `tests/test_gpu_profiles.py`: valida tiers de GPU e slugs estáveis de nomes de modelo
- `tests/test_cache_manager.py`: valida hit de cache, coleta de intermediários e invalidação de relatórios ausentes
- `tests/test_api_utils.py`: valida retry com backoff e saneamento/validação de input
- `tests/test_batch_checkpoint.py`: valida checkpoint por imagem, reset por mudança de configuração e invalidação de relatórios ausentes
- `tests/test_semantic_search.py`: valida backend local Ollama e override de modelo via variável de ambiente

---

## � Chat Interativo

A aba **Chat** permite fazer perguntas sobre uma imagem selecionada:

- Selecione a imagem e o modelo na sidebar esquerda
- A imagem é enviada apenas na **primeira mensagem** (economia de tokens)
- O `ChatAssistant` é **persistido no `session_state`** do Streamlit — não recria a sessão a cada mensagem
- Suporta **streaming** de respostas (caractere a caractere)
- Botão **"Nova Sessão"** limpa o histórico e reinicia o contexto

---

## ⚙️ Pipeline de Processamento

```
┌──────────────────────────────────────────────────────────────┐
│  1. DESCOBERTA DE IMAGENS                                    │
│     Busca recursiva em ./imagens_entrada                     │
│     Suporte a 16 extensões (JPG, PNG, RAW, HEIC, etc.)       │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  2. PREFLIGHT DE EXECUÇÃO                                    │
│     • Valida OpenAI, Ollama e diretório de saída             │
│     • Emite avisos para etapas opcionais indisponíveis       │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  3. VERIFICAÇÃO DE CACHE                                     │
│     MD5 da imagem → verifica .vision_cache/cache_index.json  │
│     Cache HIT total → retorna relatório existente            │
│     Cache parcial → reaproveita OCR/YOLO/EXIF/qualidade      │
│     Cache MISS → continua o pipeline completo                │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  4. PRÉ-PROCESSAMENTO (image_preprocessor.py)                │
│     • RAW (CR2/NEF/ARW) → RGB via rawpy                     │
│     • HEIC/HEIF → RGB via pillow-heif                        │
│     • Auto-rotação via dados EXIF (Orientation tag)          │
│     • Deskew (correção de inclinação via Hough Lines)        │
│     • Remoção de ruído (fastNlMeansDenoising)                │
│     • Upscale automático para imagens pequenas (<640px)      │
│     • Binarização adaptativa para OCR                        │
│     • Redimensionamento máximo 2048×2048 px                  │
│     • Avaliação de qualidade: blur, brilho, contraste        │
│     • Codificação JPEG em memória → base64                   │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  5. DETECÇÃO DE OBJETOS (object_detector.py)                 │
│     • YOLO11: N / S / M / L / X (padrão)                    │
│     • Detectron2: Faster R-CNN / Mask R-CNN (opcional)       │
│     • Pode ser pulada em modos documentais                   │
│     • 80 classes COCO → 17 classes forenses prioritárias     │
│     • Resultado injetado no prompt como dado factual         │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  6. OCR (engine selecionado)                                 │
│     • GLM OCR (glm-ocr:bf16) → via Ollama (padrão)          │
│     • OLMoOCR2 (7b-q8) → via Ollama (alternativa)           │
│     • Fallback automático entre engines quando cabível       │
│     • Resultado injetado no prompt como ground truth         │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  7. CONSTRUÇÃO DO PROMPT (prompt_templates.py)               │
│     • Template selecionado (7 modos pré-definidos)           │
│     • Variáveis injetadas: OCR, YOLO, Qualidade, EXIF        │
│     • Fallback para prompts locais se módulo indisponível    │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  8. ANÁLISE COM IA (para cada modelo selecionado)            │
│     • Rate limiting (api_utils.py) → OpenAI + Ollama         │
│     • GPT-5.4-mini via OpenAI API (max 4096 tokens)          │
│     • Qwen3-VL / Gemma4 / Qwen3.5 via Ollama (8192 tokens)   │
│     • Blocos <think> removidos automaticamente               │
│     • Respostas vazias rejeitadas com placeholder             │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  9. PÓS-PROCESSAMENTO (post_processor.py)                    │
│     • Extração de entidades: CPF, telefone, e-mail, IP,      │
│       URLs, placas veiculares                                │
│     • Normalização de texto (encoding, limpeza OCR)          │
│     • Classificação automática (tipo documento + ameaça)     │
│     • Validação cruzada OCR vs LLM (score consistência)      │
│     • Extração de timeline a partir de timestamps            │
│     • Geração de sumário automatizado                        │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  10. GERAÇÃO DE RELATÓRIOS (export_manager.py)               │
│     • 1 arquivo .md por imagem × modelo                      │
│     • Exportação adicional: JSON / HTML / PDF / DOCX         │
│     • HTML/PDF incorporam a imagem analisada                 │
│     • Markdown/JSON/HTML incluem avisos e telemetria         │
│     • Entrada no cache (MD5 + configuração)                  │
│     • Mesmo fluxo compartilhado entre Streamlit e CLI        │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  11. LIBERAÇÃO DE VRAM                                       │
│     • Modelos Ollama descarregados via keep_alive=0          │
│     • Libera GPU imediatamente após análise (sem timeout)    │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔐 Segurança e API Utils (`api_utils.py`)

| Funcionalidade | Configuração Padrão |
|----------------|---------------------|
| **Retry com backoff exponencial** | 3 tentativas, delay inicial 1s, máx 60s, jitter ativo |
| **Rate Limiter OpenAI** | Separado por instância (`get_openai_limiter()`) |
| **Rate Limiter Ollama** | Separado por instância (`get_ollama_limiter()`) |
| **Timeout** | Configurável por chamada |
| **Validação de arquivo** | Tamanho máx 20 MB, dimensão 10–10000 px |
| **Extensões válidas** | JPG, PNG, GIF, BMP, TIFF, WebP, HEIC, HEIF, RAW, AVIF, ICO |

---

## � Performance Estimada

| Formato | Tempo por modelo | Observação |
|---------|-----------------|------------|
| JPEG/PNG pequeno (<1 MB) | 2–5s | |
| JPEG/PNG grande (>5 MB) | 4–10s | Redimensionado para 2048px antes |
| HEIC (iPhone) | 3–7s | Conversão adicional |
| RAW (CR2/NEF) | 5–12s | Conversão rawpy mais lenta |
| Cache HIT | <0.1s | Lê relatório já existente |

*Tempos variam com hardware, modelo escolhido e carga do servidor Ollama.*

---
## 🦏 Instalação do Detectron2 (Opcional)

O **Detectron2** é a engine de detecção avançada do Facebook Research (Faster R-CNN / Mask R-CNN). É **opcional** — o YOLO11 já cobre a maioria dos casos. O Detectron2 é recomendado apenas para cenas complexas que exigem segmentação de instâncias.

> **Nota:** O Detectron2 **não possui suporte oficial para Windows**. Abaixo estão as alternativas disponíveis.

### Opção 1 — WSL2 (Recomendado para Windows)

A forma mais confiável de usar o Detectron2 no Windows é via **WSL2** (Windows Subsystem for Linux):

```bash
# 1. Instalar WSL2 (PowerShell como Admin)
wsl --install

# 2. Dentro do WSL2 (Ubuntu):
sudo apt update && sudo apt install -y python3-pip python3-venv build-essential

# 3. Criar ambiente virtual
python3 -m venv venv && source venv/bin/activate

# 4. Instalar PyTorch com CUDA (ajuste a versão CUDA conforme sua GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Instalar Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Opção 2 — Build direto no Windows (Avançado)

Requer **Visual Studio Build Tools** com workload C++ e **PyTorch com CUDA** já instalados:

```powershell
# 1. Instalar PyTorch com CUDA (ajuste cu121 conforme sua GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Instalar dependências de compilação
pip install ninja cython

# 3. Clonar e instalar o Detectron2
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
```

> ⚠️ Este método pode falhar dependendo da versão do MSVC, Python e PyTorch. Se encontrar erros de compilação, use a Opção 1 (WSL2).

### Opção 3 — Wheels pré-compilados (Experimental)

Algumas versões possuem wheels pré-compilados para Windows mantidos pela comunidade:

```powershell
# Verifique a versão do seu PyTorch e CUDA:
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

# Tente instalar via wheels (pode não estar disponível para sua versão):
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.1/index.html
```

> Substitua `cu121` e `torch2.1` pelas versões correspondentes ao seu ambiente. Consulte a [documentação oficial](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) para URLs atualizadas.

### Verificar instalação

```python
python -c "from detectron2 import model_zoo; print('Detectron2 OK')"
```

Se o import funcionar sem erros, o Detectron2 está pronto para uso na aba **Detecção** do Vision Analyzer.

---
## �🛠️ Troubleshooting

### "Ollama não acessível"
```bash
# Verificar se Ollama está rodando
ollama list

# Iniciar servidor manualmente
ollama serve

# Para servidor remoto, configure no .env:
# OLLAMA_HOST=http://192.168.1.100:11434
```

### "OPENAI_API_KEY não encontrada"
```bash
# Verificar se o arquivo .env existe e tem a chave
cat .env

# Ou exportar diretamente (sessão atual)
export OPENAI_API_KEY=sk-...  # Linux/macOS
set OPENAI_API_KEY=sk-...     # Windows CMD
```

### "ultralytics não instalado" (YOLO desativado)
```bash
pip install ultralytics
```

### "Suporte a HEIC desabilitado"
```bash
pip install pillow-heif
```

### "Suporte a RAW desabilitado"
```bash
pip install rawpy imageio
```

### "PaddleOCR não disponível"
```bash
pip install paddlepaddle paddleocr
```

### "EasyOCR não disponível"
```bash
pip install easyocr
```

### "Busca Semântica indisponível"
```bash
pip install chromadb
ollama pull nomic-embed-text
```

### "GPU não detectada" (VRAM mostra 0 GB)

O PyTorch pode estar instalado sem suporte CUDA (versão CPU-only). Verifique:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Se mostrar `False`, reinstale com CUDA:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> O sistema possui fallback via `nvidia-smi` para detectar GPU/VRAM mesmo com o PyTorch CPU-only, mas a aceleração CUDA não funcionará sem a versão correta.

### "StreamlitAPIException: session_state cannot be modified after widget"
Não atribuir manualmente `st.session_state[key]` após declarar um widget com o mesmo `key`. O Streamlit gerencia automaticamente.

---

## 📄 Dependências Principais

| Pacote | Versão mínima | Uso |
|--------|--------------|-----|
| `streamlit` | 1.30.0 | Interface gráfica |
| `openai` | 1.30.0 | API GPT-5.4-mini |
| `ollama` | 0.2.0 | Modelos locais |
| `Pillow` | 10.0.0 | Processamento de imagens |
| `pillow-heif` | 0.16.0 | Suporte HEIC/HEIF |
| `rawpy` | 0.19.0 | Suporte RAW |
| `torch` | 2.11.0 | Aceleração GPU (CUDA) |
| `torchvision` | 0.26.0 | Suporte visão PyTorch |
| `ultralytics` | 8.0.0 | YOLO11 |
| `reportlab` | 4.0.0 | Exportação PDF |
| `python-docx` | 1.1.0 | Exportação DOCX |
| `chromadb` | — | Busca semântica vetorial local |
| `opencv-python-headless` | 4.8.0 | Detecção de blur/faces |
| `numpy` | 1.24.0 | Processamento numérico |
| `python-dotenv` | 1.0.0 | Variáveis de ambiente |
| `tqdm` | 4.0.0 | Barra de progresso na extração de vídeo |
| `pytest` | 8.0.0 | Testes automatizados |

---

## 📄 Licença

MIT License — Veja o arquivo [LICENSE](LICENSE) para detalhes.

---

*Desenvolvido com 🐍 Python 3.10+ · 🤖 OpenAI · 🦙 Ollama · 🎯 YOLO11 · 🎈 Streamlit*
