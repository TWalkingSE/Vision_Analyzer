#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Semantic Search - Buscador local para Relatórios Vision Analyzer
==================================================================
Vetoriza relatórios Markdown usando ChromaDB com embeddings locais via Ollama,
evitando downloads em runtime a partir do Hugging Face Hub.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"


class SemanticSearchEngine:
    """Motor de busca semântico usando ChromaDB para relatórios Markdown."""

    def __init__(
        self,
        db_path: Path,
        ollama_url: str | None = None,
        embedding_model: str | None = None,
    ):
        """
        Inicializa o buscador apontando para o banco de dados.
        
        Args:
            db_path: Onde salvar/carregar os vetores localmente.
            ollama_url: URL do servidor Ollama para embeddings.
            embedding_model: Modelo local de embeddings carregado no Ollama.
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            self.db_path.mkdir(parents=True, exist_ok=True)

        self.ollama_url = ollama_url or os.getenv(
            "VISION_SEMANTIC_SEARCH_OLLAMA_URL",
            DEFAULT_OLLAMA_URL,
        )
        self.embedding_model = embedding_model or os.getenv(
            "VISION_SEMANTIC_SEARCH_MODEL",
            DEFAULT_OLLAMA_EMBED_MODEL,
        )
            
        if not CHROMA_AVAILABLE:
            raise ImportError("Instale a biblioteca opcional: pip install chromadb")

        # Usar chroma persistente local
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Embeddings locais via Ollama evitam dependencias do HF Hub em runtime.
        self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
            url=self.ollama_url,
            model_name=self.embedding_model,
            timeout=60,
        )
        
        # Cria ou pega a collection
        self.collection = self.client.get_or_create_collection(
            name="vision_analyzer_reports",
            embedding_function=self.embedding_fn
        )

    def _build_ollama_error(self, action: str, exc: Exception) -> RuntimeError:
        return RuntimeError(
            f"Falha ao {action} na busca semântica local via Ollama. "
            f"Verifique se o servidor está ativo em {self.ollama_url} e rode "
            f"`ollama pull {self.embedding_model}` para instalar o modelo de embeddings. "
            f"Erro original: {exc}"
        )

    def index_reports(self, reports_dir: Path):
        """
        Lê todos os .md da pasta de relatórios e salva no banco vetorial.
        Atualiza documentos modificados e adiciona novos.
        """
        reports_dir = Path(reports_dir)
        if not reports_dir.exists():
            return 0
            
        md_files = list(reports_dir.glob("*.md"))
        if not md_files:
            return 0

        # Pegar os ids e metadatas que já existem
        try:
            existing = self.collection.get(include=["metadatas"])
            existing_ids = set(existing["ids"])
            existing_meta = {
                existing["ids"][i]: existing["metadatas"][i]
                for i in range(len(existing["ids"]))
            }
        except:
            existing_ids = set()
            existing_meta = {}

        new_documents = []
        new_metadatas = []
        new_ids = []
        
        update_documents = []
        update_metadatas = []
        update_ids = []

        for md_file in md_files:
            file_id = md_file.stem
            mtime = str(md_file.stat().st_mtime)
            
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            meta = {"filename": md_file.name, "path": str(md_file.absolute()), "mtime": mtime}
            
            if file_id not in existing_ids:
                # Novo documento
                new_documents.append(content)
                new_metadatas.append(meta)
                new_ids.append(file_id)
            else:
                # Verificar se foi modificado (comparar mtime)
                old_mtime = existing_meta.get(file_id, {}).get("mtime", "")
                if old_mtime != mtime:
                    update_documents.append(content)
                    update_metadatas.append(meta)
                    update_ids.append(file_id)

        # Inserir novos
        new_count = len(new_documents)
        if new_count > 0:
            batch_size = 50
            try:
                for i in range(0, new_count, batch_size):
                    self.collection.add(
                        documents=new_documents[i:i+batch_size],
                        metadatas=new_metadatas[i:i+batch_size],
                        ids=new_ids[i:i+batch_size]
                    )
            except Exception as exc:
                raise self._build_ollama_error("indexar relatórios", exc) from exc
            logger.info(f"✅ {new_count} novos relatórios indexados no Semantic Search.")
        
        # Atualizar modificados
        upd_count = len(update_documents)
        if upd_count > 0:
            batch_size = 50
            try:
                for i in range(0, upd_count, batch_size):
                    self.collection.update(
                        documents=update_documents[i:i+batch_size],
                        metadatas=update_metadatas[i:i+batch_size],
                        ids=update_ids[i:i+batch_size]
                    )
            except Exception as exc:
                raise self._build_ollama_error("atualizar relatórios", exc) from exc
            logger.info(f"🔄 {upd_count} relatórios atualizados no Semantic Search.")
            
        return new_count + upd_count

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Busca semântica no banco de dados.
        
        Args:
            query: Termo de busca em linguagem natural.
            top_k: Número de resultados.
            
        Returns:
            Lista de dicts contendo filename, snippet de texto e score de relevância.
        """
        if not query.strip():
            return []
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )
        except Exception as exc:
            raise self._build_ollama_error("consultar os relatórios", exc) from exc
        
        formatted_results = []
        
        if not results['ids'] or not results['ids'][0]:
            return formatted_results

        # ChromaDB retorna arrays de arrays
        for i in range(len(results['ids'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]  # Menor distancia (L2/Cosine) é mais similar
            
            # Cortar um snippet do documento (em volta do contexto ou do inicio)
            snippet = doc[:300].replace('\n', ' ') + "..."
            
            formatted_results.append({
                "id": results['ids'][0][i],
                "filename": meta.get("filename", "Unknown"),
                "path": meta.get("path", ""),
                "snippet": snippet,
                "distance": dist
            })

        return formatted_results

    def get_total_documents(self) -> int:
        """Retorna o numero total de documentos vetorizados."""
        try:
            return self.collection.count()
        except:
            return 0
