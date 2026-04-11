from types import SimpleNamespace

import semantic_search


def test_semantic_search_uses_local_ollama_embeddings_by_default(monkeypatch, tmp_path):
    captured = {}

    class FakeOllamaEmbeddingFunction:
        def __init__(self, url, model_name, timeout):
            captured["url"] = url
            captured["model_name"] = model_name
            captured["timeout"] = timeout

    class FakeClient:
        def __init__(self, path):
            captured["db_path"] = path

        def get_or_create_collection(self, name, embedding_function):
            captured["collection_name"] = name
            captured["embedding_function"] = embedding_function
            return SimpleNamespace()

    monkeypatch.setattr(semantic_search, "CHROMA_AVAILABLE", True)
    monkeypatch.setattr(
        semantic_search,
        "chromadb",
        SimpleNamespace(PersistentClient=FakeClient),
    )
    monkeypatch.setattr(
        semantic_search,
        "embedding_functions",
        SimpleNamespace(OllamaEmbeddingFunction=FakeOllamaEmbeddingFunction),
    )

    engine = semantic_search.SemanticSearchEngine(tmp_path / ".chroma_db")

    assert engine.ollama_url == semantic_search.DEFAULT_OLLAMA_URL
    assert engine.embedding_model == semantic_search.DEFAULT_OLLAMA_EMBED_MODEL
    assert captured["url"] == semantic_search.DEFAULT_OLLAMA_URL
    assert captured["model_name"] == semantic_search.DEFAULT_OLLAMA_EMBED_MODEL
    assert captured["timeout"] == 60
    assert captured["collection_name"] == "vision_analyzer_reports"


def test_semantic_search_allows_env_override_for_local_backend(monkeypatch, tmp_path):
    captured = {}

    class FakeOllamaEmbeddingFunction:
        def __init__(self, url, model_name, timeout):
            captured["url"] = url
            captured["model_name"] = model_name
            captured["timeout"] = timeout

    class FakeClient:
        def __init__(self, path):
            pass

        def get_or_create_collection(self, name, embedding_function):
            return SimpleNamespace()

    monkeypatch.setenv("VISION_SEMANTIC_SEARCH_OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("VISION_SEMANTIC_SEARCH_MODEL", "bge-m3")
    monkeypatch.setattr(semantic_search, "CHROMA_AVAILABLE", True)
    monkeypatch.setattr(
        semantic_search,
        "chromadb",
        SimpleNamespace(PersistentClient=FakeClient),
    )
    monkeypatch.setattr(
        semantic_search,
        "embedding_functions",
        SimpleNamespace(OllamaEmbeddingFunction=FakeOllamaEmbeddingFunction),
    )

    engine = semantic_search.SemanticSearchEngine(tmp_path / ".chroma_db")

    assert engine.ollama_url == "http://127.0.0.1:11434"
    assert engine.embedding_model == "bge-m3"
    assert captured["url"] == "http://127.0.0.1:11434"
    assert captured["model_name"] == "bge-m3"
