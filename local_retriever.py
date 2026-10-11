"""免费中文向量检索：文档与问题使用同一个 BGE 模型，按余弦相似度排序。"""
from functools import lru_cache
from pathlib import Path
import json
import os
import tempfile
import hashlib
import numpy as np
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
CACHE_DIR = Path(os.getenv("FASTEMBED_CACHE_PATH", str(Path(tempfile.gettempdir()) / "llm-universe-fastembed")))

@lru_cache(maxsize=1)
def load_knowledge():
    records = json.loads((KNOWLEDGE_DIR / "documents.json").read_text(encoding="utf-8"))
    vectors = np.load(KNOWLEDGE_DIR / "vectors.npy", allow_pickle=False)
    if vectors.shape != (len(records), 512):
        raise ValueError("知识库向量与文档数量不匹配，请重新构建知识库。")
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR), threads=2)
    return records, vectors, model

class LocalEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self.model.passage_embed(texts)]

    def embed_query(self, text):
        return next(self.model.query_embed(text)).tolist()

@lru_cache(maxsize=1)
def get_local_retriever():
    records, vectors, model = load_knowledge()
    digest = hashlib.sha256((KNOWLEDGE_DIR / "documents.json").read_bytes()).hexdigest()[:12]
    database_dir = Path(tempfile.gettempdir()) / "llm-universe-bge-chroma" / digest
    client = chromadb.PersistentClient(path=str(database_dir), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection("knowledge_bge_zh", metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        collection.add(ids=[str(i) for i in range(len(records))],
                       embeddings=vectors.tolist(),
                       documents=[record["text"] for record in records],
                       metadatas=[record["metadata"] for record in records])
    store = Chroma(client=client, collection_name="knowledge_bge_zh", embedding_function=LocalEmbeddings(model))
    return store.as_retriever(search_kwargs={"k": 4})

def retrieve(question: str):
    return get_local_retriever().invoke(question)
