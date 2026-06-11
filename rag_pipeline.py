import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from utils.clients import get_chromadb_client

class RAGPipeline:
    def __init__(self, persist_directory: str = "db"):
        self.persist_directory = persist_directory
        self.collection_name = "research"
        self.uploaded_collection_name = "uploaded_pdfs"
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_docs: List[Dict] = []
        self.client = get_chromadb_client()
        self.collection = self._create_collection(self.collection_name)
        self.uploaded_collection = self._create_collection(self.uploaded_collection_name)
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

    def _create_collection(self, name: str):
        existing = [c.name for c in self.client.list_collections()]
        if name in existing:
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        return self.client.create_collection(
            name=name,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"),
        )

    def _split_text(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            chunks.append(text[start:end].strip())
            start += self.chunk_size - self.chunk_overlap
        return [chunk for chunk in chunks if chunk]

    def build_store(self, source_documents: List[Dict]) -> None:
        texts = []
        metadatas = []
        tokenized = []
        for idx, source in enumerate(source_documents):
            raw_text = source.get("text", "")
            metadata = {
                "title": source.get("title", "Unknown title"),
                "url": source.get("url", ""),
                "source": source.get("source", "Unknown"),
                "date": source.get("date", ""),
            }
            for chunk in self._split_text(raw_text):
                texts.append(chunk)
                metadatas.append(metadata)
                tokenized.append(chunk.split())
        if texts:
            ids = [f"doc_{i}" for i in range(len(texts))]
            self.collection.add(ids=ids, metadatas=metadatas, documents=texts)
            self.bm25_index = BM25Okapi(tokenized)
            self.bm25_docs = [
                {"page_content": texts[i], "metadata": metadatas[i]} for i in range(len(texts))
            ]

    def index_uploaded_pdf(self, text: str, filename: str) -> None:
        texts = []
        metadatas = []
        for chunk in self._split_text(text):
            texts.append(chunk)
            metadatas.append({"source": "Uploaded PDF", "url": filename})
        if texts:
            ids = [f"pdf_{i}" for i in range(len(texts))]
            self.uploaded_collection.add(ids=ids, metadatas=metadatas, documents=texts)

    def semantic_search(self, query: str, k: int = 20) -> List[Dict]:
        if not self.collection:
            return []
        results = self.collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"])
        return self._parse_collection_results(results)

    def pdf_search(self, query: str, k: int = 20) -> List[Dict]:
        if not self.uploaded_collection:
            return []
        results = self.uploaded_collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"])
        return self._parse_collection_results(results)

    def bm25_search(self, query: str, k: int = 20) -> List[Tuple[Dict, float]]:
        if not self.bm25_index or not self.bm25_docs:
            return []
        tokenized = query.split()
        scores = self.bm25_index.get_scores(tokenized)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.bm25_docs[i], float(scores[i])) for i in top_indices]

    def hybrid_search(self, query: str, k: int = 20) -> List[Dict]:
        vector_results = self.semantic_search(query, k=k)
        bm25_results = self.bm25_search(query, k=k)
        combined: Dict[str, Dict] = {}
        max_vector_score = max((item.get("score", 0) for item in vector_results), default=1)
        max_bm25 = max((score for _, score in bm25_results), default=1)
        for item in vector_results:
            key = (item.get("url"), item.get("title"), item.get("abstract", "")[:120])
            combined[key] = {
                **item,
                "combined_score": 0.6 * (item.get("score", 0) / max_vector_score if max_vector_score else 0),
            }
        for doc, score in bm25_results:
            key = (doc.get("metadata", {}).get("url"), doc.get("metadata", {}).get("title"), doc.get("page_content", "")[:120])
            base = combined.get(key, {
                "title": doc.get("metadata", {}).get("title", "Document"),
                "url": doc.get("metadata", {}).get("url", ""),
                "source": doc.get("metadata", {}).get("source", "BM25"),
                "date": doc.get("metadata", {}).get("date", ""),
                "abstract": doc.get("page_content", "")[:280].strip(),
                "score": 0.0,
                "combined_score": 0.0,
            })
            bm25_score_norm = 0.4 * (score / max_bm25 if max_bm25 else 0)
            base["combined_score"] = base.get("combined_score", 0) + bm25_score_norm
            combined[key] = base
        ranked = sorted(combined.values(), key=lambda item: item.get("combined_score", 0), reverse=True)
        return ranked[:k]

    def _parse_collection_results(self, results: Dict) -> List[Dict]:
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        parsed = []
        for doc, meta, score in zip(documents, metadatas, distances):
            parsed.append({
                "title": meta.get("title", meta.get("source", "Document")),
                "url": meta.get("url", ""),
                "source": meta.get("source", "Semantic Search"),
                "date": meta.get("date", ""),
                "abstract": doc[:280].strip(),
                "score": float(score) if score is not None else 0.0,
            })
        return parsed
