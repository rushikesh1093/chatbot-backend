import csv
import json
from pathlib import Path
from typing import Any, List, Optional, TypedDict

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


class RetrievedChunk(TypedDict):
    text: str
    source: str
    score: float


class RAGEngine:
    def __init__(
        self,
        data_dir: Path,
        vector_store_dir: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
    ) -> None:
        self.data_dir = data_dir
        self.vector_store_dir = vector_store_dir
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.vector_store_dir / "index.faiss"
        self.numpy_index_path = self.vector_store_dir / "vectors.npy"
        self.meta_path = self.vector_store_dir / "chunks.json"

        self.embedder = SentenceTransformer(model_name)
        self.dimension = self.embedder.get_sentence_embedding_dimension()

        self.index: Optional[Any] = None
        self.vectors: Optional[np.ndarray] = None
        self.chunks: List[dict] = []

        self._load_existing_index()

    def _load_existing_index(self) -> None:
        if not self.meta_path.exists():
            return

        with self.meta_path.open("r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if faiss is not None and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            return

        if self.numpy_index_path.exists():
            self.vectors = np.load(self.numpy_index_path)

    def ingest(self) -> dict:
        docs = self._load_documents()
        if not docs:
            return {"indexed": 0, "message": "No supported files found in data directory."}

        chunks: List[dict] = []
        for doc in docs:
            text_chunks = self._chunk_text(doc["text"])
            for chunk in text_chunks:
                chunks.append({"text": chunk, "source": doc["source"]})

        if not chunks:
            return {"indexed": 0, "message": "No text content found to index."}

        vectors = self.embedder.encode([c["text"] for c in chunks], convert_to_numpy=True)
        vectors = np.array(vectors, dtype=np.float32)

        if faiss is not None:
            index = faiss.IndexFlatL2(self.dimension)
            index.add(vectors)
            faiss.write_index(index, str(self.index_path))
            self.index = index
            self.vectors = None
        else:
            np.save(self.numpy_index_path, vectors)
            self.vectors = vectors
            self.index = None

        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        self.chunks = chunks

        return {
            "indexed": len(chunks),
            "files": sorted({d["source"] for d in docs}),
            "message": "Ingestion completed.",
        }

    def retrieve(self, question: str, top_k: int = 4) -> List[RetrievedChunk]:
        if not self.chunks:
            return []

        q_vec = self.embedder.encode([question], convert_to_numpy=True)
        q_vec = np.array(q_vec, dtype=np.float32)

        if self.index is not None:
            distances, indices = self.index.search(q_vec, top_k)
            distance_row = distances[0]
            index_row = indices[0]
        else:
            if self.vectors is None or len(self.vectors) == 0:
                return []

            # Fallback for environments where FAISS is unavailable.
            q = q_vec[0]
            distances_np = np.sum((self.vectors - q) ** 2, axis=1)
            k = min(top_k, len(distances_np))
            idx_sorted = np.argsort(distances_np)[:k]
            distance_row = distances_np[idx_sorted]
            index_row = idx_sorted

        results: List[RetrievedChunk] = []
        for score, idx in zip(distance_row, index_row):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "score": float(score),
                }
            )

        return results

    def _load_documents(self) -> List[dict]:
        docs: List[dict] = []

        for file_path in self.data_dir.glob("**/*"):
            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()
            try:
                if suffix == ".txt":
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    if text.strip():
                        docs.append({"source": file_path.name, "text": text})
                elif suffix == ".csv":
                    text = self._read_csv(file_path)
                    if text.strip():
                        docs.append({"source": file_path.name, "text": text})
                elif suffix == ".pdf":
                    text = self._read_pdf(file_path)
                    if text.strip():
                        docs.append({"source": file_path.name, "text": text})
                elif suffix == ".json":
                    # Load structured JSON documents and serialize into readable text
                    raw = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
                    if isinstance(raw, dict):
                        raw = [raw]

                    for entry in raw:
                        try:
                            serialized = self._serialize_json_entry(entry)
                            if serialized:
                                source_id = entry.get("id") or file_path.name
                                docs.append({"source": f"{file_path.name}#{source_id}", "text": serialized})
                        except Exception:
                            continue
                else:
                    continue
            except Exception:
                continue

        return docs

    def _serialize_json_entry(self, entry: dict) -> str:
        # Convert structured JSON entry into a human-readable text blob for embedding
        parts: List[str] = []
        if not isinstance(entry, dict):
            return ""

        # Common fields
        if "product_name" in entry:
            parts.append(f"Product: {entry.get('product_name')}")
        if "category" in entry:
            parts.append(f"Category: {entry.get('category')}")
        if "crops" in entry:
            parts.append(f"Crops: {', '.join(entry.get('crops') or [])}")
        if "benefits" in entry:
            parts.append(f"Benefits: {', '.join(entry.get('benefits') or [])}")
        if "dosage" in entry:
            parts.append(f"Dosage: {entry.get('dosage')}")
        if "application_stage" in entry:
            parts.append(f"Application stage: {entry.get('application_stage')}")
        if "frequency" in entry:
            parts.append(f"Frequency: {entry.get('frequency')}")
        if "precautions" in entry:
            parts.append(f"Precautions: {', '.join(entry.get('precautions') or [])}")

        # FAQ style
        if "faq_question" in entry and "faq_answer" in entry:
            parts.append(f"FAQ Q: {entry.get('faq_question')} A: {entry.get('faq_answer')}")

        # Crop problems
        if "crop_problem" in entry:
            parts.append(f"Problem: {entry.get('crop_problem')}")
        if "symptoms" in entry:
            parts.append(f"Symptoms: {', '.join(entry.get('symptoms') or [])}")
        if "recommended_products" in entry:
            parts.append(f"Recommended products: {', '.join(entry.get('recommended_products') or [])}")

        # Generic fallback: include any other primitive fields
        for k, v in entry.items():
            if k in {"product_name", "category", "crops", "benefits", "dosage", "application_stage", "frequency", "precautions", "faq_question", "faq_answer", "crop_problem", "symptoms", "recommended_products", "id", "type"}:
                continue
            if isinstance(v, (str, int, float)):
                parts.append(f"{k}: {v}")

        return " \n".join(parts).strip()

    def _read_csv(self, file_path: Path) -> str:
        rows: List[str] = []
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_text = ", ".join(f"{k}: {v}" for k, v in row.items())
                rows.append(row_text)
        return "\n".join(rows)

    def _read_pdf(self, file_path: Path) -> str:
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    def _chunk_text(self, text: str) -> List[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []

        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        chunks: List[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        while start < len(cleaned):
            end = start + self.chunk_size
            chunks.append(cleaned[start:end])
            start += step
        return chunks
