"""
rag_tools/rag_tools.py

A robust, modular RAG pipeline using FAISS + sentence-transformers with
multiple chunking strategies and tight integration with PromptingTechniques.

Author: Suraj (adapted assistant)
"""

from __future__ import annotations
import os
import json
import uuid
import logging
from typing import List, Optional, Tuple, Dict, Any, Iterable

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Import your prompter module
from prompting_tools.prompter import PromptingTechniques

logger = logging.getLogger("rag_tools")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)


# ---------------------------
# Helper utilities
# ---------------------------

def _batch(iterable: List[str], size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def _normalize_vectors(v: np.ndarray) -> np.ndarray:
    """L2-normalize vectors along axis 1 (inplace copy)."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0.0] = 1e-8
    return v / norms


# ---------------------------
# Chunking strategies
# ---------------------------

def fixed_chunk(text: str, chunk_size_chars: int = 1000, overlap_chars: int = 200) -> List[str]:
    """Simple fixed-size chunking in characters with overlap."""
    chunks = []
    start = 0
    L = len(text)
    if L == 0:
        return []
    while start < L:
        end = min(L, start + chunk_size_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap_chars
        if start < 0:
            start = 0
        if start >= L:
            break
    return chunks


def sliding_window_chunk(text: str, window_words: int = 120, step_words: int = 60) -> List[str]:
    """Sliding window chunking by approximate words."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        chunk_words = words[i:i + window_words]
        chunks.append(" ".join(chunk_words).strip())
        i += step_words
    return chunks


def paragraph_chunk(text: str) -> List[str]:
    """Split by blank lines (paragraphs)."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras


def sentence_chunk(text: str) -> List[str]:
    """A light-weight sentence splitter using punctuation."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(text: str,
                   embedder: SentenceTransformer,
                   sim_threshold: float = 0.75,
                   min_chunk_words: int = 40,
                   max_chunk_words: int = 220) -> List[str]:
    """
    Semantic chunking:
    - split into sentences
    - compute embeddings for sentences
    - merge sentences into chunks; start a new chunk when consecutive sentence similarity < sim_threshold
      or when chunk size exceeds max_chunk_words; also ensure min_chunk_words boundary.
    """
    sents = sentence_chunk(text)
    if not sents:
        return []

    # embed sentences in batches
    sent_embs = []
    for batch in _batch(sents, 64):
        sent_embs.append(embedder.encode(batch, convert_to_numpy=True))
    sent_embs = np.vstack(sent_embs)
    sent_embs = _normalize_vectors(sent_embs)

    chunks = []
    cur_sent_indices: List[int] = []
    cur_word_count = 0

    def flush_current():
        nonlocal cur_sent_indices, cur_word_count
        if not cur_sent_indices:
            return
        chunk_text = " ".join([sents[i] for i in cur_sent_indices]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        cur_sent_indices = []
        cur_word_count = 0

    for i in range(len(sents)):
        if not cur_sent_indices:
            cur_sent_indices.append(i)
            cur_word_count = len(sents[i].split())
            continue

        # compute similarity between last sentence in cur and this sentence
        last_idx = cur_sent_indices[-1]
        sim = float(np.dot(sent_embs[last_idx], sent_embs[i]))
        # logic: if similarity low AND we already have min_chunk_words, flush
        if (sim < sim_threshold and cur_word_count >= min_chunk_words) or (cur_word_count >= max_chunk_words):
            flush_current()
            cur_sent_indices = [i]
            cur_word_count = len(sents[i].split())
        else:
            cur_sent_indices.append(i)
            cur_word_count += len(sents[i].split())

    flush_current()
    return chunks


def hierarchical_chunk(text: str,
                       embedder: Optional[SentenceTransformer] = None,
                       chunk_size_chars: int = 1000,
                       overlap_chars: int = 200) -> List[str]:
    """
    Hierarchical chunking:
    1. Split by paragraphs.
    2. For each paragraph:
       - If small, keep as chunk.
       - If large, split by sentences and apply either semantic or fixed chunking fallback.
    """
    paragraphs = paragraph_chunk(text)
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size_chars:
            chunks.append(para)
            continue
        # larger paragraph -> sentence split
        sents = sentence_chunk(para)
        # try semantic chunking if embedder provided
        if embedder is not None:
            sub_chunks = semantic_chunk(para, embedder)
            if sub_chunks:
                chunks.extend(sub_chunks)
                continue
        # fallback: fixed chunk on paragraph
        sub = fixed_chunk(para, chunk_size_chars, overlap_chars)
        chunks.extend(sub)
    return chunks


# map names to functions
CHUNKERS = {
    "fixed": fixed_chunk,
    "sliding_window": sliding_window_chunk,
    "paragraph": paragraph_chunk,
    "sentence": sentence_chunk,
    "semantic": semantic_chunk,
    "hierarchical": hierarchical_chunk,
}


# ---------------------------
# RAGPipeline Class
# ---------------------------

class RAGPipeline:
    """
    RAGPipeline encapsulates the full flow:
    - load text/pdf
    - chunk (multiple methods)
    - embed & store in FAISS
    - retrieve and ask LLM via PromptingTechniques
    """

    def __init__(
        self,
        embed_model_name: str = "all-MiniLM-L6-v2",
        faiss_index_path: str = "faiss_index.bin",
        chunks_json_path: str = "chunks.json",
        normalize_embeddings: bool = True,
        embedding_batch_size: int = 32,
        prompter_model: str = "mistral:latest",
    ):
        self.embed_model_name = embed_model_name
        self.faiss_index_path = faiss_index_path
        self.chunks_json_path = chunks_json_path
        self.normalize_embeddings = normalize_embeddings
        self.embedding_batch_size = embedding_batch_size

        # load embedder lazily
        self._embedder: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._chunks: List[Dict[str, Any]] = []  # list of dicts: {id, text}
        # prompter
        self.prompter = PromptingTechniques(model=prompter_model)

    # -----------------------
    # embedder property
    # -----------------------
    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            # support common huggingface names (map to full)
            # many users pass "all-MiniLM-L6-v2" (short) or full "sentence-transformers/.."
            name = self.embed_model_name
            # try common full name patterns
            candidates = [
                name,
                f"sentence-transformers/{name}",
                f"all-{name}"  # fallback
            ]
            for c in candidates:
                try:
                    self._embedder = SentenceTransformer(c)
                    logger.info("Loaded embedder: %s", c)
                    break
                except Exception:
                    continue
            if self._embedder is None:
                # final attempt
                self._embedder = SentenceTransformer(self.embed_model_name)
                logger.info("Loaded embedder: %s", self.embed_model_name)
        return self._embedder

    # -----------------------
    # load / read
    # -----------------------
    def load_pdf(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        text_parts = []
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for p in reader.pages:
                try:
                    t = p.extract_text() or ""
                except Exception:
                    t = ""
                text_parts.append(t)
        text = "\n\n".join(text_parts)
        return text

    def load_text(self, text_path: str) -> str:
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"text file not found: {text_path}")
        with open(text_path, "r", encoding="utf-8") as f:
            return f.read()

    # -----------------------
    # chunking
    # -----------------------
    def chunk_text(self,
                   text: str,
                   method: str = "fixed",
                   overwrite: bool = True,
                   **kwargs) -> List[Dict[str, Any]]:
        """
        Chunk text using chosen method. Returns list of chunk dicts: {"id", "text", "meta"}
        method: one of CHUNKERS keys.
        kwargs are forwarded to chunker functions (e.g., chunk_size_chars, window_words, sim_threshold, etc.)
        """
        if method not in CHUNKERS:
            raise ValueError(f"Unknown chunking method '{method}'. Available: {list(CHUNKERS.keys())}")

        if method in ("semantic", "hierarchical"):
            # semantic/hierarchical need embedder available, so ensure we reference embedder property
            chunks_raw = CHUNKERS[method](text, self.embedder, **kwargs)
        else:
            chunks_raw = CHUNKERS[method](text, **kwargs)

        chunk_objs = []
        for i, c in enumerate(chunks_raw):
            chunk_objs.append({"id": str(uuid.uuid4()), "chunk_index": i, "text": c})

        if overwrite:
            self._chunks = chunk_objs
        else:
            self._chunks.extend(chunk_objs)

        logger.info("Chunking completed: method=%s -> %d chunks", method, len(chunk_objs))
        return self._chunks

    # -----------------------
    # embeddings + faiss
    # -----------------------
    def build_faiss(self, force_rebuild: bool = False) -> None:
        """
        Build FAISS index from current self._chunks.
        - saves index to faiss_index_path
        - saves chunks metadata to chunks_json_path
        """
        if not self._chunks:
            raise ValueError("No chunks available. Run chunk_text first.")

        if os.path.exists(self.faiss_index_path) and not force_rebuild:
            logger.info("FAISS index already exists at %s. Load it instead or set force_rebuild=True", self.faiss_index_path)
            return

        texts = [c["text"] for c in self._chunks]
        # compute embeddings batch-wise
        all_embs = []
        for batch in _batch(texts, self.embedding_batch_size):
            emb = self.embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embs.append(emb)
        emb_matrix = np.vstack(all_embs).astype("float32")

        if self.normalize_embeddings:
            faiss.normalize_L2(emb_matrix)

        dim = emb_matrix.shape[1]
        # Using inner product on normalized vectors = cosine similarity
        index = faiss.IndexFlatIP(dim)
        index.add(emb_matrix)
        self._index = index

        # Save index and chunks
        faiss.write_index(self._index, self.faiss_index_path)
        with open(self.chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)
        logger.info("Built FAISS index (dim=%d, n=%d) and saved at %s", dim, index.ntotal, self.faiss_index_path)

    def load_faiss(self) -> None:
        if not os.path.exists(self.faiss_index_path) or not os.path.exists(self.chunks_json_path):
            raise FileNotFoundError("FAISS index or chunks JSON not found on disk.")
        self._index = faiss.read_index(self.faiss_index_path)
        with open(self.chunks_json_path, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)
        logger.info("Loaded FAISS index from %s (n=%d) and %d chunks", self.faiss_index_path, self._index.ntotal, len(self._chunks))

    # -----------------------
    # retrieval
    # -----------------------
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Retrieve top_k chunks for a query.
        Returns list of (score, chunk_obj) where score is cosine similarity [-1..1]
        """
        if self._index is None:
            raise ValueError("FAISS index is not loaded. Call build_faiss() or load_faiss() first.")
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        if self.normalize_embeddings:
            faiss.normalize_L2(q_emb)
        D, I = self._index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            chunk = self._chunks[int(idx)]
            results.append((float(score), chunk))
        return results

    # -----------------------
    # prompt + ask LLM (using PromptingTechniques)
    # -----------------------
    def ask(self,
            query: str,
            top_k: int = 3,
            technique: str = "zero_shot",
            system_prompt: Optional[str] = None,
            include_scores: bool = True) -> str:
        """
        Perform retrieval and then call prompting technique (from PromptingTechniques).
        technique must be a method name present on PromptingTechniques (zero_shot, few_shot, chain_of_thought, ...).
        """
        if not hasattr(self.prompter, technique):
            raise ValueError(f"Technique '{technique}' not found in prompter. Available: {', '.join([m for m in dir(self.prompter) if not m.startswith('_')])}")

        # retrieve
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            return "[No retrieval results]"

        # build context
        pieces = []
        for score, chunk in retrieved:
            if include_scores:
                pieces.append(f"(score:{score:.3f})\n{chunk['text']}")
            else:
                pieces.append(chunk['text'])
        context = "\n\n---\n\n".join(pieces)

        # final prompt - you can customize this template
        final_prompt = (
            "You are an expert assistant. Use the context below (extracted from documents) to answer the question. "
            "If the answer cannot be found in the context, say you don't know. Cite which chunk(s) you used.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nAnswer concisely:"
        )

        # call prompter method
        method = getattr(self.prompter, technique)
        try:
            response = method(final_prompt, system_prompt)
            return response
        except Exception as e:
            logger.exception("Prompter failed, returning direct error text.")
            return f"[Prompter error] {e}"

    # -----------------------
    # utility
    # -----------------------
    def available_chunkers(self) -> List[str]:
        return list(CHUNKERS.keys())

    def available_embedding_models(self) -> List[str]:
        # Common recommended models (user can pass any sentence-transformers name)
        return [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1",
            "distiluse-base-multilingual-cased-v2",
            "sentence-transformers/all-MiniLM-L6-v2"
        ]

    def save_state(self, index_path: Optional[str] = None, chunks_path: Optional[str] = None) -> None:
        if self._index is None:
            raise ValueError("Index not built.")
        faiss.write_index(self._index, index_path or self.faiss_index_path)
        with open(chunks_path or self.chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)
        logger.info("Saved FAISS index and chunks.")

    def load_state(self, index_path: Optional[str] = None, chunks_path: Optional[str] = None) -> None:
        self.faiss_index_path = index_path or self.faiss_index_path
        self.chunks_json_path = chunks_path or self.chunks_json_path
        self.load_faiss()