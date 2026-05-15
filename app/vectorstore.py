from __future__ import annotations

import os
from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    SentenceTransformer = None
    faiss = None

from app.catalog import load_catalog, Assessment

# We will use a fast CPU-friendly model for embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None
_index = None
_catalog_assessments: List[Assessment] = []


def _get_model():
    global _model
    if _model is None and SentenceTransformer is not None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_embedding(text: str) -> np.ndarray:
    model = _get_model()
    if model is None:
        return np.zeros(384, dtype=np.float32) # Fallback if not installed
    return model.encode(text, convert_to_numpy=True)


def init_vectorstore():
    global _index, _catalog_assessments
    if faiss is None or _get_model() is None:
        print("Warning: FAISS or SentenceTransformer not installed. Vectorstore skipped.")
        return

    assessments = load_catalog()
    _catalog_assessments = assessments

    # Build corpus
    texts = []
    for a in assessments:
        # Create a rich text representation of the assessment
        text = f"{a.name}. Type: {a.test_type_label}. Levels: {', '.join(a.job_levels)}. Description: {a.description}"
        texts.append(text)

    if not texts:
        return

    embeddings = _get_model().encode(texts, convert_to_numpy=True)
    
    # Initialize FAISS index
    d = embeddings.shape[1]
    _index = faiss.IndexFlatL2(d)
    _index.add(embeddings)


def semantic_search(query: str, top_k: int = 10) -> List[Assessment]:
    model = _get_model()
    if _index is None or not _catalog_assessments or model is None:
        # Fallback to heuristic if vectorstore isn't initialized
        from app.catalog import find_by_name_fragment
        return find_by_name_fragment(query)[:top_k]

    query_embedding = model.encode(query, convert_to_numpy=True)
    # FAISS requires 2D array
    query_embedding = np.expand_dims(query_embedding, axis=0)

    distances, indices = _index.search(query_embedding, top_k)
    
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(_catalog_assessments):
            results.append(_catalog_assessments[idx])
            
    return results
