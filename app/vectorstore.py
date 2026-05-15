from __future__ import annotations

import os
from typing import List
import numpy as np

import faiss
from app.catalog import load_catalog, Assessment

_index = None
_catalog_assessments: List[Assessment] = []

def get_embedding(text: str) -> np.ndarray:
    client = ZhipuAI(api_key=os.environ.get("GLM_API_KEY"))
    response = client.embeddings.create(
        model="embedding-3",
        input=text,
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def init_vectorstore():
    global _catalog_assessments
    assessments = load_catalog()
    _catalog_assessments = assessments
    print("Vectorstore initialized (Cloud Embeddings)")


def semantic_search(query: str, top_k: int = 10) -> List[Assessment]:
    global _index
    if not _catalog_assessments:
        return []

    # Lazy index creation using Cloud Embeddings
    if _index is None:
        print("Building FAISS index with Cloud Embeddings...")
        # For the catalog, we embed in bulk
        client = ZhipuAI(api_key=os.environ.get("GLM_API_KEY"))
        texts = [f"{a.name}. {a.description}"[:1000] for a in _catalog_assessments]
        
        all_embeddings = []
        # Zhipu allows bulk embeddings
        resp = client.embeddings.create(model="embedding-3", input=texts)
        all_embeddings = [r.embedding for r in resp.data]
        
        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        _index = faiss.IndexFlatL2(embeddings_np.shape[1])
        _index.add(embeddings_np)

    query_embedding = get_embedding(query)
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = _index.search(query_embedding, top_k)
    
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(_catalog_assessments):
            results.append(_catalog_assessments[idx])
    return results
