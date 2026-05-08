import faiss
import os
import json
import numpy as np

"""
FAISS VECTOR STORE MODULE

This module is responsible for:
- Creating and managing the FAISS vector index
- Storing chunk embeddings + associated metadata
- Persisting index to disk for reuse
- Performing semantic similarity search

Similarity Method:
- Uses FAISS IndexFlatIP (Inner Product)
- With vector normalization → equivalent to cosine similarity

Design Notes:
- Each embedding corresponds to a hybrid retrieval chunk
  (section → chunk)
- Metadata is stored separately but kept index-aligned
- Retrieved results preserve source traceability for citations
"""

# Paths for persistence
INDEX_PATH = "backend/data/vector_store/index.faiss"
METADATA_PATH = "backend/data/vector_store/metadata.json"

# Global variables
index = None
metadata = []


def create_faiss_index(dimension: int):
    """
    Initialize FAISS index for cosine similarity search.

    Uses:
    - IndexFlatIP (inner product)
    - L2 normalization applied during insertion/query
    """
    global index
    index = faiss.IndexFlatIP(dimension)

def add_embeddings(embeddings: list, chunks: list):
    """
    Stores embeddings in FAISS while preserving aligned chunk metadata.

    Important:
    - embeddings and chunks MUST remain in identical order
    - vectors are normalized for cosine similarity retrieval
    - metadata includes source, section, chunk_id, title, and citation info
    """
    global index, metadata

    vectors = np.array(embeddings).astype("float32")
    # normalize for cosine similarity
    faiss.normalize_L2(vectors)
    index.add(vectors)

    metadata.extend(chunks)


def save_index():
    """
    Save FAISS index + metadata to disk.
    """
    global index, metadata

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)


def load_index():
    """
    Load FAISS index + metadata from disk.
    """
    global index, metadata

    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(INDEX_PATH)

        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

        print("FAISS index loaded successfully.")
    else:
        print("No existing FAISS index found.")


"""
Threshold values affect how strong/weak want to filter out the weaker matches:
1.0 -> strict (few results)
1.5 -> balanced (moderate results)
2.0 -> loose (more results)
"""
def search(query_embedding: list, top_k: int = 5, threshold: float = 0.75):
    """
    Performs semantic similarity search over FAISS.

    Returns:
    - Top-K retrieved chunks with full metadata

    Retrieval Output:
    - chunk text
    - source document
    - section name
    - chunk_id
    - citation metadata

    Note:
    - Threshold filtering is temporarily disabled for retrieval stability
    - Next improvement: reranking for better result ordering
    """
    global index, metadata

    if index is None or index.ntotal == 0:
        return []

    query_vector = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, top_k)

    results = []
    seen = set() # want to avoid using the same chunks
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            distance = distances[0][i] # want to be able to filter out weaker matches
            chunk = metadata[idx]
            chunk_key = (
                chunk["source"],
                chunk["section"],
                chunk["chunk_id"]
            )
            if chunk_key not in seen:
                results.append(chunk)
                seen.add(chunk_key)

    return results