import faiss
import os
import json
import numpy as np

"""
FAISS VECTOR STORE MODULE

This module is responsible for:
- Creating and managing FAISS vector index
- Storing embeddings + associated metadata
- Persisting index to disk for reuse
- Performing similarity search over stored vectors

Similarity Method:
- Uses FAISS IndexFlatIP (Inner Product)
- With vector normalization → equivalent to cosine similarity

Design Note:
- Metadata is currently stored separately from FAISS index
- Each embedding corresponds 1-to-1 with a text section
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

def add_embeddings(embeddings: list, texts: list):
    """
    Stores embeddings in FAISS and maintains parallel metadata list.

    Important:
    - embeddings and texts MUST be aligned (same ordering)
    - vectors are normalized for cosine similarity search
    """
    global index, metadata

    vectors = np.array(embeddings).astype("float32")
    # normalize for cosine similarity
    faiss.normalize_L2(vectors)
    index.add(vectors)

    metadata.extend(texts)


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
    Performs similarity search over FAISS index.

    Returns:
    - Top-K most similar text chunks

    Note:
    - Currently returns raw text only
    - Future improvement: return structured metadata + citations
    - Threshold filtering is currently disabled for stability
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
            text = metadata[idx]
            results.append(text)
            seen.add(text)

    return results