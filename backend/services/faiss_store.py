import faiss
import os
import json
import numpy as np

"""
This file essentially does:
- Create a FAISS index (store/search vectors)
- Creates the metadata list (text chunks with the associated FAISS index ids)
- Has save/load functions (persistence for the index and associated metadata)
- Has search function (leverages FAISS to essentially return the most similar text chunks from the top-n similar vectors)
"""

# Paths for persistence
INDEX_PATH = "backend/data/vector_store/index.faiss"
METADATA_PATH = "backend/data/vector_store/metadata.json"

# Global variables
index = None
metadata = []


def create_faiss_index(dimension: int):
    """
    Create a FAISS index using L2 distance (Euclidean Distance).
    """
    global index
    index = faiss.IndexFlatL2(dimension)


def add_embeddings(embeddings: list, texts: list):
    """
    Add embeddings + corresponding text to FAISS + metadata.
    """
    global index, metadata

    vectors = np.array(embeddings).astype("float32")
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
def search(query_embedding: list, top_k: int = 5, threshold: float = 1.5):
    """
    Perform similarity search using FAISS.
    Returns top_k matching text chunks. (Similar to a KNN alogrithim)
    """
    global index, metadata

    if index is None or index.ntotal == 0:
        return []

    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    results = []
    seen = set() # want to avoid using the same chunks
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            distance = distances[0][i] # want to be able to filter out weaker matches
            text = metadata[idx]

            if distance < threshold and text not in seen:
                results.append(text)
                seen.add(text)

    return results