import numpy as np

# In-memory storage
vector_store = []

def add_to_vector_store(chunks, embedding_function):
    """
    Takes text chunks and stores them with their embeddings.
    Each entry = {
        "text": chunk,
        "embedding": vector
    }
    """
    for chunk in chunks:
        embedding = embedding_function(chunk)

        vector_store.append({
            "text": chunk,
            "embedding": embedding
        })


def cosine_similarity(vec1, vec2):
    """
    Measures similarity between two vectors.
    Returns a value between:
    - 1 = very similar (if not the same)
    - 0 = not similar
    This essentially ranks the similarity between vectors -> To be used later for find relevant chunks
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def search_vector_store(query_embedding, top_k=3):
    """
    Finds the most relevant chunks for a query.
    Steps:
    1. Compare query embedding to all stored embeddings
    2. Sort by similarity
    3. Return top matches
    """
    similarities = []

    for item in vector_store:
        sim = cosine_similarity(query_embedding, item["embedding"])
        similarities.append((sim, item["text"]))

    # Sort by highest similarity
    similarities.sort(reverse=True, key=lambda x: x[0])

    # Return top_k results
    return [text for _, text in similarities[:top_k]]