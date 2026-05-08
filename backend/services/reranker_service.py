from sentence_transformers import CrossEncoder

"""
RERANKER SERVICE

Purpose:
Dense retrieval (FAISS) is optimized for recall, not always ranking precision.

This service reranks retrieved candidates by scoring:
(query, chunk_text)

Higher score = more relevant chunk.
"""

# load a pre-trained re-ranker model from HuggingFace
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_chunks(
    query: str,
    chunks: list,
    top_k: int = 3
):
    """
    Reranks retrieved chunks using cross-encoder scoring.

    Args:
        query: User question
        chunks: Retrieved chunk dictionaries
        top_k: Number of chunks to keep

    Returns:
        Best-ranked chunks
    """

    # If FAISS didn’t find anything, stop early, no re-ranking is needed
    if not chunks:
        return []

    # Build query/chunk pairs => this enables the model to directly compare user query with selected chunks
    pairs = [
        (query, chunk["text"])
        for chunk in chunks
    ]

    # each query/chunk pair gets scored where a higher score means it is more relevant to the query.
    scores = reranker.predict(pairs)

    scored_chunks = list(
        zip(scores, chunks)
    )

    # essentially sort by best score
    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # return the top_k (i.e top 3) scored chunks
    reranked = [
        chunk
        for score, chunk
        in scored_chunks[:top_k]
    ]

    return reranked