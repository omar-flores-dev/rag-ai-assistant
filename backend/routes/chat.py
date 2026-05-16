from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.llm_service import get_llm_response
from backend.services.embedding_service import get_embedding
from backend.services.faiss_store import search
from backend.services.reranker_service import rerank_chunks

'''FILE OBJECTIVE:
RAG inference pipeline (retrieval + reranking + grounded generation).

Flow:
1. Convert user query into embedding
2. Retrieve candidate chunks from FAISS vector store
3. Rerank chunks for lexical/query relevance
4. Extract chunk text for LLM grounding
5. Return generated answer with chunk-level citations

Design Notes:
- Retrieval uses hybrid chunk embeddings
- FAISS provides semantic recall
- Reranker improves precision before generation
- Each retrieved chunk preserves source traceability
- Citations map answers back to section + document level
'''

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def chat(req: ChatRequest):
    """
    End-to-end RAG inference pipeline:

    Step 1: Embed user query
    Step 2: Retrieve candidate chunks from vector store
    Step 3: Rerank chunks for relevance
    Step 4: Ground LLM response using chunk text
    Step 5: Return answer with citation metadata

    Returns:
    - response: grounded LLM answer
    - citations: chunk-level source attribution
    """
    # Step 1: Convert user query into embedding
    query_embedding = get_embedding(req.message)
    # Step 2: Retrieve broader candidate pool
    candidate_chunks = search(
        query_embedding,
        candidate_k=10
    )

    # Step 3: Precision reranking
    relevant_chunks = rerank_chunks(
        req.message,
        candidate_chunks,
        top_k=3
    )
    context_texts = [chunk["text"] for chunk in relevant_chunks]
    reply = get_llm_response(req.message, context_texts)

    return {
        "response": reply,
        "citations": 
        [
            {
            "source": chunk["source"],
            "section": chunk["section"],
            "chunk_id": chunk["chunk_id"],
            "title": chunk["title"],
            "url": chunk["url"],
            "authors": chunk["authors"],
            "year": chunk["year"]
            }
        for chunk in relevant_chunks
        ]
    }