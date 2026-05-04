from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.llm_service import get_llm_response
from backend.services.embedding_service import get_embedding
from backend.services.faiss_store import search

'''FILE OBJECTIVE:
RAG query execution pipeline (retrieval + generation).

Flow:
1. Convert user query → embedding
2. Retrieve relevant context from FAISS
3. Send query + context to LLM
4. Return generated response + retrieved context (for debugging/citations)
'''


router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def chat(req: ChatRequest):
    """
    End-to-end RAG inference pipeline:

    Step 1: Embed user query
    Step 2: Retrieve relevant document context
    Step 3: Generate response using LLM grounded in retrieved context

    Returns:
    - response: LLM-generated answer
    - context_used: retrieved chunks for debugging + future citation system
    """
    # Step 1: Convert user query into embedding
    query_embedding = get_embedding(req.message)
    # Step 2: Retrieve relevant chunks
    relevant_chunks = search(query_embedding, top_k=3)
    # Step 3: Generate response using context
    reply = get_llm_response(req.message, relevant_chunks)

    return {
        "response": reply,
        "context_used": relevant_chunks  # helpful for debugging
    }