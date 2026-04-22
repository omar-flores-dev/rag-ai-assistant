from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.llm_service import get_llm_response
from backend.services.embedding_service import get_embedding
from backend.services.vector_store import search_vector_store

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def chat(req: ChatRequest):
    # Step 1: Convert user query into embedding
    query_embedding = get_embedding(req.message)
    # Step 2: Retrieve relevant chunks
    relevant_chunks = search_vector_store(query_embedding)
    # Step 3: Generate response using context
    reply = get_llm_response(req.message, relevant_chunks)

    return {
        "response": reply,
        "context_used": relevant_chunks  # helpful for debugging
    }