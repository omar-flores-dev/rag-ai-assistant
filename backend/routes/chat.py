from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.llm_service import get_llm_response
from backend.services.embedding_service import get_embedding
from backend.services.faiss_store import search

'''FILE OBJECTIVE: How results are used in the LLM and API'''


router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def chat(req: ChatRequest):
    # Step 1: Convert user query into embedding
    query_embedding = get_embedding(req.message)
    # Step 2: Retrieve relevant chunks
    relevant_chunks = search(query_embedding, top_k=5)
    relevant_chunks = relevant_chunks[:3] # Grab the top 
    # Step 3: Generate response using context
    reply = get_llm_response(req.message, relevant_chunks)

    return {
        "response": reply,
        "context_used": relevant_chunks  # helpful for debugging
    }