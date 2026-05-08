from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.llm_service import get_llm_response
from backend.services.embedding_service import get_embedding
from backend.services.faiss_store import search

'''FILE OBJECTIVE:
RAG inference pipeline (retrieval + grounded generation).

Flow:
1. Convert user query into embedding
2. Retrieve semantically relevant chunks from FAISS
3. Extract chunk text for LLM grounding
4. Return generated answer with chunk-level citations

Design Notes:
- Retrieval uses hybrid chunk embeddings
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
    Step 2: Retrieve relevant chunks from vector store
    Step 3: Ground LLM response using retrieved chunk text
    Step 4: Return answer with citation metadata

    Returns:
    - response: grounded LLM answer
    - citations: chunk-level source attribution
    """
    # Step 1: Convert user query into embedding
    query_embedding = get_embedding(req.message)
    # Step 2: Retrieve relevant chunks
    relevant_chunks = search(query_embedding, top_k=3)
    # Step 3: Generate response using context
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