from fastapi import FastAPI
from backend.routes.chat import router as chat_router
from backend.services.ingestion_service import load_documents_structured
from backend.services.faiss_store import load_index
from backend.services.faiss_store import (
    create_faiss_index,
    add_embeddings,
    save_index
)
from backend.services.embedding_service import get_embedding
from backend.services.vector_store import add_to_vector_store, search_vector_store
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.include_router(chat_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "API running"}

# test the ingestion system
@app.get("/test-ingestion")
def test_ingestion():
    """
    Debug endpoint for verifying structured ingestion pipeline.

    Returns:
    - number of structured document sections loaded
    - sample document for inspection
    """
    structured_docs = load_documents_structured()

    return {
        "num_docs": len(structured_docs),
        "sample": structured_docs[0] if structured_docs else None
    }

@app.get("/build-index")
def build_index():
    """
    End-to-end indexing pipeline:

    1. Load structured documents (multi-source, section-aware ingestion)
    2. Apply hybrid chunking (section → overlapping retrieval chunks)
    3. Convert each chunk into embeddings
    4. Store embeddings in FAISS vector index
    5. Persist index and metadata for runtime retrieval

    PIPELINE DESIGN:
    - Ingestion layer provides structured sections with full metadata
    - Chunking layer splits sections into smaller retrieval-optimized chunks
    - Embedding layer maps each chunk into vector space
    - Vector store enables fast semantic search via FAISS

    NOTE:
    - Hybrid chunking is now implemented (section + sub-section retrieval chunks)
    - Each embedding represents a retrieval-level chunk, not a full section
    - This improves retrieval granularity and context relevance in RAG responses

    FUTURE IMPROVEMENTS:
    - Add metadata-aware retrieval filtering (source_type, year, authors)
    - Implement MMR (Maximal Marginal Relevance) for diversity in results
    - Add citation tracing (chunk → section → document mapping)
    """

    structured_docs = load_documents_structured()

    # Extract raw text from structured documents
    texts = [d["text"] for d in structured_docs]

    embeddings = [get_embedding(text) for text in texts]

    # FAISS expects fixed dimensional vectors
    dimension = len(embeddings[0])

    create_faiss_index(dimension)

    add_embeddings(embeddings, texts)

    save_index()

    return {"message": f"Indexed {len(texts)} document sections"}

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup_event():
    load_index()

@app.post("/search")
def search(req: QueryRequest):
    query_embedding = get_embedding(req.query)

    results = search_vector_store(query_embedding)

    return {"results": results}

