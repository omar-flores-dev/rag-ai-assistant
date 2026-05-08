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
    5. Persist chunk metadata for runtime retrieval + citation tracing

    PIPELINE DESIGN:
    - Ingestion layer provides structured sections with rich metadata
    - Chunking layer splits sections into retrieval-optimized chunks
    - Embedding layer maps each chunk into vector space
    - Vector store enables fast semantic retrieval
    - Metadata layer preserves chunk → section → document traceability

    CURRENT CAPABILITIES:
    - Hybrid chunking implemented
    - Citation-aware chunk metadata stored with every embedding
    - Retrieval returns both chunk text and source metadata

    FUTURE IMPROVEMENTS:
    - Add metadata-aware filtering (source_type, year, authors)
    - Implement MMR (Maximal Marginal Relevance)
    - Add cross-encoder reranking
    """

    structured_docs = load_documents_structured()

    embeddings = [get_embedding(chunk["text"]) for chunk in structured_docs]

    # FAISS expects fixed dimensional vectors
    dimension = len(embeddings[0])

    create_faiss_index(dimension)

    add_embeddings(embeddings, structured_docs)

    save_index()

    return {
    "message": f"Indexed {len(structured_docs)} chunks",
    "source_types": list(set(
        chunk["source_type"]
        for chunk in structured_docs
        ))
    }

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

