from fastapi import FastAPI
from backend.routes.chat import router as chat_router
from backend.services.ingestion_service import load_and_chunk_documents, load_documents, chunk_text
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

# test the chunking system
@app.get("/test-ingestion")
def test_ingestion():
    chunks = load_and_chunk_documents()
    return {"chunks": chunks}

@app.get("/build-index")
def build_index():
    """
    Reads documents, chunks them, creates embeddings,
    and stores them in FAISS.
    """

    # Step 1: Load raw text
    documents = load_documents()

    # Step 2: Chunk text
    chunks = chunk_text(documents)

    # Step 3: Generate embeddings
    embeddings = [get_embedding(chunk) for chunk in chunks]

    # Step 4: Create FAISS index
    dimension = len(embeddings[0])
    create_faiss_index(dimension)

    # Step 5: Add data
    add_embeddings(embeddings, chunks)

    # Step 6: Save to disk
    save_index()

    return {"message": f"Indexed {len(chunks)} chunks"}

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

