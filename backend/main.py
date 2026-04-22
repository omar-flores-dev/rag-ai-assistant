from fastapi import FastAPI
from backend.routes.chat import router as chat_router
from backend.services.ingestion_service import load_and_chunk_documents
from backend.services.ingestion_service import load_and_chunk_documents
from backend.services.embedding_service import get_embedding
from backend.services.vector_store import add_to_vector_store, search_vector_store
from pydantic import BaseModel


app = FastAPI()
app.include_router(chat_router)

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
    chunks = load_and_chunk_documents()

    add_to_vector_store(chunks, get_embedding)

    return {"message": f"Indexed {len(chunks)} chunks"}

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search(req: QueryRequest):
    query_embedding = get_embedding(req.query)

    results = search_vector_store(query_embedding)

    return {"results": results}

