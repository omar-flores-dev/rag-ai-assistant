from fastapi import FastAPI
from backend.routes.chat import router as chat_router
from backend.services.ingestion_service import load_and_chunk_documents

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
