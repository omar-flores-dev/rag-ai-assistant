from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.llm_service import get_llm_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def chat(req: ChatRequest):
    reply = get_llm_response(req.message)
    return {"response": reply}