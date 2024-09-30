from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.embedding import jina_embedding_model

router = APIRouter()


class EmbeddingRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embedding: list[float]


@router.post("/jina/embed", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    try:
        embedding = jina_embedding_model.encode([request.text])[0].tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
