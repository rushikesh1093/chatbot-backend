import os
from pathlib import Path
from typing import List, Literal, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import RAGEngine

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Load env from .env (primary). Fall back to .env.example if .env is absent.
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(BASE_DIR / ".env.example")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-4-maverick")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY_PLACEHOLDER = "your_openrouter_api_key_here"
APP_TITLE = "Agrofarm RAG API"


def get_runtime_openrouter_settings() -> tuple[str, str]:
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL", OPENROUTER_MODEL)
    return api_key, model

language_prompts = {
    "en": "Respond only in English.",
    "hi": "केवल हिंदी में उत्तर दें।",
    "ta": "தமிழில் மட்டும் பதிலளிக்கவும்.",
    "te": "తెలుగులో మాత్రమే సమాధానం ఇవ్వండి.",
    "mr": "फक्त मराठीतच उत्तर द्या.",
    "bh": "भोजपुरी में ही जवाब दीं।",
    "ha": "हरियाणवी में ही जवाब दें।",
    "bn": "শুধুমাত্র বাংলায় উত্তর দিন।",
}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    language: str = Field(default="en")
    history: List[Message] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


class IngestResponse(BaseModel):
    indexed: int
    files: Optional[List[str]] = None
    message: str


app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine(data_dir=DATA_DIR, vector_store_dir=VECTOR_STORE_DIR)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": len(rag_engine.chunks)}


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents() -> IngestResponse:
    result = rag_engine.ingest()
    return IngestResponse(**result)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    api_key, model = get_runtime_openrouter_settings()

    if not api_key or api_key == OPENROUTER_API_KEY_PLACEHOLDER:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is missing on backend.")

    language = req.language if req.language in language_prompts else "en"
    retrieved = rag_engine.retrieve(req.question, top_k=4)

    context = "\n\n".join(
        [f"Source: {item['source']}\nContent: {item['text']}" for item in retrieved]
    )

    system_prompt = (
        "You are an agrofarm product assistant. Use the provided context as primary source of truth. "
        "If context is insufficient, say that clearly and answer cautiously. "
        f"{language_prompts[language]}"
    )

    user_prompt = (
        f"Question:\n{req.question}\n\n"
        f"Relevant context:\n{context if context else 'No relevant context found in knowledge base.'}"
    )

    messages = [{"role": "system", "content": system_prompt}]

    for msg in req.history[-6:]:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 700,
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": APP_TITLE,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        answer = data["choices"][0]["message"]["content"]
        sources = sorted({item["source"] for item in retrieved})
        return ChatResponse(answer=answer, sources=sources)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat pipeline failed: {str(exc)}")
    
    
    


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
