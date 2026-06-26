"""
Verdade — Urdu Fake-News Detection API
Entry point: app factory + lifespan (model loading).
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS, DEVICE, HF_MODEL_NAME, HOST, PORT
from app.models.loader import load_model
from app.routers import history, predict
from app.services import inference as inference_svc

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger("verdade")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up…")
    inference_svc.tokenizer, inference_svc.model = load_model()
    log.info("✅  Model ready on device: %s", DEVICE)
    yield
    log.info("Shutting down.")


app = FastAPI(title="Verdade API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(history.router)


@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "model": HF_MODEL_NAME, "device": str(DEVICE)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True
    )
