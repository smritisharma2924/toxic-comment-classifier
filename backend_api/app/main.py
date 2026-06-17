from fastapi import FastAPI
from app.routers.predict import router
from contextlib import asynccontextmanager
from app.services.model_loader import load_model

from app.state import model_store # model_store = {}

from app.db.logger import init_db

from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app : FastAPI) :
    model, tokenizer = load_model()
    model_store['model'] = model
    model_store['tokenizer'] = tokenizer
    await init_db() # inside lifespan, after model_store lines:
    yield

app = FastAPI(lifespan = lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def sample() :
    return {"status" : "ok"}