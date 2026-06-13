from fastapi import FastAPI
from app.routers.predict import router

app = FastAPI()
app.include_router(router)

@app.get("/")
def sample() :
    return {"status" : "ok"}