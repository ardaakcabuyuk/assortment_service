from fastapi import FastAPI
from assortment import get_suggested_assortments

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/assortment/{musteri_no}/{obs_month}")
async def get_assortment(musteri_no: int, obs_month: str):
    return get_suggested_assortments(int(musteri_no), str(obs_month))