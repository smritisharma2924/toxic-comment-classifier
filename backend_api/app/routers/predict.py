from fastapi import APIRouter
from app.schemas.predict import PredictRequest, PredictResponse

router = APIRouter()

@router.post("/predict", response_model = PredictResponse)
def predict(request : PredictRequest) :
    return PredictResponse(
        label = "insult",
        confidence = 0.93,
        uncertain = False
    )