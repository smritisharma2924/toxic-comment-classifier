from fastapi import APIRouter
from app.schemas.predict import PredictRequest, PredictResponse

from app.state import model_store
from app.services.inference import predict as run_inference

router = APIRouter()

@router.post("/predict", response_model = PredictResponse)
def predict(request : PredictRequest) :
    
    prediction = run_inference(request.text, model_store['model'], model_store['tokenizer'])

    return prediction # or we can write return PredictResponse(label=prediction["label"], confidence=prediction["confidence"], uncertain=prediction["uncertain"] )
                      # but our run_inferance fxn (predict fxn of inference.py fxn already return a dictionary)