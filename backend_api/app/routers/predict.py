from fastapi import APIRouter
from app.schemas.predict import PredictRequest, PredictResponse

from app.state import model_store
from app.services.inference import predict as run_inference

from fastapi import BackgroundTasks
from app.db.logger import log_request

router = APIRouter()

@router.post("/predict", response_model = PredictResponse)
def predict(request : PredictRequest, background_tasks: BackgroundTasks) :
    
    prediction = run_inference(request.text, model_store['model'], model_store['tokenizer'])
    background_tasks.add_task(log_request, request.text, prediction['label'], prediction['confidence'], prediction['uncertain'])

    return prediction # or we can write return PredictResponse(label=prediction["label"], confidence=prediction["confidence"], uncertain=prediction["uncertain"] )
                      # but our run_inferance fxn (predict fxn of inference.py fxn already return a dictionary)