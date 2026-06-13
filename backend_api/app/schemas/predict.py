from pydantic import BaseModel

class PredictRequest(BaseModel) :
    text : str

class PredictResponse(BaseModel) :
    label : str
    confidence : float
    uncertain : bool

