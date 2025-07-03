from fastapi import FastAPI
from pydantic import BaseModel
from src.predictor import predict_email_multi

app = FastAPI()

class EmailRequest(BaseModel):
    email: str
    model: str = "svm"

@app.post("/predict/")
def classify_email(req: EmailRequest):
    try:
        result = predict_email_multi(req.email, model_type=req.model)
        return {
            "model": req.model,
            "email": req.email,
            "results": result
        }
    except Exception as e:
        return {"error": str(e)}
