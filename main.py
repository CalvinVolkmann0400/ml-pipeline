import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
# Loaded once when the app starts, not on every request
model = joblib.load("model.joblib")


class Client(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str


@app.post("/predict")
def predict(client: Client):
    X = pd.DataFrame([client.model_dump()])
    prediction = model.predict(X)[0]
    return {"subscribes": str(prediction)}
