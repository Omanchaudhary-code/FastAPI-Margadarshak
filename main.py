from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("mlr_model.pkl")

app = FastAPI()

class InputData(BaseModel):
    attendance: float
    friends_performance: float
    motivation_level: float
    part_time_job: int  # 0 or 1

@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array([[data.attendance, data.friends_performance, data.motivation_level, data.part_time_job]])
        prediction = model.predict(features)[0]
        return {"predicted_cgpa": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
