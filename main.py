# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(model_path)

# Initialize FastAPI app
app = FastAPI()

# Define expected input structure
class InputData(BaseModel):
    repeated_course: int       # 0 or 1
    attendance: float          # e.g., 90
    part_time_job: int         # 0 or 1
    motivation_level: float    # 1 to 10
    first_generation: int      # 0 or 1
    friends_performance: float # 1 to 10

# Endpoint for prediction
@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array([[
            data.repeated_course,
            data.attendance,
            data.part_time_job,
            data.motivation_level,
            data.first_generation,
            data.friends_performance
        ]])
        prediction = model.predict(features)[0]
        return {"predicted_cgpa": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
