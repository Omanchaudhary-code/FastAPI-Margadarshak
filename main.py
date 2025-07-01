from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import cloudpickle

# Load model using cloudpickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

try:
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Error loading model.pkl: {e}")

# Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS (allow frontend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain for production, e.g. ["https://yourfrontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional root route for testing
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the CGPA Predictor API 👋. Visit /docs to test the prediction endpoint."
    }

# ✅ Define input schema
class InputData(BaseModel):
    repeated_course: int
    attendance: float
    part_time_job: int
    motivation_level: float
    first_generation: int
    friends_performance: float

# ✅ Prediction endpoint
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
