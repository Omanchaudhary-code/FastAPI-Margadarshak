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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Production URLs
        "https://margadarshak.tech",
        "https://www.margadarshak.tech",
        "https://preview--journey-forecast-tool.lovable.app",
        "https://journey-forecast-tool.lovable.app",
        
        # Development URLs
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080",
        
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the CGPA Predictor API 👋. Visit /docs to test the prediction endpoint."
    }

# ✅ Input schema
class InputData(BaseModel):
    repeated_course: int
    attendance: float
    part_time_job: int
    motivation_level: float
    first_generation: int
    friends_performance: float

# ✅ Recommendation logic
def generate_recommendations(data: InputData, predicted_cgpa: float) -> list:
    recs = []

    if data.attendance < 70:
        recs.append("✅ Try to attend more classes to stay on track academically.")
    
    if data.repeated_course == 1:
        recs.append("📘 Focus on understanding the subjects you've repeated for better mastery.")
    
    if data.part_time_job == 1 and predicted_cgpa < 2.5:
        recs.append("🕒 Consider adjusting your part-time work hours to reduce academic stress.")
    
    if data.motivation_level < 5:
        recs.append("🔥 Boost your motivation by setting short-term goals and tracking progress.")
    
    if data.friends_performance < 2.5:
        recs.append("👥 Surround yourself with academically focused peers to stay inspired.")

    if predicted_cgpa >= 3.5:
        recs.append("🎉 Great work! Keep up the consistent performance.")

    if not recs:
        recs.append("👍 You're doing well — keep going!")

    return recs

# ✅ Prediction endpoint with recommendation
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

        prediction = max(0.0, min(prediction, 4.0))
        prediction = round(prediction, 2)

        recommendations = generate_recommendations(data, prediction)

        return {
            "predicted_cgpa": prediction,
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
