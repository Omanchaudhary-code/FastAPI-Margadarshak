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
def generate_recommendations(data, predicted_cgpa: float) -> list:
    recs = []

    # Define motivational quotes
    quotes = {
        "attendance": "80% of success is showing up. – Woody Allen",
        "repeated_course": "Failure is simply the opportunity to begin again, this time more intelligently. – Henry Ford",
        "part_time_job": "Balance is not something you find, it’s something you create. – Jana Kingsford",
        "motivation": "Start where you are. Use what you have. Do what you can. – Arthur Ashe",
        "friends": "Surround yourself with those who lift you higher. – Oprah Winfrey",
        "high_performance": "Excellence is not a skill, it’s an attitude. – Ralph Marston",
        "improve": "Progress, not perfection. – Unknown",
        "risk": "Small steps every day lead to big results. – Unknown",
        "critical": "Every setback is a setup for a comeback. – Willie Jolley"
    }

    # Conditional recommendations with quotes
    if data.attendance < 70:
        recs.append("Try to attend more classes to stay on track academically.")
        recs.append(quotes["attendance"])

    if data.repeated_course == 1:
        recs.append("Focus on understanding the subjects you've repeated for better mastery.")
        recs.append(quotes["repeated_course"])

    if data.part_time_job == 1 and predicted_cgpa < 2.5:
        recs.append("Consider adjusting your part-time work hours to reduce academic stress.")
        recs.append(quotes["part_time_job"])

    if data.motivation_level < 5:
        recs.append("Boost your motivation by setting short-term goals and tracking your progress.")
        recs.append(quotes["motivation"])

    if data.friends_performance < 2.5:
        recs.append("Engage with peers who are academically focused to stay motivated.")
        recs.append(quotes["friends"])

    # Overall CGPA feedback
    if predicted_cgpa >= 3.5:
        recs.append("Great work! Keep up the consistent performance.")
        recs.append(quotes["high_performance"])
    elif predicted_cgpa >= 2.75:
        recs.append("You're passing, but there’s room to improve further.")
        recs.append(quotes["improve"])
    elif predicted_cgpa >= 2.5:
        recs.append("You're at risk. Focus on key habits to improve your academic standing.")
        recs.append(quotes["risk"])
    else:
        recs.append("Critical risk of not graduating. Seek support and take active steps toward improvement.")
        recs.append(quotes["critical"])

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
