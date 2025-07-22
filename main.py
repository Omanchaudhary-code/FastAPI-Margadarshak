from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import cloudpickle
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Debug: List available models (fixed, no attribute error)
available_models = genai.list_models()
print("Available models:")
for m in available_models:
    print(f"- {m.name}")

# Load your prediction model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

try:
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Error loading model.pkl: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://margadarshak.tech",
        "https://www.margadarshak.tech",
        "https://preview--journey-forecast-tool.lovable.app",
        "https://journey-forecast-tool.lovable.app",
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

class InputData(BaseModel):
    repeated_course: int
    attendance: float
    part_time_job: int
    motivation_level: float
    first_generation: int
    friends_performance: float

def generate_recommendations(data, predicted_cgpa: float) -> list:
    recs = []
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

# TODO: Update this after checking printed available models above
VALID_GEMINI_MODEL = "models/text-bison-001"  # Replace with valid model name from list_models output

@app.post("/gemini/recommend")
def gemini_recommend(data: InputData):
    try:
        predicted_features = np.array([[
            data.repeated_course,
            data.attendance,
            data.part_time_job,
            data.motivation_level,
            data.first_generation,
            data.friends_performance
        ]])
        predicted_cgpa = model.predict(predicted_features)[0]
        predicted_cgpa = round(max(0.0, min(predicted_cgpa, 4.0)), 2)

        model_gemini = genai.GenerativeModel(VALID_GEMINI_MODEL)
        prompt = f"""
        Provide a short, practical and motivational academic recommendation for a student with:
        - CGPA: {predicted_cgpa}
        - Attendance: {data.attendance}%
        - Repeated Course: {"Yes" if data.repeated_course else "No"}
        - Part-time Job: {"Yes" if data.part_time_job else "No"}
        - Motivation Level: {data.motivation_level}/10
        - First Generation Student: {"Yes" if data.first_generation else "No"}
        - Peer Performance: {data.friends_performance}/4

        Focus on encouragement and actionable next steps.
        """
        response = model_gemini.generate_content(prompt)
        return {"cgpa": predicted_cgpa, "gemini_advice": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Error: {e}")

@app.post("/gemini/chat")
async def gemini_chat(request: Request):
    try:
        body = await request.json()
        user_message = body.get("message")

        chat_model = genai.GenerativeModel(VALID_GEMINI_MODEL)
        chat = chat_model.start_chat(history=[])
        reply = chat.send_message(user_message)

        return {"response": reply.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Chat Error: {e}")
