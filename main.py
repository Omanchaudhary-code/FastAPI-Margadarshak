from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import cloudpickle
from dotenv import load_dotenv
import httpx

# Load environment variables from .env file
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is missing")
print("OPENROUTER_API_KEY loaded")

# Load your prediction model using cloudpickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

try:
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model.pkl: {e}")

app = FastAPI(title="CGPA Predictor API")

# Configure CORS middleware
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
        "message": "Welcome to the CGPA Predictor API. Visit /docs to test the prediction endpoint."
    }


class InputData(BaseModel):
    repeated_course: int
    attendance: float
    part_time_job: int
    motivation_level: float
    first_generation: int
    friends_performance: float


async def generate_claude_recommendations(data: InputData, predicted_cgpa: float) -> str:
    prompt = f"""
You are an encouraging academic coach helping university students improve their academic performance.

Here is a student's profile:
- Predicted CGPA: {predicted_cgpa:.2f}
- Attendance: {data.attendance}%
- Repeated a Course: {'Yes' if data.repeated_course else 'No'}
- Has a Part-time Job: {'Yes' if data.part_time_job else 'No'}
- Motivation Level (out of 10): {data.motivation_level}
- First-generation College Student: {'Yes' if data.first_generation else 'No'}
- Peer Performance (Average CGPA of Friends on a 4.0 Scale): {data.friends_performance}/4

Write 3 short, motivational recommendations tailored to this student's academic challenges and strengths.

Instructions:
- Begin each recommendation with a bold, engaging title (e.g., Strong Start Strategy)
- Follow with 1–2 concise sentences offering practical, supportive advice
- Keep each recommendation under 40 words
- Do NOT use emojis, slang, or bullet points
- Use a warm, encouraging, and professional tone focused on growth and progress
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Referer": "https://margadarshak.tech",
        "X-Title": "Margadarshak Recommendations"
    }

    payload = {
        "model": "anthropic/claude-3.5-haiku",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error from OpenRouter API: {response.text}"
            )

        try:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except (KeyError, ValueError) as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid response structure from Claude: {e} - Response text: {response.text}"
            )


@app.post("/predict")
async def predict(data: InputData):
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

        # Clamp prediction between 0 and 4 and round to 2 decimals
        prediction = max(0.0, min(prediction, 4.0))
        prediction = round(prediction, 2)

        recommendations = await generate_claude_recommendations(data, prediction)

        return {
            "predicted_cgpa": prediction,
            "recommendations": recommendations,
            "model": "Monte Carlo v1.0"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
