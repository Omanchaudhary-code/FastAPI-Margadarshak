

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
import cloudpickle
from dotenv import load_dotenv
import random
import requests

# -----------------------------
# Environment & Model Loading
# -----------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

try:
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model.pkl: {e}")

# OpenRouter API Key (from .env)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

app = FastAPI(title="CGPA Predictor API")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://margadarshak.tech",
        "https://www.margadarshak.tech",
        "https://margapradarshak.netlify.app",
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
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Encodings
# -----------------------------
ENC = {
    "attendance": {">90%": 4, "75-90%": 3, "50-75%": 2, "<75%": 1},
    "program": {"SOM": 1, "SOS": 2, "SOE": 3, "SOL": 4, "SMS": 5, "SOA": 6, "SOED": 7},
    "year": {"1st year": 1, "2nd year": 2, "3rd year": 3, "4th year": 4, "5th year": 5, "Graduated": 6},
    "living_situation": {"At home": 1, "Hostel": 2, "Rent": 3, "Alone": 4, "Other": 5},
    "repeated_course": {"Yes": 1, "No": 0},
    "study_hours": {"<1": 1, "1-2": 2, "2-3": 3, "3-4": 4, ">4": 5},
    "agreement_1to5": {
        "Strongly disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly agree": 5
    },
    "rating": {"Worst": 1, "Bad": 2, "Neutral": 3, "Good": 4, "Best": 5},
    "freq_0to3": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Frequently": 3},
    "help_with_teachers": {"No": 0, "Rarely": 1, "Sometimes": 2, "Yes, regularly": 3},
    "stress_level": {"Not at all": 1, "Somewhat": 2, "Neutral": 3, "Moderate": 4, "Extremely": 5},
    "sleep": {"<5": 1, "5-6": 2, "6-7": 3, "7-8": 4, ">8": 5},
    "family_support": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Always": 3},
    "friend_circle": {"Worst": 1, "Bad": 2, "Neutral": 3, "Good": 4, "Best": 5},
    "income": [1, 2, 3, 4],
    "first_gen": [0, 1],
    "part_time_job": [0, 1],
    "financial_pressure": [1, 2, 3, 4, 5],
    "family_responsibilities": [0, 1, 2, 3],
    "confidence": [1, 2, 3, 4, 5],
    "motivation": [1, 2, 3, 4, 5],
}

WEIGHTS = {
    "income": [0.20, 0.35, 0.30, 0.15],
    "first_gen": [0.6, 0.4],
    "part_time_job": [0.65, 0.35],
    "financial_pressure": [0.15, 0.25, 0.30, 0.20, 0.10],
    "family_responsibilities": [0.25, 0.35, 0.25, 0.15],
    "confidence": [0.10, 0.20, 0.30, 0.25, 0.15],
    "motivation": [0.08, 0.17, 0.30, 0.28, 0.17],
}

FEATURE_ORDER = [
    "attendance", "program", "year", "living_situation", "repeated_course",
    "study_hours", "revision", "rating", "online", "group_studies",
    "help_with_teachers", "stress_level", "sleep", "family_support",
    "friend_circle", "income", "first_gen", "part_time_job",
    "financial_pressure", "family_responsibilities", "confidence",
    "motivation", "engagement_index"
]

# -----------------------------
# Schemas
# -----------------------------
class InputData(BaseModel):
    attendance: str
    program: str
    year: str
    living_situation: str
    repeated_course: str
    study_hours: str
    revision: str
    rating: str
    online: str
    group_studies: str
    help_with_teachers: str
    stress_level: str
    sleep: str
    family_support: str
    friend_circle: str

# -----------------------------
# Helpers
# -----------------------------
def _enc(map_name: str, raw: str) -> int:
    mapping = ENC[map_name]
    if raw not in mapping:
        raise HTTPException(status_code=422, detail=f"Invalid value for {map_name}: {raw}")
    return mapping[raw]

def _choice(values, weights=None):
    if weights is None:
        return random.choice(values)
    return np.random.choice(values, p=np.array(weights, dtype=float))

def encode_user_inputs(d: InputData) -> dict:
    return {
        "attendance": _enc("attendance", d.attendance),
        "program": _enc("program", d.program),
        "year": _enc("year", d.year),
        "living_situation": _enc("living_situation", d.living_situation),
        "repeated_course": _enc("repeated_course", d.repeated_course),
        "study_hours": _enc("study_hours", d.study_hours),
        "revision": _enc("agreement_1to5", d.revision),
        "rating": _enc("rating", d.rating),
        "online": _enc("freq_0to3", d.online),
        "group_studies": _enc("freq_0to3", d.group_studies),
        "help_with_teachers": _enc("help_with_teachers", d.help_with_teachers),
        "stress_level": _enc("stress_level", d.stress_level),
        "sleep": _enc("sleep", d.sleep),
        "family_support": _enc("family_support", d.family_support),
        "friend_circle": _enc("friend_circle", d.friend_circle),
    }

def autofill_features() -> dict:
    return {
        "income": int(_choice(ENC["income"], WEIGHTS["income"])),
        "first_gen": int(_choice(ENC["first_gen"], WEIGHTS["first_gen"])),
        "part_time_job": int(_choice(ENC["part_time_job"], WEIGHTS["part_time_job"])),
        "financial_pressure": int(_choice(ENC["financial_pressure"], WEIGHTS["financial_pressure"])),
        "family_responsibilities": int(_choice(ENC["family_responsibilities"], WEIGHTS["family_responsibilities"])),
        "confidence": int(_choice(ENC["confidence"], WEIGHTS["confidence"])),
        "motivation": int(_choice(ENC["motivation"], WEIGHTS["motivation"])),
    }

def compute_engagement_index(encoded: dict) -> float:
    def norm(v, vmin, vmax):
        return (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    vals = np.array([
        norm(encoded["study_hours"], 1, 5),
        norm(encoded["revision"], 1, 5),
        norm(encoded["rating"], 1, 5),
        norm(encoded["online"], 0, 3),
        norm(encoded["group_studies"], 0, 3),
        norm(encoded["help_with_teachers"], 0, 3),
        norm(encoded.get("confidence", 3), 1, 5),
        norm(encoded.get("motivation", 3), 1, 5),
        norm(encoded["family_support"], 0, 3),
    ], dtype=float)
    weights = np.array([1.2, 1.1, 0.8, 0.6, 0.6, 0.8, 1.0, 1.1, 0.8], dtype=float)
    return float(np.clip(np.average(vals, weights=weights), 0.0, 1.0))

def build_feature_vector(user_in: InputData) -> pd.DataFrame:
    enc15 = encode_user_inputs(user_in)
    auto7 = autofill_features()
    merged = {**enc15, **auto7}
    merged["engagement_index"] = compute_engagement_index({**enc15, **auto7})
    vector = {name: merged[name] for name in FEATURE_ORDER}
    return pd.DataFrame([vector])

def generate_general_recommendations(data: InputData) -> str:
    recs = []
    if data.attendance in ["<75%", "50-75%"]:
        recs.append("📌 Boost Attendance: Try to attend more classes consistently.")
    else:
        recs.append("✅ Good Attendance: Keep maintaining consistency.")
    if data.study_hours in ["<1", "1-2"]:
        recs.append("📌 Increase Study Hours: Dedicate more focused time daily.")
    else:
        recs.append("✅ Good Study Routine: Continue with regular practice.")
    if data.repeated_course == "Yes":
        recs.append("📌 Handle Repeated Courses: Focus on difficult subjects.")
    else:
        recs.append("✅ Progressing Well: Keep revising regularly.")
    return "\n".join(recs)

# -----------------------------
# LLM Integration (with Fallback)
# -----------------------------
def get_recommendations(predicted_cgpa: float, data: InputData) -> dict:
    general_recs = generate_general_recommendations(data)

    if not OPENROUTER_API_KEY:
        return {"source": "fallback", "recommendations": general_recs}

    prompt = f"""
    The student's predicted CGPA is {predicted_cgpa}.
    Their inputs are: {data.dict()}.
    Please provide 3-5 personalized, motivating academic recommendations in plain text.
    Focus on practical, supportive, and encouraging advice.
    """

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-mini",  # You can change this model
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            },
            timeout=20
        )
        response.raise_for_status()
        data_json = response.json()
        ai_text = data_json["choices"][0]["message"]["content"].strip()
        return {"source": "llm", "recommendations": ai_text}
    except Exception as e:
        print(f"⚠️ OpenRouter failed: {e}")
        return {"source": "fallback", "recommendations": general_recs}

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the CGPA Predictor API with AI fallback.",
        "features": len(FEATURE_ORDER),
    }

@app.post("/predict")
async def predict(data: InputData):
    try:
        features = build_feature_vector(data)
        prediction = float(model.predict(features)[0])
        prediction = round(max(0.0, min(prediction, 4.0)), 2)

        recommendations = get_recommendations(prediction, data)

        return {
            "predicted_cgpa": prediction,
            "recommendations": recommendations["recommendations"],
            "recommendation_source": recommendations["source"],
            "model": "Gradient Boosting v1.1 + OpenRouter AI"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

