# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import os
import cloudpickle
from dotenv import load_dotenv
import random

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
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# -----------------------------
# Encodings
# -----------------------------
ENC = {
    "attendance": {">90%": 4, "75-90%": 3, "50-75%": 2, "<75%": 1},

    "program": {
        "SOM": 1, "SOS": 2, "SOE": 3, "SOL": 4, "SMS": 5, "SOA": 6, "SOED": 7
    },

    "year": {
        "1st year": 1, "2nd year": 2, "3rd year": 3, "4th year": 4, "5th year": 5, "Graduated": 6
    },

    "living_situation": {
        "At home": 1, "Hostel": 2, "Rent": 3, "Alone": 4, "Other": 5
    },

    "repeated_course": {"Yes": 1, "No": 0},

    "study_hours": {
        "<1": 1, "1-2": 2, "2-3": 3, "3-4": 4, ">4": 5
    },

    "agreement_1to5": {  # revision, rating, friend_circle, confidence, motivation
        "Strongly disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly agree": 5
    },

    "rating": {  # faculty academic teaching capability
        "Worst": 1, "Bad": 2, "Neutral": 3, "Good": 4, "Best": 5
    },

    "freq_0to3": {  # online, group_studies, help_with_teachers (mapped below), family_support alt uses 0..3 too
        "Never": 0, "Rarely": 1, "Sometimes": 2, "Frequently": 3
    },

    "help_with_teachers": {
        "No": 0, "Rarely": 1, "Sometimes": 2, "Yes, regularly": 3
    },

    "stress_level": {
        "Not at all": 1, "Somewhat": 2, "Neutral": 3, "Moderate": 4, "Extremely": 5
    },

    "sleep": {
        "<5": 1, "5-6": 2, "6-7": 3, "7-8": 4, ">8": 5
    },

    "family_support": {
        "Never": 0, "Rarely": 1, "Sometimes": 2, "Always": 3
    },

    "friend_circle": {  # same as rating
        "Worst": 1, "Bad": 2, "Neutral": 3, "Good": 4, "Best": 5
    },

    # Auto-filled pools
    "income": [1, 2, 3, 4],  # <15k, 15-30k, 30-50k, 50k+
    "first_gen": [0, 1],
    "part_time_job": [0, 1],
    "financial_pressure": [1, 2, 3, 4, 5],
    "family_responsibilities": [0, 1, 2, 3],
    "confidence": [1, 2, 3, 4, 5],
    "motivation": [1, 2, 3, 4, 5],
}

# Weights for realistic randomness
WEIGHTS = {
    "income": [0.20, 0.35, 0.30, 0.15],
    "first_gen": [0.6, 0.4],
    "part_time_job": [0.65, 0.35],
    "financial_pressure": [0.15, 0.25, 0.30, 0.20, 0.10],
    "family_responsibilities": [0.25, 0.35, 0.25, 0.15],
    "confidence": [0.10, 0.20, 0.30, 0.25, 0.15],
    "motivation": [0.08, 0.17, 0.30, 0.28, 0.17],
}

# Final feature order expected by the model (23 features)
FEATURE_ORDER = [
    "attendance", "program", "year", "living_situation", "repeated_course",
    "study_hours", "revision", "rating", "online", "group_studies",
    "help_with_teachers", "stress_level", "sleep", "family_support",
    "friend_circle", "income", "first_gen", "part_time_job",
    "financial_pressure", "family_responsibilities", "confidence",
    "motivation", "engagement_index"
]

# -----------------------------
# Schemas (15 user-facing fields)
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

    study = norm(encoded["study_hours"], 1, 5)
    revision = norm(encoded["revision"], 1, 5)
    rating = norm(encoded["rating"], 1, 5)
    online = norm(encoded["online"], 0, 3)
    group = norm(encoded["group_studies"], 0, 3)
    help_t = norm(encoded["help_with_teachers"], 0, 3)
    conf = norm(encoded.get("confidence", 3), 1, 5)
    motiv = norm(encoded.get("motivation", 3), 1, 5)
    support = norm(encoded["family_support"], 0, 3)

    vals = np.array([study, revision, rating, online, group, help_t, conf, motiv, support], dtype=float)
    weights = np.array([1.2, 1.1, 0.8, 0.6, 0.6, 0.8, 1.0, 1.1, 0.8], dtype=float)
    return float(np.clip(np.average(vals, weights=weights), 0.0, 1.0))

def build_feature_vector(user_in: InputData) -> pd.DataFrame:
    enc15 = encode_user_inputs(user_in)
    auto7 = autofill_features()
    merged = {**enc15, **auto7}
    merged["engagement_index"] = compute_engagement_index({**enc15, **auto7})
    vector = {name: merged[name] for name in FEATURE_ORDER}
    return pd.DataFrame([vector])  # <- ensures column names for sklearn

def generate_general_recommendations(data: InputData) -> str:
    recs = []

    if data.attendance in ["<75%", "50-75%"]:
        recs.append("<b>Boost Attendance</b><br/>Try to attend most classes; consistent presence helps learning.")
    else:
        recs.append("<b>Maintain Attendance</b><br/>Good job attending classes regularly, keep up the consistency.")

    if data.study_hours in ["<1", "1-2"]:
        recs.append("<b>Increase Study Hours</b><br/>Allocate focused study time daily to strengthen understanding of subjects.")
    else:
        recs.append("<b>Keep Study Routine</b><br/>Your current study schedule is solid, continue with regular practice.")

    if data.repeated_course == "Yes":
        recs.append("<b>Handle Repeated Courses</b><br/>Focus on mastering difficult subjects to avoid repeating courses in the future.")
    else:
        recs.append("<b>Course Progress</b><br/>You are progressing well, continue reviewing material regularly.")

    return "<br/><br/>".join(recs)

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the CGPA Predictor API. Visit /docs to test the prediction endpoint.",
        "expects_features": len(FEATURE_ORDER),
        "user_facets": 15,
        "auto_filled": 8,
        "synthetic": "engagement_index",
    }

@app.post("/predict")
async def predict(data: InputData):
    try:
        features = build_feature_vector(data)   # pd.DataFrame with column names
        prediction = float(model.predict(features)[0])
        prediction = max(0.0, min(prediction, 4.0))
        prediction = round(prediction, 2)

        recommendations_html = generate_general_recommendations(data)

        return {
            "predicted_cgpa": prediction,
            "recommendations_html": recommendations_html,
            "model": "Gradient Boosting v1.1 (15+auto)"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
