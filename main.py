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
from datetime import datetime, timezone
from supabase import create_client, Client

# -----------------------------
# Environment Setup
# -----------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("⚠️ Supabase credentials not found in environment variables.")

# -----------------------------
# Model Loading
# -----------------------------
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
        "https://journey-forecast-tool-git-main-margadarshaks-projects.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Encodings & Config
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
    user_id: str
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
        "income": int(_choice([1, 2, 3, 4], WEIGHTS["income"])),
        "first_gen": int(_choice([0, 1], WEIGHTS["first_gen"])),
        "part_time_job": int(_choice([0, 1], WEIGHTS["part_time_job"])),
        "financial_pressure": int(_choice([1, 2, 3, 4, 5], WEIGHTS["financial_pressure"])),
        "family_responsibilities": int(_choice([0, 1, 2, 3], WEIGHTS["family_responsibilities"])),
        "confidence": int(_choice([1, 2, 3, 4, 5], WEIGHTS["confidence"])),
        "motivation": int(_choice([1, 2, 3, 4, 5], WEIGHTS["motivation"])),
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

# -----------------------------
# Supabase Limit Check
# -----------------------------
def check_daily_limit(user_id: str, limit: int = 3):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    res = (
        supabase.table("assessments")
        .select("id")
        .eq("user_id", user_id)
        .gte("created_at", start_of_day.isoformat())
        .execute()
    )
    if len(res.data) >= limit:
        raise HTTPException(status_code=403, detail="Daily limit reached (3 per day).")

# -----------------------------
# Enhanced Recommendation System
# -----------------------------
def generate_comprehensive_recommendations(predicted_cgpa: float, data: InputData) -> str:
    """Generate detailed, personalized recommendations based on all student data."""
    recs = []
    
    # Performance tier assessment
    if predicted_cgpa >= 3.5:
        recs.append("🌟 **Excellent Performance**: You're on track for outstanding results!")
    elif predicted_cgpa >= 3.0:
        recs.append("✅ **Good Performance**: Solid foundation with room for excellence!")
    elif predicted_cgpa >= 2.5:
        recs.append("📈 **Moderate Performance**: Focus on key improvements to boost your CGPA.")
    else:
        recs.append("⚠️ **Needs Attention**: Let's work on strengthening your academic foundation.")
    
    # Attendance-based recommendations
    if data.attendance in ["<75%", "50-75%"]:
        recs.append("📌 **Attendance Alert**: Low attendance significantly impacts learning. Aim for 85%+ attendance to stay engaged with coursework and build better understanding.")
    elif data.attendance == "75-90%":
        recs.append("👍 **Good Attendance**: You're consistent! Push to 90%+ to maximize classroom learning benefits.")
    else:
        recs.append("⭐ **Excellent Attendance**: Keep up this consistency—it's a strong foundation for success.")
    
    # Study hours and habits
    if data.study_hours in ["<1", "1-2"]:
        recs.append("📚 **Increase Study Time**: Allocate 3-4 hours daily for focused study. Quality study time directly correlates with better grades.")
    elif data.study_hours in ["2-3"]:
        recs.append("📖 **Solid Study Routine**: Consider extending to 3-4 hours daily, especially before exams.")
    else:
        recs.append("💪 **Strong Study Commitment**: Excellent dedication! Ensure you're also balancing rest and breaks.")
    
    # Revision habits
    if data.revision in ["Strongly disagree", "Disagree"]:
        recs.append("🔄 **Revision is Key**: Regular revision (weekly/bi-weekly) helps retain concepts better. Create a revision schedule.")
    elif data.revision == "Neutral":
        recs.append("🔄 **Boost Revision**: Make revision a regular habit. Review notes within 24 hours of each class.")
    
    # Course repetition
    if data.repeated_course == "Yes":
        recs.append("🎯 **Focus on Weak Areas**: Identify specific topics in repeated courses. Seek help from professors or tutors early.")
    
    # Learning resources
    if data.online in ["Never", "Rarely"]:
        recs.append("💻 **Leverage Online Resources**: Platforms like Khan Academy, Coursera, or YouTube can supplement your learning effectively.")
    
    if data.group_studies in ["Never", "Rarely"]:
        recs.append("👥 **Consider Group Study**: Collaborative learning helps clarify doubts and exposes you to different perspectives.")
    
    # Teacher interaction
    if data.help_with_teachers in ["No", "Rarely"]:
        recs.append("👨‍🏫 **Engage with Teachers**: Don't hesitate to ask questions. Regular interaction with teachers can significantly improve understanding.")
    
    # Stress management
    if data.stress_level in ["Moderate", "Extremely"]:
        recs.append("🧘 **Manage Stress**: High stress affects performance. Practice meditation, exercise, or talk to a counselor. Mental health is crucial.")
    
    # Sleep patterns
    if data.sleep in ["<5", "5-6"]:
        recs.append("😴 **Prioritize Sleep**: Aim for 7-8 hours. Sleep deprivation reduces focus and memory retention significantly.")
    elif data.sleep in ["6-7"]:
        recs.append("😊 **Good Sleep Pattern**: Try to reach 7-8 hours for optimal cognitive function.")
    else:
        recs.append("✅ **Healthy Sleep**: Great job maintaining good sleep habits!")
    
    # Social support
    if data.family_support in ["Never", "Rarely"]:
        recs.append("💬 **Seek Support**: Talk to family, friends, or campus counselors. A support system is vital for academic success.")
    
    if data.friend_circle in ["Worst", "Bad"]:
        recs.append("🤝 **Build Positive Connections**: Surround yourself with motivated peers. Your social circle influences your academic mindset.")
    
    # Living situation advice
    if data.living_situation in ["Hostel", "Rent", "Alone"]:
        recs.append("🏠 **Living Away from Home**: Create a conducive study environment. Maintain a routine and stay connected with family.")
    
    # Program-specific encouragement
    program_map = {
        "SOM": "Management", "SOS": "Science", "SOE": "Engineering",
        "SOL": "Law", "SMS": "Medical Sciences", "SOA": "Arts", "SOED": "Education"
    }
    program_name = program_map.get(data.program, "your field")
    recs.append(f"🎓 **{program_name} Student**: Focus on core concepts and practical applications in your field. Stay curious and engaged!")
    
    # Year-specific advice
    if data.year in ["1st year", "2nd year"]:
        recs.append("🌱 **Build Strong Foundations**: Early years are crucial. Master fundamentals now for easier advanced courses later.")
    elif data.year in ["3rd year", "4th year"]:
        recs.append("🚀 **Career Preparation**: Balance academics with skill development, internships, and networking for career readiness.")
    
    return "\n\n".join(recs)

def get_recommendations(predicted_cgpa: float, data: InputData) -> dict:
    """Get AI-powered personalized recommendations with fallback to comprehensive general recommendations."""
    
    # Generate comprehensive fallback recommendations
    general_recs = generate_comprehensive_recommendations(predicted_cgpa, data)
    
    if not OPENROUTER_API_KEY:
        print("⚠️ OpenRouter API key not configured, using general recommendations")
        return {"source": "fallback", "recommendations": general_recs}

    # Enhanced prompt for AI recommendations
    prompt = f"""You are an empathetic academic advisor helping a college student improve their performance.

**Student Profile:**
- Predicted CGPA: {predicted_cgpa}/4.0
- Program: {data.program}
- Year: {data.year}
- Attendance: {data.attendance}
- Study Hours: {data.study_hours}
- Living Situation: {data.living_situation}
- Repeated Course: {data.repeated_course}
- Revision Habit: {data.revision}
- Study Resources Rating: {data.rating}
- Online Learning Usage: {data.online}
- Group Study Frequency: {data.group_studies}
- Teacher Interaction: {data.help_with_teachers}
- Stress Level: {data.stress_level}
- Sleep Hours: {data.sleep}
- Family Support: {data.family_support}
- Friend Circle Quality: {data.friend_circle}

**Task:**
Provide 5-7 personalized, actionable academic improvement recommendations that:
1. Are warm, encouraging, and supportive in tone
2. Address their specific weak areas based on the profile
3. Acknowledge their strengths
4. Include concrete, practical steps they can take immediately
5. Consider their living situation, stress level, and support system
6. Are formatted with emojis and clear sections for readability

Make recommendations specific to their situation, not generic advice. Be understanding and motivational."""

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 800,
            },
            timeout=8,  # 8 second timeout as requested
        )
        response.raise_for_status()
        data_json = response.json()
        ai_text = data_json["choices"][0]["message"]["content"].strip()
        
        if ai_text:
            return {"source": "llm", "recommendations": ai_text}
        else:
            print("⚠️ OpenRouter returned empty response, using general recommendations")
            return {"source": "fallback", "recommendations": general_recs}
            
    except requests.exceptions.Timeout:
        print("⚠️ OpenRouter request timed out after 8 seconds, using general recommendations")
        return {"source": "fallback", "recommendations": general_recs}
    except requests.exceptions.RequestException as e:
        print(f"⚠️ OpenRouter request failed: {e}, using general recommendations")
        return {"source": "fallback", "recommendations": general_recs}
    except Exception as e:
        print(f"⚠️ Unexpected error with OpenRouter: {e}, using general recommendations")
        return {"source": "fallback", "recommendations": general_recs}

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to Margadarshak CGPA Predictor API"}

@app.post("/predict")
async def predict(data: InputData):
    try:
        # ✅ Step 1: Enforce daily usage limit
        check_daily_limit(data.user_id)

        # ✅ Step 2: Predict
        features = build_feature_vector(data)
        prediction = float(model.predict(features)[0])
        prediction = round(max(0.0, min(prediction, 4.0)), 2)

        # ✅ Step 3: Generate recommendations (with timeout and fallback)
        recommendations = get_recommendations(prediction, data)

        # ✅ Step 4: Log assessment in Supabase
        if supabase:
            supabase.table("assessments").insert(
                {
                    "user_id": data.user_id,
                    "predicted_cgpa": prediction,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ).execute()

        return {
            "predicted_cgpa": prediction,
            "recommendations": recommendations["recommendations"],
            "recommendation_source": recommendations["source"],
            "model": "Gradient Boosting v1.1 + OpenRouter AI",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))