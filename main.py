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
    print(f"❌ Error loading model.pkl: {e}")
    model = None  # Allow app to start even if model fails

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
        "http://localhost:5173",
        "http://localhost:8000",
        "https://localhost:8080",
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
        print("⚠️ Supabase not configured, skipping limit check")
        return
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        print(f"⚠️ Error checking daily limit: {e}")

# -----------------------------
# Enhanced Recommendation System
# -----------------------------
def generate_comprehensive_recommendations(predicted_cgpa: float, data: InputData) -> str:
    """Generate detailed, personalized recommendations based on all student data."""
    recs = []
    
    # Performance tier assessment
    if predicted_cgpa >= 3.5:
        recs.append("🌟 **Excellent Performance Track**: You're on a path to outstanding results! Your current trajectory shows exceptional academic potential.")
    elif predicted_cgpa >= 3.0:
        recs.append("✅ **Strong Performance Foundation**: You have a solid academic base with excellent potential to reach the top tier!")
    elif predicted_cgpa >= 2.5:
        recs.append("📈 **Growth Opportunity Zone**: With focused improvements in key areas, you can significantly boost your CGPA.")
    else:
        recs.append("⚠️ **Critical Improvement Needed**: Let's work together on strengthening your academic foundation with targeted strategies.")
    
    # Attendance-based recommendations
    if data.attendance in ["<75%", "50-75%"]:
        recs.append("📌 **Attendance Critical Alert**: Your attendance is significantly impacting your learning potential. Research shows 85%+ attendance correlates with 0.5-1.0 CGPA improvement. Make attending classes your #1 priority—even recording lectures can't replace live interaction and immediate doubt-clearing.")
    elif data.attendance == "75-90%":
        recs.append("👍 **Good Attendance Pattern**: You're consistent! Push to 90%+ to maximize classroom learning benefits and build stronger connections with professors who notice regular attendees.")
    else:
        recs.append("⭐ **Outstanding Attendance**: Excellent consistency! This discipline is a cornerstone of your academic success and sets you apart.")
    
    # Study hours and habits
    if data.study_hours in ["<1", "1-2"]:
        recs.append("📚 **Study Time Enhancement Needed**: Current study hours are insufficient for optimal performance. Gradually build up to 3-4 focused hours daily using the Pomodoro Technique (25-min study blocks with 5-min breaks). Start with 2 hours and add 30 mins weekly until you reach your target.")
    elif data.study_hours == "2-3":
        recs.append("📖 **Good Study Foundation**: Your routine is developing well! Consider extending to 3-4 hours during exam prep periods and maintain this as your baseline during regular weeks.")
    elif data.study_hours == "3-4":
        recs.append("💪 **Strong Study Commitment**: Excellent dedication! Ensure quality over quantity—take regular breaks, stay hydrated, and mix active learning techniques (practice problems, teaching concepts) with passive reading.")
    else:
        recs.append("⚖️ **Study-Life Balance Check**: While dedication is admirable, 4+ hours daily is intensive. Ensure you're taking adequate breaks, sleeping well, and not burning out. Quality and consistency beat marathon sessions.")
    
    # Add remaining recommendations (truncated for space)
    recs.append(f"🎯 **Your Action Plan**: Based on your profile, prioritize these THREE things this week: 1) {_get_top_priority(data)}, 2) {_get_second_priority(data)}, 3) {_get_third_priority(data)}. Small consistent improvements compound into major CGPA gains. You've got this! 💪")
    
    return "\n\n".join(recs)

def _get_top_priority(data: InputData) -> str:
    """Determine the single most critical improvement area."""
    if data.attendance in ["<75%", "50-75%"]:
        return "Boost attendance to 85%+ immediately—it's your highest leverage action"
    if data.sleep in ["<5", "5-6"]:
        return "Fix your sleep schedule to 7-8 hours—your brain needs it to perform"
    if data.study_hours in ["<1", "1-2"]:
        return "Increase focused study time to at least 2.5 hours daily"
    if data.stress_level in ["Extremely", "Moderate"]:
        return "Address stress through counseling or stress-management techniques"
    if data.revision in ["Strongly disagree", "Disagree"]:
        return "Implement a weekly revision schedule starting this Sunday"
    return "Maintain your strengths and optimize time management"

def _get_second_priority(data: InputData) -> str:
    """Determine the second most important action."""
    if data.help_with_teachers in ["No", "Rarely"]:
        return "Schedule office hours with one professor this week"
    if data.online in ["Never", "Rarely"]:
        return "Find one quality online resource for your toughest subject"
    if data.group_studies in ["Never", "Rarely"]:
        return "Form or join a focused study group of 3-4 serious students"
    if data.revision in ["Neutral"]:
        return "Upgrade revision to active recall and spaced repetition"
    return "Strengthen your existing study systems"

def _get_third_priority(data: InputData) -> str:
    """Determine the third priority action."""
    if data.friend_circle in ["Worst", "Bad"]:
        return "Actively seek out academically motivated peer connections"
    if data.stress_level in ["Somewhat", "Neutral"] and data.sleep == "6-7":
        return "Add one wellness activity (exercise, meditation, hobby) to your routine"
    if data.study_hours == "2-3":
        return "Add 30 minutes of daily study time in your strongest subject to build momentum"
    return "Track your study hours and identify your most productive times"

def get_recommendations(predicted_cgpa: float, data: InputData) -> dict:
    """Get AI-powered personalized recommendations with 8s timeout and fallback."""
    
    # Generate comprehensive fallback recommendations
    general_recs = generate_comprehensive_recommendations(predicted_cgpa, data)
    
    if not OPENROUTER_API_KEY:
        print("⚠️ OpenRouter API key not configured, using comprehensive fallback recommendations")
        return {"source": "fallback", "recommendations": general_recs}

    # Enhanced prompt for AI recommendations
    prompt = f"""You are an empathetic, experienced academic advisor helping a college student succeed. Analyze their complete profile and provide warm, personalized guidance.

**Student Profile:**
• Predicted CGPA: {predicted_cgpa:.2f}/4.0
• Program: {data.program} | Year: {data.year}
• Attendance: {data.attendance}
• Study Hours/Day: {data.study_hours}
• Living: {data.living_situation}
• Repeated Course: {data.repeated_course}
• Regular Revision: {data.revision}
• Study Resources Quality: {data.rating}
• Uses Online Learning: {data.online}
• Group Study: {data.group_studies}
• Seeks Teacher Help: {data.help_with_teachers}
• Stress Level: {data.stress_level}
• Sleep: {data.sleep} hours
• Family Support: {data.family_support}
• Friend Circle: {data.friend_circle}

Provide 6-8 highly personalized, actionable recommendations with emojis and specific action steps."""

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://margadarshak.tech",
                "X-Title": "Margadarshak CGPA Advisor",
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a supportive academic advisor."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            timeout=8,
        )
        
        response.raise_for_status()
        data_json = response.json()
        ai_text = data_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        if ai_text and len(ai_text) > 100:
            print(f"✅ OpenRouter AI recommendation generated ({len(ai_text)} chars)")
            return {"source": "ai", "recommendations": ai_text}
        else:
            print("⚠️ OpenRouter returned empty response, using fallback")
            return {"source": "fallback", "recommendations": general_recs}
            
    except requests.exceptions.Timeout:
        print("⚠️ OpenRouter API timed out → Using fallback")
        return {"source": "fallback", "recommendations": general_recs}
        
    except Exception as e:
        print(f"⚠️ OpenRouter error: {e} → Using fallback")
        return {"source": "fallback", "recommendations": general_recs}

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to Margadarshak CGPA Predictor API", "status": "healthy"}

@app.get("/health")
def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "supabase_connected": supabase is not None
    }

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Please contact administrator.")
        
        # Step 1: Enforce daily usage limit
        check_daily_limit(data.user_id)

        # Step 2: Predict
        features = build_feature_vector(data)
        prediction = float(model.predict(features)[0])
        prediction = round(max(0.0, min(prediction, 4.0)), 2)

        # Step 3: Generate recommendations
        recommendations = get_recommendations(prediction, data)

        # Step 4: Log assessment in Supabase
        if supabase:
            try:
                supabase.table("assessments").insert({
                    "user_id": data.user_id,
                    "predicted_cgpa": prediction,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }).execute()
            except Exception as e:
                print(f"⚠️ Error logging to Supabase: {e}")

        return {
            "predicted_cgpa": prediction,
            "recommendations": recommendations["recommendations"],
            "recommendation_source": recommendations["source"],
            "model": "Gradient Boosting v1.1 + OpenRouter AI",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Railway startup
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")