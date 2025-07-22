import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")

genai.configure(api_key=api_key)

def test_gemini_model():
    try:
        # Pick one model from your list
        valid_model_name = "models/gemini-2.5-pro"

        print(f"Using model: {valid_model_name}")
        model = genai.GenerativeModel(valid_model_name)

        prompt = "Give me one motivational tip for a student."
        response = model.generate_content(prompt)
        print("\n✅ Gemini Response:")
        print(response.text)
    except Exception as e:
        print("❌ Error:", e)

if __name__ == "__main__":
    test_gemini_model()
