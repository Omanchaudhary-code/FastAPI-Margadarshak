import cloudpickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math  # for sqrt

# -----------------------------
# Define feature order
# -----------------------------
FEATURE_ORDER = [
    "attendance", "program", "year", "living_situation", "repeated_course",
    "study_hours", "revision", "rating", "online", "group_studies",
    "help_with_teachers", "stress_level", "sleep", "family_support",
    "friend_circle", "income", "first_gen", "part_time_job",
    "financial_pressure", "family_responsibilities", "confidence",
    "motivation", "engagement_index"
]

# -----------------------------
# Generate synthetic dataset
# -----------------------------
def generate_fake_data(n=200):
    rng = np.random.default_rng(seed=42)
    data = {feat: rng.integers(0, 6, size=n) for feat in FEATURE_ORDER if feat != "engagement_index"}
    df = pd.DataFrame(data)
    df["engagement_index"] = df[["study_hours", "revision", "rating",
                                 "online", "group_studies", "help_with_teachers",
                                 "confidence", "motivation", "family_support"]].mean(axis=1) / 5.0
    return df

df = generate_fake_data(500)
X = df[FEATURE_ORDER]
y = 2.0 + (df["engagement_index"] * 2.0) + np.random.normal(0, 0.2, size=len(df))
y = np.clip(y, 0, 4)

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train model
# -----------------------------
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ Model trained. Test RMSE: {rmse:.3f}")

# -----------------------------
# Save model
# -----------------------------
with open("model.pkl", "wb") as f:
    cloudpickle.dump(model, f)

print("📦 Model saved to model.pkl")
