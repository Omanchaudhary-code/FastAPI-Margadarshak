import cloudpickle
from sklearn.linear_model import LinearRegression

# Sample data
X = [
    [1, 90, 0, 7, 1, 8],
    [0, 85, 1, 6, 0, 6],
    [0, 75, 0, 5, 1, 7],
    [1, 95, 1, 9, 0, 9],
    [0, 80, 1, 8, 1, 6]
]
y = [2.8, 3.1, 2.5, 3.9, 3.3]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model with cloudpickle
with open("model.pkl", "wb") as f:
    cloudpickle.dump(model, f)

print("✅ Model saved successfully as model.pkl")
