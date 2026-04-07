from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

from train_model import train_and_save_model

app = Flask(__name__)

MODEL_PATH = "trip_cost_model.pkl"

# ✅ Load or train model automatically
if os.path.exists(MODEL_PATH):
    print("✅ Loading existing model...")
    model = joblib.load(MODEL_PATH)
else:
    print("⚠️ Model not found. Training...")
    model = train_and_save_model()


@app.route("/")
def home():
    return "Trip Cost Predictor Running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        trip = pd.DataFrame([{
            "source": data["source"],
            "destination": data["destination"],
            "duration_days": int(data["duration_days"]),
            "num_people": int(data["num_people"]),
            "accommodation": data["accommodation"],
            "travel_mode": data["travel_mode"],
            "num_activities": int(data["num_activities"]),
            "travel_cost": float(data["travel_cost"])
        }])

        prediction = model.predict(trip)[0]

        return jsonify({
            "total_cost": round(prediction, 2),
            "cost_per_person": round(prediction / int(data["num_people"]), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)