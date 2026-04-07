from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "trip_cost_model.pkl"

# ❌ REMOVE training from server
if not os.path.exists(MODEL_PATH):
    raise Exception("Model file not found. Train locally and upload smaller model.")

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return "Trip Cost Predictor Running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
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

    pred = model.predict(trip)[0]

    return jsonify({
        "total_cost": round(pred, 2),
        "cost_per_person": round(pred / int(data["num_people"]), 2)
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)