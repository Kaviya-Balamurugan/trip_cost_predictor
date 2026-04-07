from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

from train_model import train_and_save_model

app = Flask(__name__)

MODEL_PATH = "trip_cost_model.pkl"

# Load or train
if os.path.exists(MODEL_PATH):
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
else:
    print("Training lightweight model...")
    model = train_and_save_model()


# ✅ ONLY ONE HOME ROUTE
@app.route("/")
def home():
    return render_template("index.html")


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