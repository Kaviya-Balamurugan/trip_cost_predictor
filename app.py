from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
import random

from train_model import train_and_save_model

app = Flask(__name__)

MODEL_PATH = "trip_cost_model.pkl"

# Load or train model
if os.path.exists(MODEL_PATH):
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
else:
    print("Training lightweight model...")
    model = train_and_save_model()


# Home route (UI)
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ✅ Handle optional travel cost
        if data.get("travel_cost") in ["", None, 0]:
            # Estimate travel cost
            distance = random.randint(100, 2000)

            mode_multiplier = {
                "bus": 1.0,
                "train": 1.5,
                "flight": 3.0
            }.get(data.get("travel_mode"), 1.0)

            estimated_travel = distance * 2 * mode_multiplier * int(data.get("num_people"))
        else:
            estimated_travel = float(data.get("travel_cost"))

        # Create dataframe
        trip = pd.DataFrame([{
            "source": data.get("source"),
            "destination": data.get("destination"),
            "duration_days": int(data.get("duration_days")),
            "num_people": int(data.get("num_people")),
            "accommodation": data.get("accommodation"),
            "travel_mode": data.get("travel_mode"),
            "num_activities": int(data.get("num_activities")),
            "travel_cost": estimated_travel
        }])

        # Prediction
        pred = model.predict(trip)[0]

        return jsonify({
            "total_cost": round(pred, 2),
            "cost_per_person": round(pred / int(data.get("num_people")), 2),
            "travel_cost_used": round(estimated_travel, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# Run locally only
if __name__ == "__main__":
    app.run(debug=True)