import pandas as pd
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


def train_and_save_model():
    # Load datasets
    bus = pd.read_csv("data/indian_bus_fare_dataset.csv")
    cost = pd.read_csv("data/cost_of_living.csv")

    # Clean column names
    bus.columns = bus.columns.str.strip()
    cost.columns = cost.columns.str.strip()

    # Rename bus columns
    bus = bus.rename(columns={
        "Fare Price (INR)": "travel_cost",
        "Source": "source",
        "Destination": "destination"
    })

    # 🔥 FIX: Use INDIA row (since dataset is country-based)
    india_data = cost[cost["Country_name"] == "India"].iloc[0]

    cost_of_living = india_data["Cost_of _living(CL)"]
    ticket_price = india_data["Ticket_price"]

    # Convert to usable values
    food_per_day = cost_of_living / 30
    rent = cost_of_living * 0.4  # simulate rent

    trips = []

    for _, row in bus.iterrows():
        duration = random.randint(2, 7)
        people = random.randint(1, 5)
        accom = random.choice(["budget", "mid", "luxury"])
        activities = random.randint(0, 3)
        mode = random.choice(["bus", "train", "flight"])

        # Travel mode multiplier
        multiplier = {
            "bus": 1.0,
            "train": 1.5,
            "flight": 3.0
        }[mode]

        # Costs
        accom_cost = (rent / 30) * duration * people * {
            "budget": 0.5,
            "mid": 1.0,
            "luxury": 2.0
        }[accom]

        food_cost = food_per_day * duration * people
        activity_cost = activities * 1000

        base_travel = row["travel_cost"]
        travel_cost = base_travel * multiplier * people

        # Add realism noise
        total_cost = (accom_cost + food_cost + activity_cost + travel_cost) * random.uniform(0.9, 1.1)

        trips.append({
            "source": row["source"],
            "destination": row["destination"],
            "duration_days": duration,
            "num_people": people,
            "accommodation": accom,
            "travel_mode": mode,
            "num_activities": activities,
            "travel_cost": travel_cost,
            "total_cost": total_cost
        })

    df = pd.DataFrame(trips)

    # Features
    X = df.drop(columns=["total_cost"])
    y = df["total_cost"]

    categorical = ["source", "destination", "accommodation", "travel_mode"]
    numeric = ["duration_days", "num_people", "num_activities", "travel_cost"]

    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
        ("num", StandardScaler(), numeric)
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=20,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "trip_cost_model.pkl")

    print("✅ Model trained and saved successfully!")

    return model


# Run directly
if __name__ == "__main__":
    train_and_save_model()