# 🚀 Trip Cost Prediction System

A Machine Learning-based web application that predicts total trip cost based on user inputs such as destination, duration, accommodation, and travel mode.

---

## 🌐 Live Demo

👉 https://your-app-url.onrender.com

---

## 📌 Features

* 🔮 Predict total trip cost using ML model
* 👥 Calculates cost per person
* ✈️ Supports multiple travel modes (bus, train, flight)
* 🏨 Handles accommodation types (budget, mid, luxury)
* 🧠 Automatically estimates travel cost if not provided
* 🌐 Deployed as a live web application
* 🎨 Simple and clean user interface

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Flask
* **Machine Learning:** Scikit-learn (Random Forest)
* **Deployment:** Render

---

## 📊 How it Works

1. User enters trip details
2. If travel cost is missing, system estimates it
3. Features are processed through ML pipeline
4. Model predicts total trip cost
5. Output is displayed with per-person cost

---

## 🧠 Machine Learning Details

* Model: Random Forest Regressor
* Features:

  * Source & Destination
  * Duration (days)
  * Number of people
  * Accommodation type
  * Travel mode
  * Number of activities
  * Travel cost (optional)
* Evaluation Metrics:

  * MAE
  * R² Score

---

## 📂 Project Structure

```
trip_cost_predictor/
│
├── app.py
├── train_model.py
├── requirements.txt
├── templates/
│   └── index.html
├── data/
│   ├── indian_bus_fare_dataset.csv
│   └── cost_of_living.csv
└── trip_cost_model.pkl
```

---

## ⚙️ Setup Instructions

### 1. Clone repository

```
git clone https://github.com/Kaviya-Balamurugan/trip_cost_predictor.git
cd trip_cost_predictor
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run application

```
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000/
```

---

## 🚀 API Usage

### Endpoint:

```
POST /predict
```

### Sample Request:

```json
{
  "source": "Delhi",
  "destination": "Chennai",
  "duration_days": 5,
  "num_people": 2,
  "accommodation": "mid",
  "travel_mode": "train",
  "num_activities": 2,
  "travel_cost": ""
}
```

### Sample Response:

```json
{
  "total_cost": 85000,
  "cost_per_person": 17000,
  "travel_cost_used": 12000
}
```

---

## 💡 Future Improvements

* 📊 Cost breakdown visualization (travel, food, stay)
* 📍 Distance-based real API integration
* 📱 Mobile responsive UI
* 🔐 User authentication system

---

## 👩‍💻 Author

**Kaviya Balamurugan**
📌 Aspiring Full Stack & ML Developer

