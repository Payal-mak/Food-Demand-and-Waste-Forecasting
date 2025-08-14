# 🍽️ Food Demand & Waste Forecasting System

A complete end-to-end system for forecasting **food demand** and **food waste** using deep learning, real-time data, and optimized workflows.

---

## Key Features

✅ Dual-objective (demand & waste) forecast  
✅ Integrates real-time weather from [Open-Meteo API](https://open-meteo.com/)  
✅ Optimized LSTM (Optuna-tuned hyperparameters)  
✅ Production Flask REST API  
✅ Logs predictions to `predictions.db` (SQLite)  
✅ Automated testing with `pytest`  
✅ End-to-end scripts and modular codebase

---

## Project Structure

food-demand-and-waste-forecasting/
│
├── api_integrations/ # Fetches external data (e.g., weather)
├── config/ # Stores configuration variables
├── data/ # For data generation and validation
├── database/ # Manages database connections and operations
├── deployment/ # Contains the Flask API and deployment scripts
├── models/ # Defines model architectures
├── monitoring/ # (Future) For model performance monitoring
├── prediction/ # Handles the prediction logic
├── preprocessing/ # Data cleaning and feature engineering pipeline
├── scheduler/ # (Future) For automated, recurring tasks
├── saved_models/ # Stores trained model artifacts
├── tests/ # Automated tests for the application
├── training/ # Scripts for model training and hyperparameter tuning
└── utils/ # Utility functions (e.g., visualization)

## ⚙️ How to Run This Project

**1. Install dependencies**
pip install -r requirements.txt

**2. Generate sample data**
python scripts/generate_dataset.py

Generates food demand/waste data + weather.

**3. Train the model**
python scripts/train_model.py

Artifacts: `lstm_model.keras`, `scaler_X.pkl`, `scaler_y.pkl`

**4. Start the API**
python app/api.py

**5. Make a prediction**

Example: send last 10 weeks (each row = [demand, waste, temp, humidity, ...]):
curl -X POST http://127.0.0.1:5001/predict
-H "Content-Type: application/json"
-d '{
"past_weeks_data": [
[400, 32, 24.5, 72], [420, 34, 23.9, 70], ... (10 arrays total)
]
}'

**Expected JSON Reply:**
{
"forecast": {
"next_week_demand": 425,
"next_week_waste": 35.2
},
"status": "success"
}

All predictions are saved to `predictions.db`.

---

## 🧩 Example Weather API Response

Weather data is fetched for enrichment (example from Open-Meteo):
{
"latitude": 28.6,
"longitude": 77.2,
"daily": {
"temperature_2m_max": [35.2, 34.9, ...],
"temperature_2m_min": [27.6, 27.5, ...],
"precipitation_sum": [0.15, 0.00, ...]
}
}

Your data prep automatically parses and integrates this.

---

## 🧠 Model Architecture

**Multi-output LSTM:**

- Input: 10 weeks of demand, waste, weather features
- Hidden Layer: LSTM (tuned units, dropout)
- Output:
  - Next week demand
  - Next week waste

Input (10 timesteps, N features)
│
[LSTM Layer(s)]
│
[Dense Layers]
/
Demand Waste

**Auto-tuning:** Run
python scripts/hyperparameter_tuning.py

to find best LSTM units, learning rate, etc.

---

## 🧪 Automated Testing

Test endpoints, data integrations, and modeling logic:
pytest tests/

---

## 🎯 Model Evaluation

Visualize predictions vs actual:
python scripts/evaluate_model.py

---

## 🐳 Dockerization (optional)

Add a Dockerfile for easy cloud or container deploy.

---

## 🤖 Extending This Project

- Swap in advanced models (see: `models/`)
- Build retraining with `scheduler/scheduler.py`
- Integrate more external signals (e.g., holidays, local events)
- Add visualization dashboards

---

## 🤝 Contact

Questions/collaboration:  
iampayal018@gmail.com
