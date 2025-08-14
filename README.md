# Unified Multivariate and Dual-Objective Forecasting Model for Sustainable Food Supply Chains

**Author:** Payal Makwana  
**Date:** August 14, 2025  
**Repository:** [Food-Demand-and-Waste-Forecasting](https://github.com/Payal-mak/Food-Demand-and-Waste-Forecasting)

---

## 📌 Project Overview

This project implements an **end-to-end, dual-objective forecasting system** for predicting **food demand** and **food waste** using a **multivariate LSTM** deep learning model.

**Key Features:**

- **Multi-output LSTM** for simultaneous demand and waste prediction
- **External data integration** (historical weather, economic indicators)
- **Automated hyperparameter optimization** with Optuna
- **Flask REST API** for real-time prediction
- **SQLite database logging** for prediction persistence
- **Modular and deployable architecture** for real-world scalability

**Goal:**  
Improve planning accuracy, reduce food waste, and contribute to sustainable food supply chain management for **food vendors, canteens, and supply chain managers**.

---

## 📂 System Architecture

The repository is organized into functional modules:

/data - Data generation and ingestion scripts
/preprocessing - Feature engineering, scaling, time-series sequence creation
/api_integrations - External data fetching (e.g., weather API)
/models - LSTM model architecture and placeholders for advanced models
/training - Training pipeline with Optuna hyperparameter tuning
/prediction - Scripts to load models and make predictions
/deployment - Flask API server for predictions
/database - SQLite database for logging predictions
/tests - Pytest-based automated tests
/utils - Visualization and helper scripts
/saved_models - Serialized trained models and scalers

---

## ⚙️ Methodology

### **1. Data Preprocessing**

- **Synthetic data**: Weekly `num_orders` (demand) and `waste_kg` for 2 years.
- **Feature Engineering**:
  - **Holiday flag** (binary)
  - **Temperature**: Historical average for Ahmedabad (via Open-Meteo API)
  - **Economic indicators**: `fuel_price` and `inflation_index` placeholders
- **Scaling**: MinMaxScaler `[0, 1]`
- **Sequence creation**: 10-week lookback window

---

### **2. Model Training**

1. **Baseline LSTM**: Established initial performance.
2. **Hyperparameter Tuning**:
   - **Tool:** Optuna (50 trials)
   - **Best Parameters:**
     - `lstm_units`: 38
     - `dropout_rate`: 0.366
     - `learning_rate`: 0.0001129
   - **Validation Loss:** 0.0877
3. **Final model** trained with best parameters — captured weekly variance better.

---

### **3. Deployment**

- **Flask API** endpoint `/predict` loads trained model on startup.
- **Prediction persistence** — every forecast is logged into `predictions.db` via SQLite.
- **Automated testing** with `pytest`.

---

## 🛠 Installation

git clone https://github.com/Payal-mak/Food-Demand-and-Waste-Forecasting.git
cd Food-Demand-and-Waste-Forecasting
pip install -r requirements.txt


---

## 🚀 Usage

### **1. Train the Model**

python training/train_models.py


### **2. Run the API Server**

python deployment/flask_api.py

Server will start at:  
`http://127.0.0.1:5001/predict`

---

### **3. Make Predictions**

Using **cURL**:
curl -X POST http://127.0.0.1:5001/predict
-H "Content-Type: application/json"
-d '{"features": [[...10 weeks of scaled feature data...]]}'


**Expected Response:**
{
"demand_forecast": [...],
"waste_forecast": [...]
}


Every prediction will be stored in:  
`database/predictions.db`

---

## 📊 Evaluation

Run:
python utils/evaluate_model.py

Generates:

- **Prediction vs Actual plots** using `plot_pred_vs_actual()` from `utils/visualization.py`
- **Quantitative metrics** for model performance

---

## ✅ Results

- **Fully functional** dual-objective forecasting prototype
- **REST API + Database** integration for end-to-end usability
- **Model accuracy** improved significantly after Optuna tuning
- Predictions closely follow actual demand & waste patterns

---

## 🔮 Future Work

- **Real Data Integration** (replace synthetic dataset)
- **Feature expansion** (marketing promotions, events)
- **Advanced models** (Transformers, hybrid architectures)
- **Monitoring dashboard** for performance tracking
- **Docker containerization** for cloud deployment

---

## 🧪 Testing

pytest tests/

Automated tests ensure:

- API endpoint responds successfully
- Predictions are generated without errors
- Database logging is functional

---

## 👩‍💻 Author

**Payal Makwana**  
Mail - iampayal018@gmail.com
📧 Contact via GitHub Issues for questions or contributions.
