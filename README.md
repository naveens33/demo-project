# Credit Score & Loan Default Risk Prediction API

A FastAPI-based machine learning service that predicts:

* Credit Score (300-900 scale)
* Risk Band (Low / Moderate / High / Very High)
* Default Probability (0-1)
* Final Default Decision (Default / No Default)

This is a bank-style risk scoring engine inspired by CIBIL/Experian patterns.

---

## Features

* Predicts Credit Score
* Predicts Default Probability
* Determines Risk Band
* Generates Default / No Default decision
* Production-ready FastAPI structure
* Synthetic but banking-like dataset
* Ready for deployment (AWS, Azure, GCP, Render, Docker)

---

## Project Structure

```
credit-risk-api
 app.py
 credit_score_model.pkl
 model_training.py
 README.md
 requirements.txt
```

---

## Installation

### 1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model (optional)

```bash
python tain_model.py
```

This generates the file:

```
credit_score_model.pkl
```

---

## Run the API Server

```bash
uvicorn app:app --reload
```

API Documentation will be available at:

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## GET /

Health check endpoint.

Example:

```json
{
  "message": "Credit Score & Risk API is running"
}
```

---

## POST /predict

### Request Example

```json
{
  "credit_score": 540,
  "annual_income": 400000,
  "loan_amount": 1200000,
  "loan_term": 36,
  "employment_years": 5,
  "dependents": 2,
  "missed_payments": 1,
  "credit_utilization": 0.68
}
```

### Response Example

```json
{
  "predicted_credit_score": 643,
  "risk_band": "High Risk",
  "default_probability": 0.32,
  "default_label": "No Default"
}
```

---

# Risk Band Logic

| Score Range | Risk Band      |
| ----------- | -------------- |
| 750-900     | Low Risk       |
| 650-749     | Moderate Risk  |
| 550-649     | High Risk      |
| 300-549     | Very High Risk |

Default probability decision:

| Probability | Decision   |
| ----------- | ---------- |
| < 0.50      | No Default |
| e 0.50      | Default    |

---

# Sample Input Scenarios

### 1. Low Risk Borrower

```json
{
  "credit_score": 780,
  "annual_income": 2500000,
  "loan_amount": 800000,
  "loan_term": 48,
  "employment_years": 10,
  "dependents": 1,
  "missed_payments": 0,
  "credit_utilization": 0.23
}
```

### 2. High Risk Borrower

```json
{
  "credit_score": 520,
  "annual_income": 350000,
  "loan_amount": 1200000,
  "loan_term": 24,
  "employment_years": 1,
  "dependents": 3,
  "missed_payments": 2,
  "credit_utilization": 0.92
}
```

---

# Testing with Curl

```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"credit_score":600,"annual_income":600000,"loan_amount":900000,"loan_term":36,"employment_years":3,"dependents":2,"missed_payments":1,"credit_utilization":0.45}'
```

---

# License

MIT License.
