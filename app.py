import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Credit Score & Risk Evaluation API")

# Load the new credit score model
model = joblib.load("credit_score_model.pkl")

# Input schema
class CreditRequest(BaseModel):
    credit_score: int
    annual_income: int
    loan_amount: int
    loan_term: int
    employment_years: int
    dependents: int
    missed_payments: int
    credit_utilization: float


@app.get("/")
def home():
    return {"message": "Credit Score & Risk API is running"}


def get_risk_band(score):
    if score >= 750:
        return "Low Risk"
    elif score >= 650:
        return "Moderate Risk"
    elif score >= 550:
        return "High Risk"
    return "Very High Risk"


def get_default_probability(score):
    if score >= 750:
        return round(np.random.uniform(0.05, 0.10), 2)
    elif score >= 650:
        return round(np.random.uniform(0.10, 0.30), 2)
    elif score >= 550:
        return round(np.random.uniform(0.30, 0.55), 2)
    return round(np.random.uniform(0.55, 0.90), 2)


@app.post("/predict")
def predict_credit_risk(data: CreditRequest):

    input_data = [[
        data.credit_score,
        data.annual_income,
        data.loan_amount,
        data.loan_term,
        data.employment_years,
        data.dependents,
        data.missed_payments,
        data.credit_utilization
    ]]

    predicted_score = model.predict(input_data)[0]
    predicted_score = int(np.clip(predicted_score, 300, 900))

    risk_band = get_risk_band(predicted_score)
    probability = get_default_probability(predicted_score)

    return {
        "predicted_credit_score": predicted_score,
        "risk_band": risk_band,
        "default_probability": probability
    }
