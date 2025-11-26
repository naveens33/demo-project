import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def generate_synthetic_data(n=2000):
    np.random.seed(42)

    data = {
        "credit_score": np.random.randint(300, 900, n),
        "annual_income": np.random.randint(200000, 3000000, n),
        "loan_amount": np.random.randint(50000, 2000000, n),
        "loan_term": np.random.choice([12, 24, 36, 48, 60], n),
        "employment_years": np.random.randint(0, 30, n),
        "dependents": np.random.randint(0, 5, n),
        "missed_payments": np.random.randint(0, 6, n),
        "credit_utilization": np.random.uniform(0.1, 1.0, n)
    }

    df = pd.DataFrame(data)

    df["risk_label"] = (
        df["credit_score"]
        - df["missed_payments"] * 20
        - (df["credit_utilization"] * 50)
        + (df["annual_income"] / 100000)
        - (df["loan_amount"] / 50000)
    )

    df["risk_label"] = df["risk_label"].clip(300, 900)

    return df


def train_and_save():
    df = generate_synthetic_data()

    X = df.drop(["risk_label"], axis=1)
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "credit_score_model.pkl")
    print("Model saved as credit_score_model.pkl")


if __name__ == "__main__":
    train_and_save()
