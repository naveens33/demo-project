import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------
# STEP 1: Generate synthetic dataset (acts like real-world data)
# ---------------------------------------------------------
def generate_synthetic_data(n=2000):
    np.random.seed(42)  # Ensures reproducibility

    # ------------------------------
    # Creating feature columns
    # ------------------------------
    # Each column simulates real financial parameters used by banks
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

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # ---------------------------------------------------------
    # STEP 2: Create a synthetic "risk_label" (target variable)
    # ---------------------------------------------------------
    # This mimics how actual credit risk is calculated in industry:
    #   higher credit score → lower risk
    #   more missed payments → higher risk
    #   higher credit utilization → higher risk
    #   higher income → lower risk
    #   higher loan amount → higher risk
    df["risk_label"] = (
        df["credit_score"]
        - df["missed_payments"] * 20
        - (df["credit_utilization"] * 50)
        + (df["annual_income"] / 100000)
        - (df["loan_amount"] / 50000)
    )

    # Clipping risk score between 300–900 like CIBIL-style scoring
    df["risk_label"] = df["risk_label"].clip(300, 900)

    return df


# ---------------------------------------------------------
# STEP 3: Train the model and save it as a .pkl file
# ---------------------------------------------------------
def train_and_save():
    # Load or generate dataset
    df = generate_synthetic_data()

    # ------------------------------
    # Feature matrix (X) and label (y)
    # ------------------------------
    X = df.drop(["risk_label"], axis=1)  # Input features
    y = df["risk_label"]                # Target (what we want to predict)

    # ---------------------------------------------------------
    # STEP 4: Train/Test Split
    # ---------------------------------------------------------
    # 80% data → training
    # 20% data → testing (evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------
    # STEP 5: Model Selection & Training
    # ---------------------------------------------------------
    # Using Random Forest Regressor (supervised ML)
    model = RandomForestRegressor(
        n_estimators=300,   # number of trees
        max_depth=15,       # tree depth
        random_state=42     # reproducibility
    )

    # Train the model on training data
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # STEP 6: Save the trained model to disk
    # ---------------------------------------------------------
    joblib.dump(model, "credit_score_model.pkl")
    print("Model saved as credit_score_model.pkl")


# Run training when executed directly
if __name__ == "__main__":
    train_and_save()
    