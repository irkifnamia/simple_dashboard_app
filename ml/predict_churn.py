"""
Load the trained churn model and predict churn risk for new customers.

Run train_churn_model.py before using this script.
"""

from model_utils import load_model, make_prediction_dataframe


MODEL_FILENAME = "churn_model.pkl"


def predict_churn(input_data):
    """Predict churn risk for one customer dictionary or a list of dictionaries."""
    model = load_model(MODEL_FILENAME)
    input_df = make_prediction_dataframe(input_data)

    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)

    results = []
    for prediction, probability in zip(predictions, probabilities):
        churn_probability = float(probability[1]) if len(probability) > 1 else 0.0
        results.append(
            {
                "churn_risk": "High" if bool(prediction) else "Low",
                "churn_probability": round(churn_probability, 4),
            }
        )

    return results


if __name__ == "__main__":
    sample_customer = {
        "age": 30,
        "monthly_income": 4500,
        "total_orders": 3,
        "loyalty_points": 120,
        "state": "Selangor",
        "membership_status": "Active",
    }

    print(predict_churn(sample_customer))
