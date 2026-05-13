"""
Load the trained total order model and predict total orders for new customers.

Run train_orders_model.py before using this script.
"""

from model_utils import load_model, make_prediction_dataframe


MODEL_FILENAME = "total_order_model.pkl"


def predict_orders(input_data):
    """Predict total orders for one customer dictionary or a list of dictionaries."""
    # Step 1: Load the trained model pipeline from the models folder.
    model = load_model(MODEL_FILENAME)

    # Step 2: Convert the input into a pandas DataFrame.
    input_df = make_prediction_dataframe(input_data)

    # Step 3: Predict total orders.
    predictions = model.predict(input_df)

    # Step 4: Return friendly prediction results.
    return [
        {
            "predicted_total_orders": round(float(prediction), 2),
        }
        for prediction in predictions
    ]


if __name__ == "__main__":
    sample_customer = {
        "age": 30,
        "monthly_income": 4500,
        "loyalty_points": 120,
        "state": "Selangor",
        "membership_status": "Active",
    }

    print(predict_orders(sample_customer))
