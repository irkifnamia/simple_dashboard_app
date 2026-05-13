"""
Train and select the best total order prediction model.

This script is separate from app.py, so it does not change the dashboard or CRUD
workflow. It reads the latest customer data from Supabase when possible, falls
back to the sample CSV for local practice, compares several regression models,
and saves the best model plus evaluation files into the models folder.
"""

import math

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from model_utils import (
    MODELS_DIR,
    clean_dataframe,
    ensure_models_dir,
    load_latest_customer_data,
    save_model,
)


MODEL_FILENAME = "total_order_model.pkl"
EVALUATION_FILENAME = "total_order_evaluation.csv"
FEATURE_IMPORTANCE_FILENAME = "total_order_feature_importance.csv"
TARGET_COLUMN = "total_order"
TARGET_FALLBACK_COLUMN = "total_orders"


def find_target_column(df):
    """
    Find the total order target column.

    The requested name is total_order. The existing dashboard data currently
    uses total_orders, so we support it as a fallback without changing app.py.
    """
    if TARGET_COLUMN in df.columns:
        return TARGET_COLUMN

    if TARGET_FALLBACK_COLUMN in df.columns:
        return TARGET_FALLBACK_COLUMN

    raise ValueError("The data needs a total_order column to train this model.")


def prepare_features(df, target_column):
    """
    Select safe model features.

    Target leakage columns, identifiers, contact details, and date audit fields
    are removed before training. Numeric columns are scaled, and categorical
    columns are encoded in the preprocessing pipeline.
    """
    leakage_columns = {
        target_column,
        TARGET_COLUMN,
        TARGET_FALLBACK_COLUMN,
        "id",
        "user_id",
        "order_count",
        "orders",
        "customer_code",
        "customer_name",
        "full_name",
        "email",
        "phone",
        "address",
        "date_of_birth",
        "created_at",
        "updated_at",
    }

    feature_df = df.drop(columns=[c for c in leakage_columns if c in df.columns])

    # Convert obvious numeric-looking columns so they are treated as numbers.
    for column in feature_df.columns:
        converted = pd.to_numeric(feature_df[column], errors="coerce")
        if converted.notna().sum() >= feature_df[column].notna().sum() * 0.8:
            feature_df[column] = converted

    numeric_features = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
        column for column in feature_df.columns if column not in numeric_features
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns were found for total order training.")

    return feature_df, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """Build preprocessing steps for missing values, scaling, and encoding."""
    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                # Fill missing number values with the middle value.
                ("imputer", SimpleImputer(strategy="median")),
                # Scale numeric values so linear regression behaves well.
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numbers", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                # Fill missing text values with the most common value.
                ("imputer", SimpleImputer(strategy="most_frequent")),
                # Convert text categories into numeric columns for the model.
                ("encoder", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categories", categorical_transformer, categorical_features))

    return ColumnTransformer(transformers=transformers)


def make_one_hot_encoder():
    """Create a OneHotEncoder that works across scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_candidate_models():
    """Return the regression models to compare."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    }


def train(csv_path=None):
    """Train all total order models and save the best model plus reports."""
    ensure_models_dir()

    # Step 1: Load the latest customer data from Supabase or local CSV fallback.
    df = clean_dataframe(load_latest_customer_data(csv_path))
    if df.empty:
        raise ValueError("No customer rows were found for total order training.")

    # Step 2: Use the existing total_order column as the target variable.
    target_column = find_target_column(df)
    y = pd.to_numeric(df[target_column], errors="coerce")
    usable_rows = y.notna()
    df = df.loc[usable_rows].copy()
    y = y.loc[usable_rows]

    if len(df) < 2:
        raise ValueError("At least two rows with total order values are needed.")

    # Step 3: Prepare features safely before model training.
    X, numeric_features, categorical_features = prepare_features(df, target_column)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    results = []
    trained_models = {}

    # Step 4: Train and compare all required regression models.
    for model_name, regressor in get_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ]
        )

        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["cv_rmse"] = cross_validation_rmse(pipeline, X_train, y_train)
        results.append({"model": model_name, **metrics})

    evaluation_df = pd.DataFrame(results)
    evaluation_path = MODELS_DIR / EVALUATION_FILENAME
    evaluation_df.to_csv(evaluation_path, index=False)

    # Step 5: Pick the strongest model by lowest RMSE, then highest R2 score.
    best_row = (
        evaluation_df.sort_values(
            by=["rmse", "r2_score"],
            ascending=[True, False],
            na_position="last",
        )
        .iloc[0]
    )
    best_model_name = best_row["model"]
    best_model = trained_models[best_model_name]

    # Step 6: Save the best model and any available feature importance.
    model_path = save_model(best_model, MODEL_FILENAME)
    save_feature_importance(best_model, best_model_name)

    print(f"Best total order model: {best_model_name}")
    print(f"Saved model to: {model_path}")
    print(f"Saved evaluation to: {evaluation_path}")


def evaluate_model(model, X_test, y_test):
    """Calculate test-set regression metrics for one trained model."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": math.sqrt(mse),
        "r2_score": r2_score(y_test, predictions),
    }


def cross_validation_rmse(model, X_train, y_train):
    """Calculate cross-validation RMSE when enough rows are available."""
    split_count = min(5, len(X_train))
    if split_count < 2:
        return math.nan

    cv = KFold(n_splits=split_count, shuffle=True, random_state=42)
    negative_mse_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_squared_error",
        error_score=math.nan,
    )

    rmse_scores = [math.sqrt(abs(score)) for score in negative_mse_scores]
    return sum(rmse_scores) / len(rmse_scores)


def save_feature_importance(model, model_name):
    """Save feature importance or coefficient strength when available."""
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    regressor = model.named_steps["regressor"]

    if hasattr(regressor, "feature_importances_"):
        importance_values = regressor.feature_importances_
        value_column = "importance"
    elif hasattr(regressor, "coef_"):
        importance_values = abs(regressor.coef_)
        value_column = "absolute_coefficient"
    else:
        return

    importance_df = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_names,
            value_column: importance_values,
        }
    ).sort_values(value_column, ascending=False)

    importance_path = MODELS_DIR / FEATURE_IMPORTANCE_FILENAME
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importance to: {importance_path}")


if __name__ == "__main__":
    train()
