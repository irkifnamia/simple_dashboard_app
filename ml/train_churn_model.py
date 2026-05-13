"""
Train and select the best customer churn risk model.

This script is separate from app.py, so it does not change the dashboard or CRUD
workflow. It reads the latest customers from Supabase when possible, falls back
to the sample CSV for local practice, compares several classifiers, and saves
the best model plus evaluation files into the models folder.
"""

import math

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from model_utils import (
    MODELS_DIR,
    clean_dataframe,
    ensure_models_dir,
    load_latest_customer_data,
    save_model,
)


MODEL_FILENAME = "churn_model.pkl"
EVALUATION_FILENAME = "churn_evaluation.csv"
FEATURE_IMPORTANCE_FILENAME = "churn_feature_importance.csv"
TARGET_COLUMN = "churn_risk"


def create_churn_target(df):
    """
    Create the target column the model will learn.

    If the dataset already has a churn-like column, we reuse it. Otherwise, a
    beginner-friendly rule marks a customer as high risk when they show low
    purchase frequency, low total sales/value, or long inactivity.
    """
    existing_target = find_existing_churn_column(df)
    if existing_target:
        return normalize_binary_target(df[existing_target])

    churn_risk = pd.Series(False, index=df.index)

    purchase_column = find_first_existing_column(
        df,
        ["purchase_frequency", "total_orders", "order_count", "orders", "frequency"],
    )
    if purchase_column:
        purchases = to_numeric_series(df[purchase_column])
        churn_risk = churn_risk | is_low_value(purchases)

    sales_column = find_first_existing_column(
        df,
        [
            "total_sales",
            "total_spent",
            "total_revenue",
            "sales",
            "revenue",
            "monthly_income",
        ],
    )
    if sales_column:
        sales = to_numeric_series(df[sales_column])
        churn_risk = churn_risk | is_low_value(sales)

    inactivity_column = find_first_existing_column(
        df,
        [
            "days_since_last_purchase",
            "days_inactive",
            "inactive_days",
            "last_purchase_days",
        ],
    )
    if inactivity_column:
        inactive_days = to_numeric_series(df[inactivity_column])
        churn_risk = churn_risk | is_high_value(inactive_days)
    else:
        last_activity_column = find_first_existing_column(
            df,
            [
                "last_purchase_date",
                "last_order_date",
                "last_active_date",
                "last_activity_date",
                "updated_at",
            ],
        )
        if last_activity_column:
            inactive_days = days_since_date(df[last_activity_column])
            churn_risk = churn_risk | is_high_value(inactive_days)

    return churn_risk.astype(int)


def find_existing_churn_column(df):
    """Find a churn label that may already exist in the data."""
    for column in ["churn", "churn_risk", "is_churned", "churned"]:
        if column in df.columns:
            return column

    return None


def normalize_binary_target(series):
    """Convert common yes/no or high/low labels into 1 and 0."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    high_risk_values = {"1", "yes", "true", "high", "churn", "churned", "inactive"}
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(high_risk_values)
        .astype(int)
    )


def find_first_existing_column(df, possible_columns):
    """Return the first matching column name from a list of possibilities."""
    for column in possible_columns:
        if column in df.columns:
            return column

    return None


def to_numeric_series(series):
    """Safely convert a column to numbers for rule checks."""
    return pd.to_numeric(series, errors="coerce")


def is_low_value(series):
    """Mark values in the lowest 25 percent as low."""
    clean_series = series.dropna()
    if clean_series.empty:
        return pd.Series(False, index=series.index)

    threshold = clean_series.quantile(0.25)
    return series.fillna(clean_series.median()) <= threshold


def is_high_value(series):
    """Mark values in the highest 25 percent as high."""
    clean_series = series.dropna()
    if clean_series.empty:
        return pd.Series(False, index=series.index)

    threshold = clean_series.quantile(0.75)
    return series.fillna(clean_series.median()) >= threshold


def days_since_date(series):
    """Convert last activity dates into number of inactive days."""
    dates = pd.to_datetime(series, errors="coerce", utc=True)
    latest_date = dates.max()

    if pd.isna(latest_date):
        return pd.Series(pd.NA, index=series.index)

    return (latest_date - dates).dt.days


def prepare_features(df):
    """
    Select safe model features.

    Identifier columns and obvious target/leakage columns are removed. Numeric
    columns are scaled, and categorical columns are one-hot encoded.
    """
    drop_columns = {
        TARGET_COLUMN,
        "churn",
        "is_churned",
        "churned",
        "id",
        "user_id",
        "customer_id",
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
    feature_df = df.drop(columns=[c for c in drop_columns if c in df.columns])

    numeric_features = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
        column for column in feature_df.columns if column not in numeric_features
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns were found for churn training.")

    return feature_df, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """Build preprocessing steps for missing values, scaling, and encoding."""
    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numbers", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
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
    """Return the classification models to compare."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
    }


def train(csv_path=None):
    """Train all churn models, save the best one, and write evaluation outputs."""
    ensure_models_dir()

    # Step 1: Load the latest customer data from Supabase or local CSV fallback.
    df = clean_dataframe(load_latest_customer_data(csv_path))
    if df.empty:
        raise ValueError("No customer rows were found for churn training.")

    # Step 2: Create churn_risk if the dataset does not already provide churn.
    df[TARGET_COLUMN] = create_churn_target(df)
    y = df[TARGET_COLUMN].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The churn target has only one class. Add more varied customer data "
            "or adjust the churn risk rule before training."
        )

    # Step 3: Prepare features safely.
    X, numeric_features, categorical_features = prepare_features(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []
    trained_models = {}

    # Step 4: Train and compare all required classification models.
    for model_name, classifier in get_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

        metrics = evaluate_model(pipeline, X_test, y_test)
        cv_metrics = cross_validate_model(pipeline, X_train, y_train)
        results.append({"model": model_name, **metrics, **cv_metrics})

    evaluation_df = pd.DataFrame(results)
    evaluation_path = MODELS_DIR / EVALUATION_FILENAME
    evaluation_df.to_csv(evaluation_path, index=False)

    # Step 5: Pick the strongest model by F1, then ROC-AUC, then accuracy.
    best_row = (
        evaluation_df.sort_values(
            by=["f1_score", "roc_auc", "accuracy"],
            ascending=False,
            na_position="last",
        )
        .iloc[0]
    )
    best_model_name = best_row["model"]
    best_model = trained_models[best_model_name]

    model_path = save_model(best_model, MODEL_FILENAME)
    save_feature_importance(best_model, best_model_name)

    print(f"Best churn model: {best_model_name}")
    print(f"Saved model to: {model_path}")
    print(f"Saved evaluation to: {evaluation_path}")


def evaluate_model(model, X_test, y_test):
    """Calculate test-set metrics for one trained model."""
    predictions = model.predict(X_test)
    probabilities = get_positive_class_scores(model, X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": safe_roc_auc(y_test, probabilities),
    }


def cross_validate_model(model, X_train, y_train):
    """Run cross-validation when each class has enough rows."""
    smallest_class_count = y_train.value_counts().min()
    split_count = min(5, int(smallest_class_count))

    if split_count < 2:
        return {
            "cv_accuracy": math.nan,
            "cv_precision": math.nan,
            "cv_recall": math.nan,
            "cv_f1_score": math.nan,
            "cv_roc_auc": math.nan,
        }

    cv = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1_score": "f1",
            "roc_auc": "roc_auc",
        },
        error_score=math.nan,
    )

    return {
        "cv_accuracy": scores["test_accuracy"].mean(),
        "cv_precision": scores["test_precision"].mean(),
        "cv_recall": scores["test_recall"].mean(),
        "cv_f1_score": scores["test_f1_score"].mean(),
        "cv_roc_auc": scores["test_roc_auc"].mean(),
    }


def get_positive_class_scores(model, X):
    """Return probability-like scores for ROC-AUC when a model supports them."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return None


def safe_roc_auc(y_true, probabilities):
    """Calculate ROC-AUC only when it is possible."""
    if probabilities is None or y_true.nunique() < 2:
        return math.nan

    try:
        return roc_auc_score(y_true, probabilities)
    except ValueError:
        return math.nan


def save_feature_importance(model, model_name):
    """Save feature importance or coefficient strength when available."""
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    classifier = model.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
        value_column = "importance"
    elif hasattr(classifier, "coef_"):
        importance_values = abs(classifier.coef_[0])
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
