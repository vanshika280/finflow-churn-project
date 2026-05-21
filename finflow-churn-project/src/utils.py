from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"

TARGET_COLUMN = "Churned"
ID_COLUMN = "Customer_ID"

RECOMMENDED_FEATURES = [
    "Age",
    "Account_Age_Months",
    "Avg_Monthly_Transactions",
    "Last_Login_Days_Ago",
    "Support_Tickets",
    "Feature_Usage_Score",
    "Avg_Monthly_Revenue",
]

COLUMN_ALIASES = {
    "Customer_ID": [
        "Customer_ID",
        "CustomerID",
        "Customer Id",
        "Customer Id",
        "Customer",
        "ID",
    ],
    "Age": ["Age", "Customer_Age"],
    "Account_Age_Months": [
        "Account_Age_Months",
        "Account Age Months",
        "Tenure_Months",
        "Customer_Tenure_Months",
        "Account_Tenure_Months",
    ],
    "Avg_Monthly_Transactions": [
        "Avg_Monthly_Transactions",
        "Average_Monthly_Transactions",
        "Monthly_Transactions",
        "Avg Transactions Monthly",
    ],
    "Last_Login_Days_Ago": [
        "Last_Login_Days_Ago",
        "Days_Since_Last_Login",
        "Last Login Days Ago",
        "Inactive_Days",
    ],
    "Support_Tickets": [
        "Support_Tickets",
        "Support_Tickets_Last_6M",
        "Support Tickets",
        "Tickets_Last_6M",
        "Complaints",
    ],
    "Feature_Usage_Score": [
        "Feature_Usage_Score",
        "Feature Usage Score",
        "Product_Usage_Score",
        "App_Feature_Usage_Score",
    ],
    "Avg_Monthly_Revenue": [
        "Avg_Monthly_Revenue",
        "Avg_Monthly_Revenue_INR",
        "Average_Monthly_Revenue",
        "Monthly_Revenue",
        "Revenue",
    ],
    "Churned": ["Churned", "Churn", "Is_Churned", "Exited"],
}


def ensure_directories() -> None:
    """Create output folders used by the project."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known alternate column names to the canonical project names."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    normalized_to_actual = {_normalize_name(col): col for col in df.columns}
    rename_map = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue

        for alias in aliases:
            actual = normalized_to_actual.get(_normalize_name(alias))
            if actual and actual != canonical:
                rename_map[actual] = canonical
                break

    return df.rename(columns=rename_map)


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load and normalize the dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    df = normalize_columns(df)
    validate_required_columns(df)
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in [ID_COLUMN, TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    missing_features = [
        col for col in RECOMMENDED_FEATURES[:-1] if col not in df.columns
    ]
    if missing_features:
        raise ValueError(f"Missing required model feature(s): {missing_features}")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model features that are present in the dataset."""
    return [col for col in RECOMMENDED_FEATURES if col in df.columns]


def prepare_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return numeric model matrix, target, and feature list."""
    feature_columns = get_feature_columns(df)
    model_df = df[feature_columns].copy()

    for col in feature_columns:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if y.isna().any():
        raise ValueError("Target column contains non-numeric or missing values.")

    return model_df, y.astype(int), feature_columns


def risk_level(churn_probability: float) -> str:
    if churn_probability < 0.35:
        return "Low Risk"
    if churn_probability <= 0.65:
        return "Medium Risk"
    return "High Risk"


def calculate_expected_months(last_login_days_ago: float) -> float:
    if pd.isna(last_login_days_ago):
        last_login_days_ago = 0
    return max(1.0, 12.0 - float(last_login_days_ago) / 30.0)


def add_customer_value_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Add Avg_Monthly_Revenue, Expected_Months, and CLV columns.

    If true revenue is unavailable, monthly transactions are used as a practical
    proxy so prioritization can still be demonstrated.
    """
    df = df.copy()
    revenue_available = "Avg_Monthly_Revenue" in df.columns

    if revenue_available:
        df["Avg_Monthly_Revenue"] = pd.to_numeric(
            df["Avg_Monthly_Revenue"], errors="coerce"
        )
    else:
        df["Avg_Monthly_Revenue"] = pd.to_numeric(
            df["Avg_Monthly_Transactions"], errors="coerce"
        )

    df["Avg_Monthly_Revenue"] = df["Avg_Monthly_Revenue"].fillna(
        df["Avg_Monthly_Revenue"].median()
    )
    df["Last_Login_Days_Ago"] = pd.to_numeric(
        df["Last_Login_Days_Ago"], errors="coerce"
    ).fillna(0)
    df["Expected_Months"] = df["Last_Login_Days_Ago"].apply(calculate_expected_months)
    df["CLV"] = df["Avg_Monthly_Revenue"] * df["Expected_Months"]

    return df, revenue_available


def make_priority_segment(row: pd.Series, clv_top_33_threshold: float) -> str:
    if row["Churn_Probability"] >= 0.65 and row["CLV"] >= clv_top_33_threshold:
        return "High Priority"
    if row["Churn_Probability"] >= 0.35:
        return "Medium Priority"
    return "Low Priority"


def primary_retention_reason(row: pd.Series) -> str:
    reasons = []

    if row.get("Last_Login_Days_Ago", 0) >= 45:
        reasons.append(("High inactivity", row.get("Last_Login_Days_Ago", 0)))
    if row.get("Feature_Usage_Score", 10) <= 3:
        reasons.append(("Low feature usage", 10 - row.get("Feature_Usage_Score", 10)))
    if row.get("Support_Tickets", 0) >= 4:
        reasons.append(("High support tickets", row.get("Support_Tickets", 0)))
    if row.get("Avg_Monthly_Transactions", 999) <= 20:
        reasons.append(
            ("Low transaction activity", 20 - row.get("Avg_Monthly_Transactions", 0))
        )

    if not reasons:
        return "High churn probability and customer value"

    return sorted(reasons, key=lambda item: item[1], reverse=True)[0][0]


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()


def class_distribution_text(y: pd.Series) -> str:
    counts = y.value_counts().sort_index()
    total = len(y)
    parts = []
    for label, count in counts.items():
        pct = count / total * 100
        parts.append(f"{label}: {count} ({pct:.1f}%)")
    return ", ".join(parts)


def numeric_churn_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    summary_rows = []
    for feature in features:
        grouped = df.groupby(TARGET_COLUMN)[feature].mean(numeric_only=True)
        non_churn_mean = grouped.get(0, np.nan)
        churn_mean = grouped.get(1, np.nan)
        summary_rows.append(
            {
                "Feature": feature,
                "Non_Churn_Mean": non_churn_mean,
                "Churn_Mean": churn_mean,
                "Difference_Churn_minus_Non_Churn": churn_mean - non_churn_mean,
            }
        )
    return pd.DataFrame(summary_rows)
