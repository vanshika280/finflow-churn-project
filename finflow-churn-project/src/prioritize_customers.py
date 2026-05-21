import json

import pandas as pd

from utils import (
    ID_COLUMN,
    MODELS_DIR,
    OUTPUTS_DIR,
    add_customer_value_columns,
    ensure_directories,
    load_dataset,
    make_priority_segment,
    primary_retention_reason,
)


def load_model_metadata() -> dict:
    metadata_path = MODELS_DIR / "model_metadata.json"
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_prioritized_segments(df: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    value_df, revenue_available = add_customer_value_columns(df)
    working = value_df.merge(scores, on=ID_COLUMN, how="left")

    clv_top_33_threshold = working["CLV"].quantile(0.67)
    working["Priority_Score"] = working["Churn_Probability"] * working["CLV"]
    working["Priority_Segment"] = working.apply(
        make_priority_segment, axis=1, clv_top_33_threshold=clv_top_33_threshold
    )

    output_columns = [
        ID_COLUMN,
        "Churn_Probability",
        "Risk_Level",
        "Avg_Monthly_Revenue",
        "Expected_Months",
        "CLV",
        "Priority_Score",
        "Priority_Segment",
    ]
    prioritized = working[output_columns].sort_values(
        "Priority_Score", ascending=False
    )
    return prioritized, revenue_available


def create_top_20_save_list(
    df: pd.DataFrame, prioritized: pd.DataFrame
) -> pd.DataFrame:
    behavior_columns = [
        ID_COLUMN,
        "Last_Login_Days_Ago",
        "Feature_Usage_Score",
        "Support_Tickets",
        "Avg_Monthly_Transactions",
    ]
    behavior = df[behavior_columns].copy()
    top_20 = prioritized.head(20).merge(behavior, on=ID_COLUMN, how="left")
    top_20["Primary_Retention_Reason"] = top_20.apply(primary_retention_reason, axis=1)
    top_20.insert(0, "Rank", range(1, len(top_20) + 1))

    return top_20[
        [
            "Rank",
            ID_COLUMN,
            "Churn_Probability",
            "CLV",
            "Priority_Score",
            "Priority_Segment",
            "Primary_Retention_Reason",
        ]
    ]


def top_driver_text(feature_importance: pd.DataFrame, model_name: str) -> str:
    if feature_importance.empty:
        return "Feature importance was unavailable."

    model_importance = feature_importance[feature_importance["Model"] == model_name]
    if model_importance.empty:
        model_importance = feature_importance

    top_features = model_importance.sort_values("Importance", ascending=False).head(3)
    parts = [
        f"{row.Feature} ({row.Importance:.3f})" for row in top_features.itertuples()
    ]
    return ", ".join(parts)


def write_strategy_proposal(
    prioritized: pd.DataFrame,
    top_20: pd.DataFrame,
    revenue_available: bool,
) -> None:
    metadata = load_model_metadata()
    best_model = metadata.get("best_model", "the selected churn model")

    feature_importance_path = OUTPUTS_DIR / "feature_importance.csv"
    if feature_importance_path.exists():
        feature_importance = pd.read_csv(feature_importance_path)
    else:
        feature_importance = pd.DataFrame()

    high_priority_count = int(
        (prioritized["Priority_Segment"] == "High Priority").sum()
    )
    high_risk_count = int((prioritized["Risk_Level"] == "High Risk").sum())
    top_reasons = top_20["Primary_Retention_Reason"].value_counts().head(4)
    reason_text = ", ".join(
        f"{reason}: {count}" for reason, count in top_reasons.items()
    )
    revenue_note = (
        "True monthly revenue was available and used for CLV."
        if revenue_available
        else "Monthly revenue was not available, so Avg_Monthly_Transactions was used as a value proxy. True CLV should be recalculated with revenue data."
    )

    proposal = f"""# FinFlow Retention Strategy Proposal

## Model and Prioritization Context

- Selected model: {best_model}
- Top churn drivers from feature importance: {top_driver_text(feature_importance, best_model)}
- High-risk customers identified: {high_risk_count}
- High-priority customers identified: {high_priority_count}
- Most common top-20 save-list reasons: {reason_text}
- CLV note: {revenue_note}

## Strategy 1: High-Value Inactive Customer Reactivation

Target customers with high churn probability, high CLV, high Last_Login_Days_Ago, low transaction activity, or low feature usage.

Recommended actions:
- Send personalized in-app and email reactivation messages with cashback on the next UPI/card transaction.
- Highlight underused FinFlow features such as bill pay, savings pockets, budgeting, and instant card controls.
- Use a 7-day activation window and track next login, next transaction, and feature adoption.

Business rationale:
These customers are valuable but disengaging. A low-cost incentive tied to a transaction can rebuild the usage habit before account closure.

## Strategy 2: Service Recovery for Friction-Heavy Customers

Target customers with high churn probability and high support ticket volume.

Recommended actions:
- Route high-value customers to priority callbacks or senior support queues.
- Offer fee waivers, faster complaint resolution, and proactive status updates for unresolved issues.
- Create a post-resolution follow-up flow to confirm satisfaction and encourage one meaningful transaction.

Business rationale:
Support tickets are a direct sign of friction. Fast service recovery protects customer trust and can prevent high-value customers from moving salary, payments, or deposits to another bank.

## Measurement Plan

- Primary campaign KPI: retained customers after 30 and 90 days.
- Engagement KPI: login within 7 days, transaction within 14 days, and feature usage score improvement.
- Financial KPI: CLV protected, campaign cost per retained customer, and incremental revenue retained.
"""

    with open(OUTPUTS_DIR / "retention_strategy_proposal.md", "w", encoding="utf-8") as f:
        f.write(proposal)


def main() -> None:
    ensure_directories()

    scores_path = OUTPUTS_DIR / "churn_probability_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(
            "Run python src/train_model.py before prioritizing customers."
        )

    df = load_dataset()
    scores = pd.read_csv(scores_path)

    prioritized, revenue_available = create_prioritized_segments(df, scores)
    prioritized.to_csv(OUTPUTS_DIR / "prioritized_customer_segments.csv", index=False)

    top_20 = create_top_20_save_list(df, prioritized)
    top_20.to_csv(OUTPUTS_DIR / "top_20_save_list.csv", index=False)

    write_strategy_proposal(prioritized, top_20, revenue_available)

    print(f"Saved prioritized segments to: {OUTPUTS_DIR / 'prioritized_customer_segments.csv'}")
    print(f"Saved top-20 save list to: {OUTPUTS_DIR / 'top_20_save_list.csv'}")
    print(f"Saved strategy proposal to: {OUTPUTS_DIR / 'retention_strategy_proposal.md'}")


if __name__ == "__main__":
    main()
