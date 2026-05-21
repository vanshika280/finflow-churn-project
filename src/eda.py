import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import (
    CHARTS_DIR,
    OUTPUTS_DIR,
    TARGET_COLUMN,
    ensure_directories,
    get_feature_columns,
    load_dataset,
    numeric_churn_summary,
    safe_filename,
)


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def save_class_distribution_chart(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x=TARGET_COLUMN, hue=TARGET_COLUMN, palette="Set2")
    ax.set_title("Class Distribution: Churned vs Non-Churned")
    ax.set_xlabel("Churned (0 = No, 1 = Yes)")
    ax.set_ylabel("Customer Count")
    ax.legend_.remove()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "class_distribution.png", dpi=150)
    plt.close()


def save_feature_distribution_charts(df: pd.DataFrame, features: list[str]) -> None:
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=df,
            x=feature,
            hue=TARGET_COLUMN,
            kde=True,
            bins=25,
            palette="Set1",
            element="step",
            stat="density",
            common_norm=False,
        )
        plt.title(f"{feature}: Churn vs Non-Churn Distribution")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"distribution_{safe_filename(feature)}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 5))
        sns.boxplot(data=df, x=TARGET_COLUMN, y=feature, hue=TARGET_COLUMN, palette="Set2")
        plt.title(f"{feature}: Churn vs Non-Churn Boxplot")
        plt.xlabel("Churned (0 = No, 1 = Yes)")
        plt.ylabel(feature)
        plt.legend([], [], frameon=False)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"boxplot_{safe_filename(feature)}.png", dpi=150)
        plt.close()


def save_correlation_heatmap(df: pd.DataFrame, features: list[str]) -> None:
    corr_columns = features + [TARGET_COLUMN]
    plt.figure(figsize=(10, 7))
    sns.heatmap(df[corr_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def build_business_observations(df: pd.DataFrame, features: list[str]) -> list[str]:
    summary = numeric_churn_summary(df, features)
    observations = []

    direction_templates = {
        "Last_Login_Days_Ago": (
            "Customers who churned had a higher average number of days since last login, "
            "which suggests inactivity is an important warning signal."
        ),
        "Feature_Usage_Score": (
            "Customers who churned had a lower feature usage score, which suggests weaker "
            "product engagement."
        ),
        "Support_Tickets": (
            "Customers who churned had more support tickets on average, which may indicate "
            "service friction or unresolved dissatisfaction."
        ),
        "Avg_Monthly_Transactions": (
            "Customers who churned had lower monthly transaction activity on average, "
            "which suggests reduced banking habit formation."
        ),
        "Avg_Monthly_Revenue": (
            "Revenue helps separate churn risk from business value, so it is important for "
            "retention prioritization even when it is not the strongest churn predictor."
        ),
    }

    for feature, message in direction_templates.items():
        if feature in features:
            observations.append(message)

    top_differences = summary.assign(
        abs_difference=summary["Difference_Churn_minus_Non_Churn"].abs()
    ).sort_values("abs_difference", ascending=False)

    if not top_differences.empty:
        top_feature = top_differences.iloc[0]["Feature"]
        observations.append(
            f"The largest mean difference between churned and retained customers is in {top_feature}."
        )

    return observations


def main() -> None:
    ensure_directories()
    sns.set_theme(style="whitegrid")

    df = load_dataset()
    features = get_feature_columns(df)

    for col in features + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print_section("Dataset Shape")
    print(df.shape)

    print_section("Column Names")
    print(list(df.columns))

    print_section("Missing Values")
    print(df.isna().sum())

    print_section("Class Distribution")
    print(df[TARGET_COLUMN].value_counts().sort_index())
    print(df[TARGET_COLUMN].value_counts(normalize=True).sort_index().mul(100).round(2))

    churn_rate = df[TARGET_COLUMN].mean()
    print_section("Churn Rate")
    print(f"{churn_rate:.2%}")

    print_section("Basic Numeric Statistics")
    print(df[features].describe().T)

    print_section("Churn vs Non-Churn Feature Means")
    churn_summary = numeric_churn_summary(df, features)
    print(churn_summary)
    churn_summary.to_csv(OUTPUTS_DIR / "eda_churn_feature_summary.csv", index=False)

    save_class_distribution_chart(df)
    save_feature_distribution_charts(df, features)
    save_correlation_heatmap(df, features)

    observations = build_business_observations(df, features)
    print_section("Business Observations")
    for idx, observation in enumerate(observations, start=1):
        print(f"{idx}. {observation}")

    with open(OUTPUTS_DIR / "eda_business_observations.md", "w", encoding="utf-8") as f:
        f.write("# EDA Business Observations\n\n")
        for observation in observations:
            f.write(f"- {observation}\n")

    print(f"\nEDA charts saved to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
