import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from utils import (
    CHARTS_DIR,
    ID_COLUMN,
    MODELS_DIR,
    OUTPUTS_DIR,
    TARGET_COLUMN,
    class_distribution_text,
    ensure_directories,
    load_dataset,
    prepare_model_frame,
    risk_level,
)


def build_models() -> dict[str, Pipeline]:
    return {
        "Decision Tree": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=4,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
    }


def get_feature_importance(
    trained_models: dict[str, Pipeline], feature_columns: list[str]
) -> pd.DataFrame:
    rows = []
    for model_name, pipeline in trained_models.items():
        estimator = pipeline.named_steps["model"]
        importances = estimator.feature_importances_
        for feature, importance in zip(feature_columns, importances):
            rows.append(
                {
                    "Model": model_name,
                    "Feature": feature,
                    "Importance": float(importance),
                }
            )
    return pd.DataFrame(rows).sort_values(["Model", "Importance"], ascending=[True, False])


def save_feature_importance_chart(feature_importance: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance,
        x="Importance",
        y="Feature",
        hue="Model",
        palette="Set2",
    )
    plt.title("Feature Importance by Model")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "feature_importance.png", dpi=150)
    plt.close()


def save_churn_probability_scores(
    df: pd.DataFrame,
    best_model: Pipeline,
    X_full: pd.DataFrame,
) -> pd.DataFrame:
    probabilities = best_model.predict_proba(X_full)[:, 1]
    score_df = pd.DataFrame(
        {
            ID_COLUMN: df[ID_COLUMN],
            "Churn_Probability": probabilities,
            "Predicted_Churn": (probabilities >= 0.5).astype(int),
        }
    )
    score_df["Risk_Level"] = score_df["Churn_Probability"].apply(risk_level)
    score_df.to_csv(OUTPUTS_DIR / "churn_probability_scores.csv", index=False)
    return score_df


def main() -> None:
    ensure_directories()
    sns.set_theme(style="whitegrid")

    df = load_dataset()
    X, y, feature_columns = prepare_model_frame(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = build_models()
    trained_models = {}
    metrics = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        metrics[model_name] = evaluate_model(model, X_test, y_test)

    best_model_name = sorted(
        metrics,
        key=lambda name: (metrics[name]["f1_score"], metrics[name]["roc_auc"]),
        reverse=True,
    )[0]
    best_model = trained_models[best_model_name]

    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns,
    }
    joblib.dump(model_bundle, MODELS_DIR / "churn_model.pkl")

    feature_importance = get_feature_importance(trained_models, feature_columns)
    feature_importance.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    save_feature_importance_chart(feature_importance)

    scores = save_churn_probability_scores(df, best_model, X)

    metadata = {
        "best_model": best_model_name,
        "selection_rule": "Highest F1-score, with ROC-AUC as the tie breaker",
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "class_distribution": class_distribution_text(y),
        "test_size": 0.2,
        "random_state": 42,
        "stratified_split": True,
        "metrics": metrics,
        "score_summary": {
            "customers_scored": int(len(scores)),
            "high_risk_customers": int((scores["Risk_Level"] == "High Risk").sum()),
            "medium_risk_customers": int((scores["Risk_Level"] == "Medium Risk").sum()),
            "low_risk_customers": int((scores["Risk_Level"] == "Low Risk").sum()),
        },
    }

    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Model evaluation results:")
    for model_name, result in metrics.items():
        print(f"\n{model_name}")
        print(f"Accuracy : {result['accuracy']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall   : {result['recall']:.4f}")
        print(f"F1-score : {result['f1_score']:.4f}")
        print(f"ROC-AUC  : {result['roc_auc']:.4f}")
        print(f"Confusion matrix: {result['confusion_matrix']}")

    print(f"\nBest model selected: {best_model_name}")
    print(f"Saved model to: {MODELS_DIR / 'churn_model.pkl'}")
    print(f"Saved churn probability scores to: {OUTPUTS_DIR / 'churn_probability_scores.csv'}")


if __name__ == "__main__":
    main()
