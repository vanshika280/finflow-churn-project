# FinFlow Churn Prediction and Retention Prioritization

## Business Problem

FinFlow is a neobank with 2 million users and approximately 8% customer churn every quarter. The goal of this project is to identify customers likely to churn, estimate customer value, and create a prioritized retention list so marketing and support teams can act before account closure.

## Dataset Description

The project uses a 1000-record synthetic customer dataset stored as `dataset.csv`.

Expected fields:

- `Customer_ID`
- `Age`
- `Account_Age_Months`
- `Avg_Monthly_Transactions`
- `Last_Login_Days_Ago`
- `Support_Tickets`
- `Feature_Usage_Score`
- `Avg_Monthly_Revenue`, if available
- `Churned`

The code handles common alternate names. For example, `Support_Tickets_Last_6M` is mapped to `Support_Tickets`, and `Avg_Monthly_Revenue_INR` is mapped to `Avg_Monthly_Revenue`.

Target variable:

- `Churned = 0`: customer did not churn
- `Churned = 1`: customer churned

## EDA Summary

The EDA script checks dataset shape, columns, missing values, churn class distribution, churn rate, numeric feature statistics, and churn vs non-churn feature distributions.

Business observations expected from this dataset:

- Customers with higher `Last_Login_Days_Ago` may be more likely to churn.
- Customers with lower `Feature_Usage_Score` may be more disengaged.
- Customers with more `Support_Tickets` may be dissatisfied.
- Customers with lower transaction activity may have weaker product habit formation.

EDA charts are saved in `outputs/charts/`.

## Modeling Approach

Two models are trained from scratch on the provided synthetic dataset:

1. Decision Tree Classifier
2. Gradient Boosting Classifier

No pre-trained models, external ML APIs, AutoML tools, or downloaded models are used.

`Customer_ID` is excluded from model features, and `Churned` is used only as the target.

## Why Decision Tree and Gradient Boosting Were Used

Decision Tree is easy to explain to business stakeholders and provides clear feature importance.

Gradient Boosting combines many small trees and usually captures non-linear behavior patterns better than a single tree while still supporting feature importance.

## Evaluation Metrics

Both models are evaluated with:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

Because churn can be imbalanced, model selection prioritizes F1-score and ROC-AUC rather than accuracy alone.

## Best Model Selected

The training script selects the best model automatically using highest F1-score, with ROC-AUC as the tie breaker.

For the provided dataset, the selected model was **Gradient Boosting**.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decision Tree | 0.7850 | 0.4493 | 0.8611 | 0.5905 | 0.8962 |
| Gradient Boosting | 0.8950 | 0.7143 | 0.6944 | 0.7042 | 0.9443 |

The selected model is saved to:

`models/churn_model.pkl`

Model metadata and metrics are saved to:

`models/model_metadata.json`

## CLV Formula

Customer lifetime value is calculated as:

```text
CLV = Avg_Monthly_Revenue x Expected_Months
```

Expected months:

```text
Expected_Months = max(1, 12 - Last_Login_Days_Ago / 30)
```

If revenue is unavailable, the project uses `Avg_Monthly_Transactions` as a proxy for customer value. True CLV should be recalculated when revenue data is available.

## Priority Score Formula

```text
Priority_Score = Churn_Probability x CLV
```

This ranks customers by both churn risk and business value.

## Top-20 Save List Explanation

The top-20 save list contains customers with the highest `Priority_Score`. Each customer receives a primary retention reason based on behavior:

- High inactivity
- Low feature usage
- High support tickets
- Low transaction activity

The output is saved to:

`outputs/top_20_save_list.csv`

## Retention Strategies

Strategy 1: High-value inactive customer reactivation

- Target high-value customers with high inactivity, low transactions, or low feature usage.
- Send personalized reactivation offers, cashback on the next transaction, and reminders about underused FinFlow banking features.

Strategy 2: Service recovery for friction-heavy customers

- Target high-risk customers with high support ticket counts.
- Prioritize support callbacks, fee waivers, faster complaint resolution, and relationship manager outreach for valuable customers.

The full proposal is saved to:

`outputs/retention_strategy_proposal.md`

## How to Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run EDA:

```bash
python src/eda.py
```

Train models and generate churn scores:

```bash
python src/train_model.py
```

Create prioritization outputs:

```bash
python src/prioritize_customers.py
```

On macOS systems where `python` points nowhere, use `python3` instead:

```bash
python3 src/eda.py
python3 src/train_model.py
python3 src/prioritize_customers.py
```

## Frontend Dashboard

This project includes a custom HTML/CSS/JavaScript dashboard. It is not built with Streamlit.

Start a local server from the project root:

```bash
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000/frontend/
```

The dashboard reads the generated CSV files in `outputs/` and visualizes:

- Churn risk distribution
- Retention priority segments
- Churn probability spread
- CLV versus churn risk
- Feature importance
- Top-20 save list and retention reasons
- EDA chart gallery

## Output Files

- `outputs/churn_probability_scores.csv`
- `outputs/prioritized_customer_segments.csv`
- `outputs/top_20_save_list.csv`
- `outputs/feature_importance.csv`
- `outputs/retention_strategy_proposal.md`
- `outputs/eda_churn_feature_summary.csv`
- `outputs/eda_business_observations.md`
- `outputs/charts/class_distribution.png`
- `outputs/charts/feature_importance.png`
- `outputs/charts/distribution_*.png`
- `outputs/charts/boxplot_*.png`
- `outputs/charts/correlation_heatmap.png`
- `models/churn_model.pkl`
- `models/model_metadata.json`

## Project Structure

```text
finflow-churn-project/
  dataset.csv
  requirements.txt
  README.md
  src/
    train_model.py
    eda.py
    prioritize_customers.py
    utils.py
  models/
    churn_model.pkl
    model_metadata.json
  outputs/
    churn_probability_scores.csv
    prioritized_customer_segments.csv
    top_20_save_list.csv
    feature_importance.csv
    retention_strategy_proposal.md
    charts/
```

## Team Contribution Section

- ML Engineer: Model training, evaluation, feature importance, and model metadata.
- Data Analyst: EDA, charts, churn distribution insights, and business observations.
- Business Analyst: CLV logic, prioritization framework, and retention strategy.
- Presenter: Presentation flow, demo explanation, and final storytelling.
- Code Reviewer: Code quality, file structure, testing, and README cleanup.

## 5-7 Minute Presentation Outline

Slide 1: FinFlow business problem and quarterly churn impact

Slide 2: Dataset overview and target variable

Slide 3: EDA insights and churn patterns

Slide 4: Modeling approach: Decision Tree vs Gradient Boosting

Slide 5: Evaluation results and selected model

Slide 6: CLV and priority score methodology

Slide 7: Top-20 save list and customer segments

Slide 8: Two retention strategies and expected business value

Slide 9: Team contribution
