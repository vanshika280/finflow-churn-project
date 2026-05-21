# FinFlow Retention Strategy Proposal

## Model and Prioritization Context

- Selected model: Gradient Boosting
- Top churn drivers from feature importance: Last_Login_Days_Ago (0.434), Avg_Monthly_Transactions (0.197), Support_Tickets (0.196)
- High-risk customers identified: 160
- High-priority customers identified: 32
- Most common top-20 save-list reasons: High inactivity: 19, Low transaction activity: 1
- CLV note: True monthly revenue was available and used for CLV.

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
