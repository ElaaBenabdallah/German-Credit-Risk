# German Credit Risk Classification 💳

### Predicting Credit Risk with Machine Learning

This project applies **K-Nearest Neighbors (KNN)** to classify credit applicants as **Good** or **Bad** risk using the **German Credit dataset**.

The goal is not only to build an accurate classifier, but to understand the **business consequences of credit-risk errors** — especially when a high-risk applicant is incorrectly classified as low-risk.

---

## 🎯 Project Overview

The analysis follows an end-to-end machine learning workflow:

**EDA → Data Preprocessing → Feature Scaling → KNN → Cross-Validation → Evaluation → Cost Analysis**

Key steps include:

- 🔎 Exploratory analysis of credit amount, loan duration, installment rate, and risk
- 🧹 One-hot encoding of categorical variables
- ⚖️ Feature standardization for distance-based learning
- 🤖 KNN classification with cross-validation to select the optimal `K`
- 📊 Evaluation using Accuracy, Precision, Recall, F1, Specificity and ROC-AUC
- 💰 Cost-sensitive analysis of False Positives vs. False Negatives

---

## 📊 Key Visualizations

### Credit Risk Distribution
<img width="900" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/546bd51f-fce6-4e99-9365-aa9c080c426c" />



### Bad Risk Rate by Loan Duration
<img width="900" height="600" alt="Figure_8" src="https://github.com/user-attachments/assets/8154c7fc-1c01-42a7-af86-f1c1aa4c20df" />


### Confusion Matrix
<img width="900" height="600" alt="Figure_10" src="https://github.com/user-attachments/assets/04df401f-3a65-4ea7-8307-0ad18f297a25" />



### ROC Curve
<img width="900" height="600" alt="Figure_11" src="https://github.com/user-attachments/assets/a829f776-3033-4c44-a004-3d8d43e52812" />



---

## 📈 Model Results

| Metric | Result |
|---|---:|
| Best K | **7** |
| Test Accuracy | **0.7080** |
| Bad Risk Recall | **0.2667** |
| F1-Score | **[0.354]** |
| ROC-AUC | **0.203** |
| Total Classification Cost | **145** |

> **Why does this matter?**  
> In credit risk, accuracy alone is not enough. Misclassifying a **Bad applicant as Good** can be significantly more costly than incorrectly rejecting a Good applicant. This project therefore evaluates the model from both a **machine learning and business perspective**.

---

## 💡 Key Takeaway

The project demonstrates how machine learning can support credit-risk decisions while highlighting an important principle:

**The best model is not necessarily the one with the highest accuracy — it is the one that balances predictive performance with business risk.**

---

## 🛠️ Tech Stack

**Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn**

---

## 📁 Project Structure

```text
German-Credit-Risk/
│
├──german_credit_risk.py
│
├── images/
│   └── project visualizations
│
├── German-Credit-Risk-Report.pdf
│
└── README.md

