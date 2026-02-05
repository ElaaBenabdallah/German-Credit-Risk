# German Credit Risk Assessment (KNN)

This repository contains a complete **credit risk classification** project using the **German Credit (Statlog)** dataset.  
The objective is to predict whether a loan applicant is a **Good** or **Bad** credit risk using the **K-Nearest Neighbors (KNN)** algorithm in **R**.

## Project overview
Banks face two types of classification errors:
- **False Negative (Bad → Good):** approving a risky applicant (typically the most costly error)
- **False Positive (Good → Bad):** rejecting a reliable applicant (lost opportunity)

Because of this, the evaluation focuses not only on accuracy, but also on the **confusion matrix**, **recall/sensitivity for the Bad class**, and a **cost-sensitive interpretation**.
