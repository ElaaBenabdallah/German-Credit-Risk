# ============================================================
# GERMAN CREDIT RISK CLASSIFICATION
# K-NEAREST NEIGHBORS (KNN)
# ============================================================
#
# Objective:
# Predict whether a credit applicant represents:
#
#   Good Credit Risk
#   Bad Credit Risk
#
# Workflow:
#
# 1. Data acquisition
# 2. Data preparation
# 3. Exploratory Data Analysis
# 4. Risk-rate analysis
# 5. Train/Test split
# 6. Categorical encoding
# 7. Feature scaling
# 8. KNN hyperparameter tuning
# 9. Final model
# 10. Confusion matrix
# 11. Classification metrics
# 12. ROC / AUC
# 13. Cost-sensitive evaluation
# 14. Business interpretation
#
# Dataset:
# UCI Statlog German Credit Data
#
# ============================================================


# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_curve,
    roc_auc_score,
    cohen_kappa_score
)


# Reproducibility
RANDOM_STATE = 123

np.random.seed(RANDOM_STATE)


# Plot style
sns.set_theme(style="whitegrid")

plt.rcParams["figure.figsize"] = (9, 6)


# ============================================================
# 1. DOWNLOAD THE DATASET
# ============================================================

print("=" * 60)
print("1. DOWNLOADING DATASET")
print("=" * 60)

url = (
    "https://archive.ics.uci.edu/static/public/"
    "144/statlog+german+credit+data.zip"
)

zip_path = "german_credit_data.zip"
extract_folder = "german_credit_data"

if not os.path.exists("german.data"):

    print("Downloading dataset...")

    urllib.request.urlretrieve(
        url,
        zip_path
    )

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            extract_folder
        )

    # Find german.data inside extracted folder
    german_data_path = None

    for root, dirs, files in os.walk(
        extract_folder
    ):

        if "german.data" in files:

            german_data_path = os.path.join(
                root,
                "german.data"
            )

            break

    if german_data_path is None:

        raise FileNotFoundError(
            "german.data could not be found."
        )

else:

    german_data_path = "german.data"


print(
    "Dataset located at:",
    german_data_path
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("2. LOADING DATA")
print("=" * 60)


column_names = [

    "checking_account",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "risk"
]


credit = pd.read_csv(
    german_data_path,
    sep=r"\s+",
    header=None,
    names=column_names
)


print(
    "Number of observations:",
    credit.shape[0]
)

print(
    "Number of variables:",
    credit.shape[1]
)


# ============================================================
# 3. CONVERT TARGET VARIABLE
# ============================================================

# Original UCI coding:
#
# 1 = Good
# 2 = Bad

credit["risk"] = credit["risk"].map({
    1: "Good",
    2: "Bad"
})


# ============================================================
# 4. BASIC DATA CHECK
# ============================================================

print("\n" + "=" * 60)
print("3. DATA OVERVIEW")
print("=" * 60)

print("\nFirst five rows:")
print(credit.head())


print("\nData types:")
print(credit.dtypes)


print("\nMissing values:")
print(
    credit.isnull().sum()
)


print("\nRisk distribution:")
print(
    credit["risk"].value_counts()
)


print("\nRisk proportions:")
print(
    credit["risk"]
    .value_counts(
        normalize=True
    )
    .round(4)
)


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("4. EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# 5.1 Risk distribution
# ------------------------------------------------------------

risk_counts = (
    credit["risk"]
    .value_counts()
    .reindex(["Good", "Bad"])
)

risk_percentages = (
    risk_counts /
    len(credit) *
    100
)


plt.figure()

ax = sns.barplot(
    x=risk_counts.index,
    y=risk_counts.values
)

for i, value in enumerate(
    risk_counts.values
):

    ax.text(
        i,
        value + 15,
        f"{value} ({risk_percentages.iloc[i]:.1f}%)",
        ha="center"
    )

plt.title(
    "German Credit Risk Distribution"
)

plt.xlabel(
    "Credit Risk"
)

plt.ylabel(
    "Number of Applicants"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 5.2 Credit amount distribution
# ------------------------------------------------------------

plt.figure()

sns.histplot(
    data=credit,
    x="credit_amount",
    bins=30,
    kde=True
)

plt.title(
    "Distribution of Credit Amount"
)

plt.xlabel(
    "Credit Amount"
)

plt.ylabel(
    "Number of Applicants"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 5.3 Credit amount by risk
# ------------------------------------------------------------

plt.figure()

sns.boxplot(
    data=credit,
    x="risk",
    y="credit_amount"
)

plt.title(
    "Credit Amount by Risk Class"
)

plt.xlabel(
    "Credit Risk"
)

plt.ylabel(
    "Credit Amount"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 5.4 Loan duration distribution
# ------------------------------------------------------------

plt.figure()

sns.histplot(
    data=credit,
    x="duration",
    bins=25,
    kde=True
)

plt.title(
    "Distribution of Loan Duration"
)

plt.xlabel(
    "Duration (Months)"
)

plt.ylabel(
    "Number of Applicants"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 5.5 Loan duration by risk
# ------------------------------------------------------------

plt.figure()

sns.boxplot(
    data=credit,
    x="risk",
    y="duration"
)

plt.title(
    "Loan Duration by Credit Risk"
)

plt.xlabel(
    "Credit Risk"
)

plt.ylabel(
    "Duration (Months)"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 5.6 Installment rate by risk
# ------------------------------------------------------------

installment_counts = (
    credit
    .groupby(
        ["installment_rate", "risk"]
    )
    .size()
    .reset_index(
        name="count"
    )
)


plt.figure()

sns.barplot(
    data=installment_counts,
    x="installment_rate",
    y="count",
    hue="risk"
)

plt.title(
    "Installment Rate by Credit Risk"
)

plt.xlabel(
    "Installment Rate Category"
)

plt.ylabel(
    "Number of Applicants"
)

plt.tight_layout()

plt.show()


# ============================================================
# 6. BAD RISK RATE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("5. BAD RISK RATE ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# 6.1 Bad risk rate by installment rate
# ------------------------------------------------------------

installment_risk = (
    credit
    .groupby("installment_rate")
    .agg(
        total=("risk", "size"),
        bad_cases=(
            "risk",
            lambda x: (x == "Bad").sum()
        )
    )
)

installment_risk["bad_rate"] = (
    installment_risk["bad_cases"] /
    installment_risk["total"]
)


print(
    "\nBad risk rate by installment rate:"
)

print(
    installment_risk
)


plt.figure()

ax = sns.barplot(
    data=installment_risk.reset_index(),
    x="installment_rate",
    y="bad_rate"
)

for container in ax.containers:

    ax.bar_label(
        container,
        fmt="%.2f"
    )

plt.title(
    "Bad Credit Risk Rate by Installment Rate"
)

plt.xlabel(
    "Installment Rate Category"
)

plt.ylabel(
    "Bad Risk Rate"
)

plt.ylim(
    0,
    max(
        installment_risk["bad_rate"]
    ) * 1.2
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 6.2 Create duration groups
# ------------------------------------------------------------

credit["duration_group"] = pd.cut(
    credit["duration"],
    bins=[
        -np.inf,
        12,
        24,
        np.inf
    ],
    labels=[
        "≤ 12 months",
        "13–24 months",
        "> 24 months"
    ]
)


duration_risk = (
    credit
    .groupby(
        "duration_group",
        observed=False
    )
    .agg(
        total=("risk", "size"),
        bad_cases=(
            "risk",
            lambda x: (x == "Bad").sum()
        )
    )
)

duration_risk["bad_rate"] = (
    duration_risk["bad_cases"] /
    duration_risk["total"]
)


print(
    "\nBad risk rate by loan duration:"
)

print(
    duration_risk
)


# ------------------------------------------------------------
# 6.3 Bad risk rate by duration
# ------------------------------------------------------------

plt.figure()

sns.barplot(
    data=duration_risk.reset_index(),
    x="duration_group",
    y="bad_rate"
)

plt.title(
    "Bad Credit Risk Rate by Loan Duration"
)

plt.xlabel(
    "Loan Duration"
)

plt.ylabel(
    "Bad Risk Rate"
)

plt.ylim(
    0,
    max(
        duration_risk["bad_rate"]
    ) * 1.2
)

plt.tight_layout()

plt.show()


# Remove temporary variable
credit.drop(
    columns=["duration_group"],
    inplace=True
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("6. TRAIN / TEST SPLIT")
print("=" * 60)


X = credit.drop(
    columns=["risk"]
)

y = credit["risk"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=RANDOM_STATE
)


print(
    "Training observations:",
    len(X_train)
)

print(
    "Testing observations:",
    len(X_test)
)


print(
    "\nTraining class proportions:"
)

print(
    y_train.value_counts(
        normalize=True
    ).round(4)
)


print(
    "\nTesting class proportions:"
)

print(
    y_test.value_counts(
        normalize=True
    ).round(4)
)


# ============================================================
# 8. IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()


numeric_features = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical features:")

print(
    categorical_features
)


print("\nNumerical features:")

print(
    numeric_features
)


# ============================================================
# 9. PREPROCESSING PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("7. PREPROCESSING")
print("=" * 60)


# One-hot encode categorical variables
#
# StandardScaler scales numerical variables.
#
# For KNN this is extremely important because KNN
# relies on distances between observations.

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numerical",

            StandardScaler(),

            numeric_features
        )

    ]
)


# ============================================================
# 10. CREATE KNN PIPELINE
# ============================================================

knn_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "knn",
            KNeighborsClassifier()
        )

    ]
)


# ============================================================
# 11. KNN HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("8. KNN HYPERPARAMETER TUNING")
print("=" * 60)


# Odd values help reduce ties.

k_values = list(
    range(
        1,
        32,
        2
    )
)


param_grid = {
    "knn__n_neighbors": k_values
}


# Stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


grid_search = GridSearchCV(

    estimator=knn_pipeline,

    param_grid=param_grid,

    cv=cv,

    scoring="accuracy",

    return_train_score=True,

    n_jobs=-1

)


grid_search.fit(
    X_train,
    y_train
)


print(
    "\nBest K:",
    grid_search.best_params_[
        "knn__n_neighbors"
    ]
)


print(
    "Best cross-validation accuracy:",
    round(
        grid_search.best_score_,
        4
    )
)


# ============================================================
# 12. KNN TUNING RESULTS
# ============================================================

cv_results = pd.DataFrame(
    grid_search.cv_results_
)


k_results = cv_results[
    [
        "param_knn__n_neighbors",
        "mean_test_score",
        "std_test_score"
    ]
].copy()


k_results.rename(
    columns={
        "param_knn__n_neighbors": "k",
        "mean_test_score": "cv_accuracy",
        "std_test_score": "cv_std"
    },
    inplace=True
)


k_results["k"] = (
    k_results["k"]
    .astype(int)
)


print(
    "\nKNN tuning results:"
)

print(
    k_results
)


# ============================================================
# 13. VISUALIZE K SELECTION
# ============================================================

plt.figure()

plt.plot(
    k_results["k"],
    k_results["cv_accuracy"],
    marker="o"
)

best_k = grid_search.best_params_[
    "knn__n_neighbors"
]

best_cv_score = grid_search.best_score_


plt.axvline(
    best_k,
    linestyle="--",
    label=f"Best K = {best_k}"
)

plt.scatter(
    [best_k],
    [best_cv_score],
    s=100
)

plt.title(
    "KNN Cross-Validation Performance by K"
)

plt.xlabel(
    "Number of Neighbors (K)"
)

plt.ylabel(
    "Cross-Validation Accuracy"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 14. FINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("9. FINAL KNN MODEL")
print("=" * 60)


best_model = grid_search.best_estimator_


# Predictions
y_pred = best_model.predict(
    X_test
)


# Probability / risk score
#
# KNN probability represents the proportion of
# neighbors belonging to each class.

y_proba = best_model.predict_proba(
    X_test
)


# Identify the probability column corresponding to Bad
bad_class_index = list(
    best_model.classes_
).index("Bad")


bad_probability = y_proba[
    :,
    bad_class_index
]


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("10. CONFUSION MATRIX")
print("=" * 60)


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "Good",
        "Bad"
    ]
)


cm_df = pd.DataFrame(
    cm,

    index=[
        "Actual Good",
        "Actual Bad"
    ],

    columns=[
        "Predicted Good",
        "Predicted Bad"
    ]
)


print(
    cm_df
)


# Visual confusion matrix

plt.figure()

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - KNN"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.show()


# ============================================================
# 16. CONFUSION MATRIX COMPONENTS
# ============================================================

TN = cm[0, 0]

FP = cm[1, 0]

FN = cm[0, 1]

TP = cm[1, 1]


print("\nConfusion matrix components:")

print(
    "True Negatives :",
    TN
)

print(
    "False Positives:",
    FP
)

print(
    "False Negatives:",
    FN
)

print(
    "True Positives :",
    TP
)


# ============================================================
# 17. CLASSIFICATION METRICS
# ============================================================

print("\n" + "=" * 60)
print("11. MODEL PERFORMANCE")
print("=" * 60)


accuracy = accuracy_score(
    y_test,
    y_pred
)


precision_bad = precision_score(
    y_test,
    y_pred,
    pos_label="Bad"
)


recall_bad = recall_score(
    y_test,
    y_pred,
    pos_label="Bad"
)


f1_bad = f1_score(
    y_test,
    y_pred,
    pos_label="Bad"
)


balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


kappa = cohen_kappa_score(
    y_test,
    y_pred
)


specificity_good = TN / (
    TN + FP
)


print(
    f"Accuracy:              {accuracy:.4f}"
)

print(
    f"Precision - Bad:       {precision_bad:.4f}"
)

print(
    f"Recall - Bad:          {recall_bad:.4f}"
)

print(
    f"F1 Score - Bad:        {f1_bad:.4f}"
)

print(
    f"Specificity - Good:    {specificity_good:.4f}"
)

print(
    f"Balanced Accuracy:     {balanced_accuracy:.4f}"
)

print(
    f"Cohen's Kappa:         {kappa:.4f}"
)


# Full classification report

print(
    "\nDetailed classification report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ============================================================
# 18. ROC CURVE / AUC
# ============================================================

print("\n" + "=" * 60)
print("12. ROC / AUC")
print("=" * 60)


# Convert target to binary:
# Good = 0
# Bad = 1

y_test_binary = (
    y_test == "Bad"
).astype(int)


auc_value = roc_auc_score(
    y_test_binary,
    bad_probability
)


fpr, tpr, thresholds = roc_curve(
    y_test_binary,
    bad_probability
)


print(
    f"AUC: {auc_value:.4f}"
)


# ROC plot

plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"KNN (AUC = {auc_value:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title(
    "ROC Curve - KNN Credit Risk Model"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 19. COST-SENSITIVE EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("13. COST-SENSITIVE EVALUATION")
print("=" * 60)


# Based on the German Credit problem formulation:
#
# False Positive:
# Good applicant predicted as Bad
# Cost = 1
#
# False Negative:
# Bad applicant predicted as Good
# Cost = 5
#
# Therefore:
#
# Total Cost =
# (FP × 1) + (FN × 5)

cost_false_positive = 1

cost_false_negative = 5


cost_fp = (
    FP *
    cost_false_positive
)


cost_fn = (
    FN *
    cost_false_negative
)


total_cost = (
    cost_fp +
    cost_fn
)


average_cost = (
    total_cost /
    len(y_test)
)


print(
    "False Positive count:",
    FP
)

print(
    "False Negative count:",
    FN
)

print(
    "False Positive cost:",
    cost_fp
)

print(
    "False Negative cost:",
    cost_fn
)

print(
    "Total classification cost:",
    total_cost
)

print(
    "Average cost per applicant:",
    round(
        average_cost,
        4
    )
)


# ============================================================
# 20. COST VISUALIZATION
# ============================================================

cost_data = pd.DataFrame({

    "Error Type": [
        "Good → Bad\n(False Positive)",
        "Bad → Good\n(False Negative)"
    ],

    "Count": [
        FP,
        FN
    ],

    "Cost per Error": [
        cost_false_positive,
        cost_false_negative
    ],

    "Total Cost": [
        cost_fp,
        cost_fn
    ]

})


plt.figure()

ax = sns.barplot(
    data=cost_data,
    x="Error Type",
    y="Total Cost"
)


for i, value in enumerate(
    cost_data["Total Cost"]
):

    ax.text(
        i,
        value + 2,
        str(value),
        ha="center"
    )


plt.title(
    "Estimated Cost of Classification Errors"
)

plt.xlabel(
    "Classification Error"
)

plt.ylabel(
    "Total Cost"
)

plt.tight_layout()

plt.show()


# ============================================================
# 21. FINAL MODEL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("14. FINAL MODEL SUMMARY")
print("=" * 60)


model_summary = pd.DataFrame({

    "Metric": [

        "Best K",

        "Cross-Validation Accuracy",

        "Test Accuracy",

        "Precision - Bad",

        "Recall - Bad",

        "F1 - Bad",

        "Specificity - Good",

        "Balanced Accuracy",

        "Cohen's Kappa",

        "AUC",

        "False Positives",

        "False Negatives",

        "Total Cost",

        "Average Cost"

    ],

    "Value": [

        best_k,

        round(
            best_cv_score,
            4
        ),

        round(
            accuracy,
            4
        ),

        round(
            precision_bad,
            4
        ),

        round(
            recall_bad,
            4
        ),

        round(
            f1_bad,
            4
        ),

        round(
            specificity_good,
            4
        ),

        round(
            balanced_accuracy,
            4
        ),

        round(
            kappa,
            4
        ),

        round(
            auc_value,
            4
        ),

        FP,

        FN,

        total_cost,

        round(
            average_cost,
            4
        )

    ]

})


print(
    model_summary.to_string(
        index=False
    )
)


# ============================================================
# 22. BUSINESS INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("15. BUSINESS INTERPRETATION")
print("=" * 60)


print(
    f"""
The KNN model selected K = {best_k} using
5-fold stratified cross-validation.

The model achieved a test accuracy of
{accuracy * 100:.1f}%.

For Bad-risk applicants:

- Precision = {precision_bad * 100:.1f}%
- Recall = {recall_bad * 100:.1f}%
- F1 Score = {f1_bad * 100:.1f}%

The model's specificity for Good-risk
applicants was {specificity_good * 100:.1f}%.

The ROC-AUC was {auc_value:.3f}.

The test set contained:

- {FP} Good applicants incorrectly classified as Bad.
- {FN} Bad applicants incorrectly classified as Good.

Using the cost assumptions:

- Good → Bad cost = {cost_false_positive}
- Bad → Good cost = {cost_false_negative}

the estimated total classification cost was
{total_cost}, with an average cost of
{average_cost:.3f} per applicant.
"""
)


print("GERMAN CREDIT RISK ANALYSIS COMPLETED")
