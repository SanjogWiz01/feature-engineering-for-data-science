
"""
FEATURE SELECTION MASTER SCRIPT
Realistic Example: Election Prediction System

Goal:
Predict whether a voter will support a candidate based on
demographics and political preferences.

This script demonstrates multiple Feature Selection methods
in a single workflow.

Techniques included:

1 Variance Threshold
2 Correlation Filtering
3 Chi-Square Selection
4 Mutual Information
5 Recursive Feature Elimination (RFE)
6 L1 Regularization (Lasso)
7 Tree-Based Feature Importance

Author: Learning Feature Engineering
"""

# ---------------------------------------------------------
# 1 Import Libraries
# ---------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# 2 Create Synthetic Election Dataset
# ---------------------------------------------------------

np.random.seed(42)

n_samples = 1000

data = pd.DataFrame({

    "age": np.random.randint(18, 70, n_samples),

    "income": np.random.randint(15000, 120000, n_samples),

    "education_level": np.random.choice(
        ["highschool", "bachelor", "master", "phd"], n_samples
    ),

    "region": np.random.choice(
        ["urban", "rural", "semiurban"], n_samples
    ),

    "political_interest": np.random.randint(1, 10, n_samples),

    "social_media_hours": np.random.randint(0, 8, n_samples),

    "previous_vote": np.random.choice([0, 1], n_samples),

    "attended_rally": np.random.choice([0, 1], n_samples),

    "trust_government": np.random.randint(1, 10, n_samples),

    "economic_satisfaction": np.random.randint(1, 10, n_samples),

})

# Target variable
data["support_candidate"] = (
    (data["political_interest"] +
     data["trust_government"] +
     data["previous_vote"] * 3)
    > 12
).astype(int)

print("Dataset Preview")
print(data.head())


# ---------------------------------------------------------
# 3 Handle Categorical Variables
# ---------------------------------------------------------

data = pd.get_dummies(data, columns=["education_level", "region"], drop_first=True)

print("\nDataset after encoding:")
print(data.head())


# ---------------------------------------------------------
# 4 Separate Features and Target
# ---------------------------------------------------------

X = data.drop("support_candidate", axis=1)
y = data["support_candidate"]

print("\nFeature shape:", X.shape)


# ---------------------------------------------------------
# 5 Train Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------------------------------------------
# 6 Feature Scaling
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 7 Variance Threshold
# Remove features with extremely low variance
# ---------------------------------------------------------

print("\nApplying Variance Threshold")

var_selector = VarianceThreshold(threshold=0.01)

X_var = var_selector.fit_transform(X_train)

selected_features_variance = X.columns[var_selector.get_support()]

print("Selected Features (Variance):")
print(selected_features_variance)


# ---------------------------------------------------------
# 8 Correlation-Based Feature Removal
# ---------------------------------------------------------

print("\nCorrelation Analysis")

corr_matrix = data.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

drop_columns = [
    column for column in upper_triangle.columns
    if any(upper_triangle[column] > 0.9)
]

print("Highly correlated features to drop:")
print(drop_columns)


# ---------------------------------------------------------
# 9 Univariate Feature Selection (Chi-Square)
# ---------------------------------------------------------

print("\nChi-Square Feature Selection")

chi_selector = SelectKBest(score_func=chi2, k=5)

chi_selector.fit(X_train_scaled, y_train)

chi_scores = pd.Series(
    chi_selector.scores_, index=X.columns
)

print("Chi-Square Scores")
print(chi_scores.sort_values(ascending=False))


# ---------------------------------------------------------
# 10 Mutual Information
# ---------------------------------------------------------

print("\nMutual Information Scores")

mi_scores = mutual_info_classif(X_train_scaled, y_train)

mi_series = pd.Series(mi_scores, index=X.columns)

print(mi_series.sort_values(ascending=False))


# ---------------------------------------------------------
# 11 Recursive Feature Elimination
# ---------------------------------------------------------

print("\nRunning RFE")

model = LogisticRegression(max_iter=200)

rfe = RFE(model, n_features_to_select=5)

rfe.fit(X_train_scaled, y_train)

rfe_features = X.columns[rfe.support_]

print("Selected Features (RFE)")
print(rfe_features)


# ---------------------------------------------------------
# 12 L1 Regularization (Lasso)
# ---------------------------------------------------------

print("\nL1 Regularization Feature Selection")

lasso = Lasso(alpha=0.01)

lasso.fit(X_train_scaled, y_train)

lasso_coef = pd.Series(lasso.coef_, index=X.columns)

print("Lasso Coefficients")
print(lasso_coef)


important_lasso = lasso_coef[lasso_coef != 0]

print("Selected Features (Lasso)")
print(important_lasso.index)


# ---------------------------------------------------------
# 13 Tree-Based Feature Importance
# ---------------------------------------------------------

print("\nRandom Forest Feature Importance")

rf = RandomForestClassifier(n_estimators=200)

rf.fit(X_train, y_train)

importance = pd.Series(
    rf.feature_importances_, index=X.columns
)

print(importance.sort_values(ascending=False))


# ---------------------------------------------------------
# 14 Visualization of Feature Importance
# ---------------------------------------------------------

importance.sort_values().plot(
    kind="barh",
    figsize=(10,6),
    title="Random Forest Feature Importance"
)

plt.show()


# ---------------------------------------------------------
# 15 Final Feature Selection
# ---------------------------------------------------------

top_features = importance.sort_values(ascending=False).head(5).index

print("\nTop Selected Features:")
print(top_features)


# ---------------------------------------------------------
# 16 Final Model Training
# ---------------------------------------------------------

final_model = LogisticRegression()

final_model.fit(X_train[top_features], y_train)

accuracy = final_model.score(X_test[top_features], y_test)

print("\nFinal Model Accuracy:", accuracy)