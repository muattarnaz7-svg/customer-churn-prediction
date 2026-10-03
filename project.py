import os
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE


csv_path = "feature_engineered_churn_data.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset file not found: {csv_path}")

df = pd.read_csv(csv_path)
if "Churn" not in df.columns:
    raise ValueError("The dataset must contain the target column 'Churn'.")
print(df.shape)

# Fix target formatting if needed
if "Churn" in df.columns:
    df["Churn"] = df["Churn"].astype(str).str.strip().str.lower()
    df["Churn"] = df["Churn"].replace({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce")
    df = df.dropna(subset=["Churn"]).copy()
    df["Churn"] = df["Churn"].astype(int)

# Drop any ID-like column if still present
for col in ["customerID"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Convert remaining categorical columns to dummy variables before modeling
categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
if "Churn" in categorical_cols:
    categorical_cols.remove("Churn")

if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Option 1: class_weight parameter use karna (kayi models mein available)
model = LogisticRegression(class_weight="balanced")

# Option 2: SMOTE (Synthetic Minority Oversampling)
# pip install imbalanced-learn
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE: {y_train_balanced.value_counts().to_dict()}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(class_weight="balanced"),
    "KNN": KNeighborsClassifier()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1-Score": f1_score(y_test, predictions)
    })

results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
print(results_df)

import matplotlib.pyplot as plt

results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(
    kind="bar", figsize=(12, 6)
)
plt.title("Model Comparison — Churn Prediction")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

cv_results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    cv_results.append({"Model": name, "Mean F1 (CV)": scores.mean(), "Std": scores.std()})

cv_df = pd.DataFrame(cv_results).sort_values("Mean F1 (CV)", ascending=False)
print(cv_df)

best_model_name = results_df.iloc[0]["Model"]
print(f"Best performing model (so far): {best_model_name}")

# Model ko save kar lein next class ke liye (tuning se pehle)
best_model = models[best_model_name]
joblib.dump(best_model, "best_model_initial.pkl")

