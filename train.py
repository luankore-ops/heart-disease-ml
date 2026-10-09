import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# --- Dati ---
cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
df = pd.read_csv("data/processed.cleveland.data", header=None,
                 names=cols, na_values="?")

df["target"] = (df["num"] > 0).astype(int)
df["hr_ratio"] = df["thalach"] / (220 - df["age"])
df["bp_chol"] = df["trestbps"] * df["chol"] / 1000
df["oldpeak_exang"] = df["oldpeak"] * df["exang"]

X = df.drop(columns=["num", "target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Pipeline ---
cat_cols = ["cp", "restecg", "slope", "thal"]
base_num = ["age", "sex", "trestbps", "chol", "fbs", "thalach",
            "exang", "oldpeak", "ca"]
new_num = ["hr_ratio", "bp_chol", "oldpeak_exang"]

def make_pipe(model, num_cols):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", model)])

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "GradBoost": GradientBoostingClassifier(random_state=42),
}
feature_sets = {"base": base_num, "base+nuove": base_num + new_num}

# --- Cross-validation (solo sul training set) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "roc_auc"]

for fs_name, num_cols in feature_sets.items():
    print(f"\n=== Feature set: {fs_name} ===")
    for name, model in models.items():
        res = cross_validate(make_pipe(model, num_cols),
                             X_train[num_cols + cat_cols], y_train,
                             cv=cv, scoring=scoring)
        print(f"{name:11s}", "  ".join(
            f"{m}={res['test_' + m].mean():.3f}±{res['test_' + m].std():.3f}"
            for m in scoring))

from sklearn.metrics import (classification_report, roc_auc_score,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

num_cols = base_num
final = make_pipe(LogisticRegression(max_iter=1000), num_cols)
final.fit(X_train[num_cols + cat_cols], y_train)

X_te = X_test[num_cols + cat_cols]
y_pred = final.predict(X_te)
y_proba = final.predict_proba(X_te)[:, 1]

print("\n=== TEST SET ===")
print(classification_report(y_test, y_pred, target_names=["sano", "malato"]))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                        display_labels=["sano", "malato"])
plt.savefig("confusion_matrix.png", dpi=150)

import numpy as np
from sklearn.inspection import permutation_importance

# Coefficienti della Logistic Regression (feature numeriche standardizzate)
names = final.named_steps["prep"].get_feature_names_out()
coefs = final.named_steps["model"].coef_[0]
order = np.argsort(np.abs(coefs))[::-1]
print("\n=== Coefficienti (top 10) ===")
for i in order[:10]:
    print(f"{names[i]:25s} {coefs[i]:+.3f}")

# Permutation importance sulle colonne originali
perm = permutation_importance(final, X_te, y_test, n_repeats=30,
                              scoring="roc_auc", random_state=42)
order = np.argsort(perm.importances_mean)[::-1]
print("\n=== Permutation importance (ROC-AUC) ===")
for i in order:
    print(f"{X_te.columns[i]:10s} {perm.importances_mean[i]:.3f} ± {perm.importances_std[i]:.3f}")