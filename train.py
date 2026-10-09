from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.data import CAT_COLS, NUM_COLS, load
from src.model import build_pipeline

# --- Dati ---
df = load("data/processed.cleveland.data")
df["hr_ratio"] = df["thalach"] / (220 - df["age"])
df["bp_chol"] = df["trestbps"] * df["chol"] / 1000
df["oldpeak_exang"] = df["oldpeak"] * df["exang"]
NEW_NUM = ["hr_ratio", "bp_chol", "oldpeak_exang"]

X = df.drop(columns=["num", "target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Cross-validation (solo sul training set) ---
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "GradBoost": GradientBoostingClassifier(random_state=42),
}
feature_sets = {"base": NUM_COLS, "base+nuove": NUM_COLS + NEW_NUM}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "roc_auc"]

for fs_name, num_cols in feature_sets.items():
    print(f"\n=== Feature set: {fs_name} ===")
    for name, model in models.items():
        pipe = build_pipeline(model, num_cols=num_cols)
        res = cross_validate(pipe, X_train[num_cols + CAT_COLS], y_train,
                             cv=cv, scoring=scoring)
        print(f"{name:11s}", "  ".join(
            f"{m}={res['test_' + m].mean():.3f}±{res['test_' + m].std():.3f}"
            for m in scoring))

# --- Valutazione finale sul test set ---
final = build_pipeline()
final.fit(X_train[NUM_COLS + CAT_COLS], y_train)
X_te = X_test[NUM_COLS + CAT_COLS]
y_pred = final.predict(X_te)
y_proba = final.predict_proba(X_te)[:, 1]

print("\n=== TEST SET ===")
print(classification_report(y_test, y_pred, target_names=["sano", "malato"]))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                        display_labels=["sano", "malato"])
out = Path(__file__).parent / "confusion_matrix.png"
plt.savefig(out, dpi=150)
print("Salvato in:", out)

# --- Interpretabilità ---
names = final.named_steps["prep"].get_feature_names_out()
coefs = final.named_steps["model"].coef_[0]
order = np.argsort(np.abs(coefs))[::-1]
print("\n=== Coefficienti (top 10) ===")
for i in order[:10]:
    print(f"{names[i]:25s} {coefs[i]:+.3f}")

perm = permutation_importance(final, X_te, y_test, n_repeats=30,
                              scoring="roc_auc", random_state=42)
order = np.argsort(perm.importances_mean)[::-1]
print("\n=== Permutation importance (ROC-AUC) ===")
for i in order:
    print(f"{X_te.columns[i]:10s} {perm.importances_mean[i]:.3f} ± {perm.importances_std[i]:.3f}")