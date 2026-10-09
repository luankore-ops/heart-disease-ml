import numpy as np
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)

from src.data import CAT_COLS, NUM_COLS, load
from src.model import build_pipeline

df = load("data/processed.cleveland.data")
X = df[NUM_COLS + CAT_COLS]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pipe = build_pipeline()

# --- Grid search sul training set ---
grid = {"model__C": np.logspace(-3, 2, 11)}
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc")
gs.fit(X_train, y_train)

print("=== ROC-AUC (CV) per valore di C ===")
res = gs.cv_results_
for c, m, s in zip(res["param_model__C"], res["mean_test_score"], res["std_test_score"]):
    print(f"C={float(c):8.3f}  AUC={m:.3f} ± {s:.3f}")
print("\nMigliore: C =", round(gs.best_params_["model__C"], 3),
      "| AUC =", round(gs.best_score_, 3))

# --- Cross-validation annidata: tuning vs C=1 ---
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
nested = cross_val_score(
    GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc"),
    X_train, y_train, cv=outer, scoring="roc_auc")
default = cross_val_score(pipe, X_train, y_train, cv=outer, scoring="roc_auc")

print("\n=== Confronto (stessi fold esterni) ===")
print(f"Con tuning (annidata): {nested.mean():.3f} ± {nested.std():.3f}")
print(f"C=1 (default):         {default.mean():.3f} ± {default.std():.3f}")