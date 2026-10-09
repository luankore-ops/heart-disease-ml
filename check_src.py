from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from src.data import load, NUM_COLS, CAT_COLS
from src.model import build_pipeline

df = load("data/processed.cleveland.data")
X = df[NUM_COLS + CAT_COLS]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc = cross_val_score(build_pipeline(), X_train, y_train, cv=cv, scoring="roc_auc")
print(f"ROC-AUC CV: {auc.mean():.3f} ± {auc.std():.3f}")