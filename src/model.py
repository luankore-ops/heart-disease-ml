from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CAT_COLS, NUM_COLS


def build_pipeline(model=None, num_cols=NUM_COLS, cat_cols=CAT_COLS):
    """Pipeline: imputazione, scaling/one-hot, modello (default LogReg)."""
    if model is None:
        model = LogisticRegression(max_iter=1000)
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", model)])