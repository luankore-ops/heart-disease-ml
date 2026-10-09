import numpy as np
import pandas as pd

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
CAT_COLS = ["cp", "restecg", "slope", "thal"]
NUM_COLS = ["age", "sex", "trestbps", "chol", "fbs", "thalach",
            "exang", "oldpeak", "ca"]


def load(path):
    """Carica un file UCI 'processed.*.data' e aggiunge il target binario."""
    d = pd.read_csv(path, header=None, names=COLS, na_values="?")
    d["chol"] = d["chol"].replace(0, np.nan)        # 0 = valore mancante
    d["trestbps"] = d["trestbps"].replace(0, np.nan)
    d["target"] = (d["num"] > 0).astype(int)
    return d