from pathlib import Path

import joblib

from src.data import CAT_COLS, NUM_COLS, load
from src.model import build_pipeline

df = load("data/processed.cleveland.data")

model = build_pipeline()
model.fit(df[NUM_COLS + CAT_COLS], df["target"])

out = Path(__file__).parent / "models" / "heart_model.joblib"
out.parent.mkdir(exist_ok=True)
joblib.dump(model, out)
print(f"Modello salvato in: {out} (addestrato su {len(df)} pazienti)")