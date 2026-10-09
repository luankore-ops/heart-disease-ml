import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import CAT_COLS, NUM_COLS

MODEL_PATH = Path(__file__).parent / "models" / "heart_model.joblib"


def parse_args():
    p = argparse.ArgumentParser(
        description="Stima la probabilità di malattia cardiaca (uso di ricerca, "
                    "non è un dispositivo medico).")
    p.add_argument("--age", type=float, required=True, help="età in anni")
    p.add_argument("--sex", type=int, required=True, choices=[0, 1],
                   help="0 = donna, 1 = uomo")
    p.add_argument("--cp", type=int, required=True, choices=[1, 2, 3, 4],
                   help="dolore toracico: 1 tipico, 2 atipico, 3 non anginoso, 4 asintomatico")
    p.add_argument("--trestbps", type=float, required=True,
                   help="pressione a riposo (mm Hg)")
    p.add_argument("--chol", type=float, required=True,
                   help="colesterolo sierico (mg/dl)")
    p.add_argument("--fbs", type=int, required=True, choices=[0, 1],
                   help="glicemia a digiuno > 120 mg/dl (1 = sì)")
    p.add_argument("--restecg", type=int, required=True, choices=[0, 1, 2],
                   help="ECG a riposo: 0 normale, 1 anomalia ST-T, 2 ipertrofia ventricolare")
    p.add_argument("--thalach", type=float, required=True,
                   help="frequenza cardiaca massima raggiunta")
    p.add_argument("--exang", type=int, required=True, choices=[0, 1],
                   help="angina da sforzo (1 = sì)")
    p.add_argument("--oldpeak", type=float, required=True,
                   help="depressione ST da sforzo rispetto al riposo")
    p.add_argument("--slope", type=int, required=True, choices=[1, 2, 3],
                   help="pendenza ST: 1 ascendente, 2 piatta, 3 discendente")
    p.add_argument("--ca", type=float, default=np.nan, choices=[0, 1, 2, 3],
                   help="vasi principali colorati alla fluoroscopia (opzionale)")
    p.add_argument("--thal", type=float, default=np.nan, choices=[3, 6, 7],
                   help="3 normale, 6 difetto fisso, 7 difetto reversibile (opzionale)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="soglia di decisione (default 0.5)")
    return p.parse_args()


def main():
    args = parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit("Modello non trovato: esegui prima 'python train_final.py'.")

    model = joblib.load(MODEL_PATH)
    row = {c: getattr(args, c) for c in NUM_COLS + CAT_COLS}
    X = pd.DataFrame([row], columns=NUM_COLS + CAT_COLS).astype(float)

    proba = model.predict_proba(X)[0, 1]
    label = "malato" if proba >= args.threshold else "sano"

    print(f"Probabilità di malattia: {proba:.1%}")
    print(f"Classe prevista (soglia {args.threshold}): {label}")

    missing = [c for c in ("ca", "thal") if np.isnan(row[c])]
    if missing:
        print(f"ATTENZIONE: {', '.join(missing)} non fornito/i, sostituito/i con "
              "il valore più tipico di Cleveland: la stima è meno affidabile.")
    print("Nota: strumento di ricerca. Probabilità non calibrate fuori da "
          "Cleveland e non validate clinicamente.")


if __name__ == "__main__":
    main()