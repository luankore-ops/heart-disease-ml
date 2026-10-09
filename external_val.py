from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data import CAT_COLS, NUM_COLS, load
from src.model import build_pipeline

num_cols, cat_cols = NUM_COLS, CAT_COLS

# --- Modello completo, addestrato su tutto Cleveland ---
train = load("data/processed.cleveland.data")
model = build_pipeline()
model.fit(train[num_cols + cat_cols], train["target"])

sets = {"Hungarian": "data/processed.hungarian.data",
        "Switzerland": "data/processed.switzerland.data",
        "VA": "data/processed.va.data"}

for name, path in sets.items():
    d = load(path)
    X, y = d[num_cols + cat_cols], d["target"]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(f"\n=== {name} (n={len(d)}) ===")
    print("Malati:", round(y.mean(), 2),
          "| mancanti ca:", round(d["ca"].isna().mean(), 2),
          "thal:", round(d["thal"].isna().mean(), 2))
    print("ROC-AUC:", round(roc_auc_score(y, proba), 3),
          "| recall:", round(recall_score(y, pred), 3),
          "| precision:", round(precision_score(y, pred), 3))

# --- Modello senza ca e thal ---
num_r = [c for c in num_cols if c != "ca"]
cat_r = [c for c in cat_cols if c != "thal"]
feats_r = num_r + cat_r

model_r = build_pipeline(num_cols=num_r, cat_cols=cat_r)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_cv = cross_val_predict(model_r, train[feats_r], train["target"],
                         cv=cv, method="predict_proba")[:, 1]
print("\n##### MODELLO SENZA ca E thal #####")
print("Cleveland (CV) ROC-AUC:", round(roc_auc_score(train["target"], p_cv), 3))

model_r.fit(train[feats_r], train["target"])
for name, path in sets.items():
    d = load(path)
    proba = model_r.predict_proba(d[feats_r])[:, 1]
    pred = (proba >= 0.5).astype(int)
    y = d["target"]
    print(f"{name:12s} ROC-AUC: {roc_auc_score(y, proba):.3f}"
          f" | recall: {recall_score(y, pred):.3f}"
          f" | precision: {precision_score(y, pred):.3f}")


# --- ROC-AUC con intervalli di confidenza (bootstrap) ---
def boot_ci(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def fmt(y, p):
    lo, hi = boot_ci(y, p)
    return f"{roc_auc_score(y, p):.3f} [{lo:.3f}-{hi:.3f}]"


print("\n##### ROC-AUC con IC 95% (bootstrap) #####")
print(f"{'Dataset':12s} | {'Completo':22s} | {'Senza ca/thal':22s}")

y_c = train["target"]
p_full = cross_val_predict(model, train[num_cols + cat_cols], y_c,
                           cv=cv, method="predict_proba")[:, 1]
print(f"{'Cleveland CV':12s} | {fmt(y_c, p_full):22s} | {fmt(y_c, p_cv):22s}")

for name, path in sets.items():
    d = load(path)
    y = d["target"]
    pf = model.predict_proba(d[num_cols + cat_cols])[:, 1]
    pr = model_r.predict_proba(d[feats_r])[:, 1]
    print(f"{name:12s} | {fmt(y, pf):22s} | {fmt(y, pr):22s}")


# --- Soglia di decisione (scelta su Cleveland CV) ---
def pick_threshold(y, p, target=0.90):
    best = 0.0
    for t in np.arange(0.01, 0.99, 0.01):
        if recall_score(y, (p >= t).astype(int)) >= target:
            best = t          # l'ultima che soddisfa il target = la più alta
    return best


def report(y, p, t):
    pred = (p >= t).astype(int)
    tn = ((pred == 0) & (np.asarray(y) == 0)).sum()
    fp = ((pred == 1) & (np.asarray(y) == 0)).sum()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return (f"recall={recall_score(y, pred):.3f} "
            f"precision={precision_score(y, pred):.3f} "
            f"specificità={spec:.3f}")


print("\n##### SOGLIA per recall >= 0.90 (scelta su Cleveland CV) #####")
for label, p_cle, mdl, feats in [
    ("Completo", p_full, model, num_cols + cat_cols),
    ("Senza ca/thal", p_cv, model_r, feats_r),
]:
    t = pick_threshold(y_c, p_cle)
    print(f"\n--- {label}: soglia = {t:.2f} ---")
    print(f"{'Cleveland CV':12s} {report(y_c, p_cle, t)}")
    for name, path in sets.items():
        d = load(path)
        p = mdl.predict_proba(d[feats])[:, 1]
        print(f"{name:12s} {report(d['target'], p, t)}")


# --- Calibrazione ---
def calib_summary(y, p):
    y = np.asarray(y)
    prev = y.mean()
    return (f"Brier={brier_score_loss(y, p):.3f} "
            f"(baseline {prev * (1 - prev):.3f}) | "
            f"prob. media={p.mean():.3f} vs prevalenza={prev:.3f}")


print("\n##### CALIBRAZIONE #####")
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, (label, p_cle, mdl, feats) in zip(axes, [
    ("Completo", p_full, model, num_cols + cat_cols),
    ("Senza ca/thal", p_cv, model_r, feats_r),
]):
    print(f"\n--- {label} ---")
    curves = {"Cleveland CV": (y_c, p_cle)}
    for name, path in sets.items():
        d = load(path)
        curves[name] = (d["target"], mdl.predict_proba(d[feats])[:, 1])

    ax.plot([0, 1], [0, 1], "k--", label="calibrazione perfetta")
    for name, (y, p) in curves.items():
        print(f"{name:12s} {calib_summary(y, p)}")
        frac, mean_p = calibration_curve(y, p, n_bins=5, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", label=name)
    ax.set_title(label)
    ax.set_xlabel("Probabilità prevista")
    ax.set_ylabel("Frazione osservata di malati")
    ax.legend(fontsize=8)

plt.tight_layout()
out = Path(__file__).parent / "calibration.png"
plt.savefig(out, dpi=150)
print("\nSalvato in:", out)
plt.show()