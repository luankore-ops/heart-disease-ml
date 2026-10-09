import pandas as pd

cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

df = pd.read_csv("data/processed.cleveland.data", header=None,
                 names=cols, na_values="?")

print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print(df["num"].value_counts().sort_index())

from sklearn.model_selection import train_test_split

df["target"] = (df["num"] > 0).astype(int)

# Feature engineering
df["hr_ratio"] = df["thalach"] / (220 - df["age"])      # FC max raggiunta / FC max teorica
df["bp_chol"] = df["trestbps"] * df["chol"] / 1000      # interazione pressione-colesterolo
df["oldpeak_exang"] = df["oldpeak"] * df["exang"]       # depressione ST solo se angina da sforzo

X = df.drop(columns=["num", "target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(X_train.shape, X_test.shape)
print(y_train.mean(), y_test.mean())

print(X_train[["hr_ratio", "bp_chol", "oldpeak_exang"]].describe())