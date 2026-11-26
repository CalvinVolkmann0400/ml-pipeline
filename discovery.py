import os
import time

import joblib
import pandas as pd
from pycaret.classification import compare_models, plot_model, pull, setup
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/bank.csv"
TARGET = "y"

df = pd.read_csv(DATA_PATH, sep=";")
# duration is the length of the call, which is only known after the call ends
df = df.drop(columns=["duration"])


# ---------- PyCaret workflow ----------
start = time.perf_counter()
setup(df, target=TARGET, session_id=42, verbose=False)
top3 = compare_models(n_select=3, verbose=False)
print(pull().to_string())
best = top3[0]
print("Best model:", best)
plot_model(best, plot="confusion_matrix", save=True)
os.replace("Confusion Matrix.png", "images/confusion_matrix.png")
print(f"PyCaret section took {time.perf_counter() - start:.1f} seconds\n")

# ---------- scikit-learn workflow ----------
# Rebuild PyCaret's best model (Ridge Classifier) by hand
start = time.perf_counter()
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

numeric = X.select_dtypes(include="number").columns
categorical = X.select_dtypes(exclude="number").columns
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])

model = Pipeline([("preprocess", preprocess), ("classifier", RidgeClassifier(random_state=42))])
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "model.joblib")
print(f"scikit-learn section took {time.perf_counter() - start:.1f} seconds")
