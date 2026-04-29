"""
train.py — Train a traffic situation classifier
Outputs: model.pkl, assets/model_report.txt
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

DATA_PATH   = Path("data/traffic_clean.csv")
MODEL_PATH  = Path("model.pkl")
REPORT_PATH = Path("assets/model_report.txt")

DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
FEATURES  = [
    "Hour", "Minute", "IsWeekend",
    "CarCount", "BikeCount", "BusCount", "TruckCount",
    "car_frac", "bike_frac", "bus_frac", "truck_frac",
    "DayOrdinal",
]
TARGET = "Traffic Situation"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Day of the week"] = pd.Categorical(df["Day of the week"], categories=DAY_ORDER, ordered=True)
    df["DayOrdinal"] = df["Day of the week"].cat.codes
    return df


def build_xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def evaluate(model, X, y, name="Model"):
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
    print(f"  {name:<30} F1 = {score.mean():.3f} ± {score.std():.3f}")
    return score.mean()


def train_and_save(df: pd.DataFrame):
    X, y = build_xy(df)

    le   = LabelEncoder()
    y_enc = le.fit_transform(y)

    candidates = {
        "Random Forest":     RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }

    print("Cross-validating candidates...")
    scores = {}
    for name, model in candidates.items():
        scores[name] = evaluate(model, X, y_enc, name)

    best_name  = max(scores, key=scores.get)
    best_model = candidates[best_name]
    print(f"\nBest model: {best_name} (F1={scores[best_name]:.3f})")

    best_model.fit(X, y_enc)

    y_pred = best_model.predict(X)
    report = classification_report(y_enc, y_pred, target_names=le.classes_)
    cm     = confusion_matrix(y_enc, y_pred)

    report_txt = f"""Traffic Situation Classifier — Training Report
{'='*54}
Best model:  {best_name}
CV F1 score: {scores[best_name]:.4f} (5-fold stratified)

Features used:
  {', '.join(FEATURES)}

Classification report (train set):
{report}

Confusion matrix (rows=actual, cols=predicted):
  Classes: {list(le.classes_)}
{cm}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_txt)
    print(f"Report saved to {REPORT_PATH}")

    bundle = {"model": best_model, "label_encoder": le, "features": FEATURES}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model saved to {MODEL_PATH}")

    return best_model, le


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found — run `python src/preprocess.py` first.")

    print(f"Loading {DATA_PATH}...")
    df = load(DATA_PATH)
    print(f"  {len(df):,} rows, {len(FEATURES)} features, target: {TARGET}\n")

    train_and_save(df)
    print("\nDone.")
