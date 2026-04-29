"""
predict.py — Make a traffic situation prediction from user input
Usage:  python src/predict.py
        python src/predict.py --hour 8 --day Monday --car 40 --bike 5 --bus 3 --truck 10
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

MODEL_PATH = Path("model.pkl")
DAY_ORDER  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WEEKEND    = {"Saturday","Sunday"}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl not found — run `python src/train.py` first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_features(hour, minute, day, car, bike, bus, truck):
    total = car + bike + bus + truck
    if total == 0:
        car_f = bike_f = bus_f = truck_f = 0.0
    else:
        car_f, bike_f, bus_f, truck_f = car/total, bike/total, bus/total, truck/total

    is_weekend  = int(day in WEEKEND)
    day_ordinal = DAY_ORDER.index(day) if day in DAY_ORDER else 0

    return np.array([[hour, minute, is_weekend,
                      car, bike, bus, truck,
                      car_f, bike_f, bus_f, truck_f,
                      day_ordinal]])


def predict(hour, minute, day, car, bike, bus, truck):
    bundle = load_model()
    model  = bundle["model"]
    le     = bundle["label_encoder"]

    X     = build_features(hour, minute, day, car, bike, bus, truck)
    y_enc = model.predict(X)[0]
    label = le.inverse_transform([y_enc])[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]

    return label, proba, le.classes_


def interactive():
    print("\n── Traffic Situation Predictor ──────────────────")
    hour   = int(input("Hour (0–23): ").strip())
    minute = int(input("Minute (0/15/30/45): ").strip())
    day    = input(f"Day ({'/'.join(d[:3] for d in DAY_ORDER)}): ").strip().capitalize()
    day    = next((d for d in DAY_ORDER if d.startswith(day[:3])), "Monday")
    car    = int(input("CarCount: ").strip())
    bike   = int(input("BikeCount: ").strip())
    bus    = int(input("BusCount: ").strip())
    truck  = int(input("TruckCount: ").strip())
    return hour, minute, day, car, bike, bus, truck


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict traffic situation")
    parser.add_argument("--hour",   type=int, help="Hour (0–23)")
    parser.add_argument("--minute", type=int, default=0)
    parser.add_argument("--day",    type=str, help="Day of week e.g. Monday")
    parser.add_argument("--car",    type=int, default=20)
    parser.add_argument("--bike",   type=int, default=5)
    parser.add_argument("--bus",    type=int, default=3)
    parser.add_argument("--truck",  type=int, default=8)
    args = parser.parse_args()

    if args.hour is not None and args.day is not None:
        h, m, d = args.hour, args.minute, args.day.capitalize()
        d = next((day for day in DAY_ORDER if day.startswith(d[:3])), "Monday")
        c, b, bu, tr = args.car, args.bike, args.bus, args.truck
    else:
        h, m, d, c, b, bu, tr = interactive()

    label, proba, classes = predict(h, m, d, c, b, bu, tr)

    print(f"\n  Prediction  →  {label.upper()}")
    if proba is not None:
        print("  Probabilities:")
        for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
            bar = "█" * int(p * 20)
            print(f"    {cls:<8} {bar:<20} {p*100:.1f}%")
    print()
