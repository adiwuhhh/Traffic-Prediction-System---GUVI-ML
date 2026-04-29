"""
preprocess.py — Cleaning and feature engineering for TrafficTwoMonth dataset
Outputs:  data/traffic_clean.csv
          assets/traffic_agg.json   (pre-aggregated data for the dashboard)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_PATH  = Path("data/TrafficTwoMonth.csv")
CLEAN_PATH = Path("data/traffic_clean.csv")
AGG_PATH   = Path("assets/traffic_agg.json")

DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# ── Load ───────────────────────────────────────────────────────────────────────
def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# ── Clean ──────────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning...")

    # Normalise string columns
    df["Traffic Situation"] = df["Traffic Situation"].str.strip().str.lower()
    df["Day of the week"]   = df["Day of the week"].str.strip()

    # Parse time → hour (0-23)
    df["Time"] = pd.to_datetime(df["Time"], format="%I:%M:%S %p")
    df["Hour"] = df["Time"].dt.hour
    df["Minute"] = df["Time"].dt.minute

    # Ordered categorical day
    df["Day of the week"] = pd.Categorical(
        df["Day of the week"], categories=DAY_ORDER, ordered=True
    )

    # Drop exact duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped {n_before - len(df)} duplicate rows")

    # Sanity: Total should equal sum of vehicle counts
    computed = df["CarCount"] + df["BikeCount"] + df["BusCount"] + df["TruckCount"]
    mismatch = (df["Total"] != computed).sum()
    if mismatch:
        print(f"  Warning: {mismatch} rows where Total != sum of counts (using computed total)")
        df["Total"] = computed

    # Clip negatives (shouldn't exist but guard anyway)
    count_cols = ["CarCount","BikeCount","BusCount","TruckCount","Total"]
    for col in count_cols:
        df[col] = df[col].clip(lower=0)

    print(f"  {len(df):,} rows after cleaning")
    return df


# ── Feature engineering ────────────────────────────────────────────────────────
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features...")

    # Time-of-day buckets
    def time_bucket(h):
        if h < 5:   return "late_night"
        if h < 9:   return "morning_rush"
        if h < 12:  return "late_morning"
        if h < 14:  return "midday"
        if h < 17:  return "afternoon"
        if h < 20:  return "evening_rush"
        if h < 22:  return "evening"
        return "night"

    df["TimeBucket"] = df["Hour"].apply(time_bucket)

    # Weekend flag
    df["IsWeekend"] = df["Day of the week"].isin(["Saturday","Sunday"]).astype(int)

    # Rolling hour-mean within each day (shifted to avoid leakage)
    df = df.sort_values(["Date","Hour","Minute"])
    df["TotalRolling4"] = (
        df.groupby(["Date"])["Total"]
        .transform(lambda x: x.rolling(4, min_periods=1).mean().shift(1))
    )

    # Situation encoded as ordinal
    sit_map = {"low":0, "normal":1, "high":2, "heavy":3}
    df["SituationOrdinal"] = df["Traffic Situation"].map(sit_map)

    # Normalised counts (fraction of total)
    for col, key in [("CarCount","car_frac"),("BikeCount","bike_frac"),
                     ("BusCount","bus_frac"),("TruckCount","truck_frac")]:
        df[key] = df[col] / df["Total"].replace(0, np.nan)

    print(f"  Feature columns added: TimeBucket, IsWeekend, TotalRolling4, SituationOrdinal, *_frac")
    return df


# ── Aggregate for dashboard JSON ───────────────────────────────────────────────
def aggregate(df: pd.DataFrame) -> dict:
    print("Aggregating for dashboard...")

    heatmap = {}
    for day in DAY_ORDER:
        sub = df[df["Day of the week"] == day]
        heatmap[day] = {
            str(h): round(float(sub[sub["Hour"]==h]["Total"].mean()), 1)
            for h in range(24)
            if len(sub[sub["Hour"]==h]) > 0
        }

    hourly = []
    for h in range(24):
        sub = df[df["Hour"]==h]["Total"]
        if len(sub):
            hourly.append({
                "hour": h,
                "avg":  round(float(sub.mean()), 1),
                "max":  int(sub.max()),
                "min":  int(sub.min()),
            })

    vehicles = {
        "car":   int(df["CarCount"].sum()),
        "bike":  int(df["BikeCount"].sum()),
        "bus":   int(df["BusCount"].sum()),
        "truck": int(df["TruckCount"].sum()),
    }

    situations = df["Traffic Situation"].value_counts().to_dict()

    daily = [
        {"date": int(d), "avg": round(float(df[df["Date"]==d]["Total"].mean()), 1)}
        for d in sorted(df["Date"].unique())
    ]

    peak_hour  = int(df.groupby("Hour")["Total"].mean().idxmax())
    busiest_day = str(df.groupby("Day of the week")["Total"].mean().idxmax())

    kpis = {
        "total_vehicles":  int(df["Total"].sum()),
        "avg_per_interval": round(float(df["Total"].mean()), 1),
        "peak_hour":        peak_hour,
        "busiest_day":      busiest_day,
        "records":          len(df),
    }

    return dict(heatmap=heatmap, hourly=hourly, vehicles=vehicles,
                situations=situations, daily=daily, kpis=kpis)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load(DATA_PATH)
    df = clean(df)
    df = engineer(df)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nCleaned dataset → {CLEAN_PATH}")

    agg = aggregate(df)
    AGG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AGG_PATH, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Dashboard JSON  → {AGG_PATH}")

    print("\nDone.")
