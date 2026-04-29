"""
eda.py — Exploratory Data Analysis for TrafficTwoMonth dataset
Saves all plots to assets/plots/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from pathlib import Path

DATA_PATH  = Path("data/TrafficTwoMonth.csv")
OUT_PATH   = Path("assets/plots")
OUT_PATH.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#e8e8e4",
    "grid.linewidth":   0.5,
    "figure.dpi":       140,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})

PALETTE   = ["#378ADD","#1D9E75","#BA7517","#D85A30"]
SIT_PAL   = {"heavy":"#E24B4A","high":"#EF9F27","normal":"#378ADD","low":"#1D9E75"}
DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# ── Load & clean ──────────────────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Time"] = pd.to_datetime(df["Time"], format="%I:%M:%S %p")
    df["Hour"] = df["Time"].dt.hour
    df["Day of the week"] = pd.Categorical(df["Day of the week"], categories=DAY_ORDER, ordered=True)
    df["Traffic Situation"] = df["Traffic Situation"].str.strip().str.lower()
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_hourly_avg(df):
    hourly = df.groupby("Hour")["Total"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(hourly.index, hourly.values, alpha=0.15, color="#378ADD")
    ax.plot(hourly.index, hourly.values, color="#378ADD", lw=2)
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=7, rotation=45)
    ax.set_xlabel("Hour of day", fontsize=9)
    ax.set_ylabel("Avg vehicles / 15 min", fontsize=9)
    ax.set_title("Hourly average traffic volume", fontsize=11, pad=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{int(x)}"))
    fig.savefig(OUT_PATH / "hourly_avg.png")
    plt.close(fig)
    print("  saved hourly_avg.png")


def plot_heatmap(df):
    pivot = df.pivot_table(index="Day of the week", columns="Hour", values="Total", aggfunc="mean")
    pivot = pivot.reindex(DAY_ORDER)
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd",
        linewidths=0.3, linecolor="#f0f0ec",
        cbar_kws={"label":"Avg vehicles", "shrink":0.7},
        fmt=".0f", annot=False,
    )
    ax.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=7, rotation=45)
    ax.set_yticklabels(DAY_ORDER, fontsize=8, rotation=0)
    ax.set_xlabel("Hour of day", fontsize=9)
    ax.set_ylabel("")
    ax.set_title("Day × hour heatmap — avg vehicle count", fontsize=11, pad=10)
    ax.grid(False)
    fig.savefig(OUT_PATH / "heatmap_day_hour.png")
    plt.close(fig)
    print("  saved heatmap_day_hour.png")


def plot_vehicle_mix(df):
    totals = {
        "Cars":   df["CarCount"].sum(),
        "Bikes":  df["BikeCount"].sum(),
        "Buses":  df["BusCount"].sum(),
        "Trucks": df["TruckCount"].sum(),
    }
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        totals.values(), labels=totals.keys(),
        colors=PALETTE, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"linewidth":1.5,"edgecolor":"white"},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
    # donut
    centre = plt.Circle((0,0), 0.5, color="white")
    ax.add_artist(centre)
    ax.set_title("Vehicle type distribution", fontsize=11, pad=6)
    fig.savefig(OUT_PATH / "vehicle_mix.png")
    plt.close(fig)
    print("  saved vehicle_mix.png")


def plot_situation_dist(df):
    order  = ["heavy","high","normal","low"]
    counts = df["Traffic Situation"].value_counts().reindex(order).fillna(0)
    colors = [SIT_PAL[s] for s in order]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(order, counts.values, color=colors, height=0.55)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                f"{int(val):,}", va="center", fontsize=9)
    ax.set_xlabel("Number of 15-min intervals", fontsize=9)
    ax.set_title("Traffic situation distribution", fontsize=11, pad=10)
    ax.set_xlim(0, counts.max() * 1.18)
    fig.savefig(OUT_PATH / "situation_dist.png")
    plt.close(fig)
    print("  saved situation_dist.png")


def plot_daily_trend(df):
    daily = df.groupby("Date")["Total"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(daily["Date"], daily["Total"], color="#378ADD", alpha=0.6, width=0.7)
    ax.plot(daily["Date"], daily["Total"].rolling(3, center=True).mean(),
            color="#185FA5", lw=2, label="3-day rolling avg")
    ax.set_xlabel("Day of month", fontsize=9)
    ax.set_ylabel("Avg vehicles / 15 min", fontsize=9)
    ax.set_title("Daily average traffic volume", fontsize=11, pad=10)
    ax.legend(fontsize=8)
    fig.savefig(OUT_PATH / "daily_trend.png")
    plt.close(fig)
    print("  saved daily_trend.png")


def plot_boxplot_by_day(df):
    fig, ax = plt.subplots(figsize=(9, 4))
    data_by_day = [df[df["Day of the week"]==d]["Total"].values for d in DAY_ORDER]
    bp = ax.boxplot(data_by_day, patch_artist=True, medianprops={"color":"white","lw":2})
    for patch, color in zip(bp["boxes"], PALETTE + PALETTE[:3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels([d[:3] for d in DAY_ORDER], fontsize=9)
    ax.set_ylabel("Total vehicles / 15 min", fontsize=9)
    ax.set_title("Traffic volume distribution by day of week", fontsize=11, pad=10)
    fig.savefig(OUT_PATH / "boxplot_by_day.png")
    plt.close(fig)
    print("  saved boxplot_by_day.png")


def plot_correlation(df):
    cols = ["CarCount","BikeCount","BusCount","TruckCount","Total"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                cbar_kws={"shrink":0.8})
    ax.set_title("Vehicle type correlation matrix", fontsize=11, pad=10)
    ax.grid(False)
    fig.savefig(OUT_PATH / "correlation.png")
    plt.close(fig)
    print("  saved correlation.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"  {len(df):,} rows loaded\n")

    print("Generating EDA plots...")
    plot_hourly_avg(df)
    plot_heatmap(df)
    plot_vehicle_mix(df)
    plot_situation_dist(df)
    plot_daily_trend(df)
    plot_boxplot_by_day(df)
    plot_correlation(df)

    print(f"\nAll plots saved to {OUT_PATH}/")

    # Basic stats summary
    print("\n── Quick stats ──────────────────────────────")
    print(df[["CarCount","BikeCount","BusCount","TruckCount","Total"]].describe().round(1).to_string())
    print("\nTraffic situation counts:")
    print(df["Traffic Situation"].value_counts().to_string())
