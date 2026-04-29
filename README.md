# Traffic EDA Dashboard

> Exploratory data analysis and traffic situation classification for urban vehicle count data collected via computer vision — 5,952 fifteen-minute intervals across two months.

![Dashboard preview](assets/plots/hourly_avg.png)

---

## What's in this repo

```
traffic-prediction/
├── data/
│   └── TrafficTwoMonth.csv          # Raw dataset (4 vehicle classes, 15-min cadence)
├── src/
│   ├── preprocess.py                # Cleaning + feature engineering → data/traffic_clean.csv
│   ├── eda.py                       # EDA plots saved to assets/plots/
│   ├── train.py                     # Trains classifier → model.pkl
│   └── predict.py                   # CLI prediction tool
├── assets/
│   ├── style.css                    # Dashboard stylesheet
│   ├── dashboard.js                 # Chart rendering logic
│   └── traffic_agg.json             # Pre-aggregated data for the dashboard
├── index.html                       # Interactive EDA dashboard (open in browser)
├── app.py                           # FastAPI prediction server
├── requirements.txt
└── README.md
```

---

## Dataset

| Column | Description |
|--------|-------------|
| `Time` | 15-minute interval start time |
| `Date` | Day of month (1–31) |
| `Day of the week` | Monday–Sunday |
| `CarCount` | Cars detected in interval |
| `BikeCount` | Bikes detected |
| `BusCount` | Buses detected |
| `TruckCount` | Trucks detected |
| `Total` | Sum of all vehicle counts |
| `Traffic Situation` | `heavy` / `high` / `normal` / `low` |

Collected by a roadside computer vision model. ~5,952 rows spanning ~2 months.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. View the dashboard

Just open `index.html` in any browser — no server needed.  
The dashboard reads from the pre-built `assets/traffic_agg.json`.

```bash
open index.html          # macOS
start index.html         # Windows
xdg-open index.html      # Linux
```

### 3. Run the EDA pipeline

```bash
# Clean data and regenerate the dashboard JSON
python src/preprocess.py

# Generate all EDA plots → assets/plots/
python src/eda.py
```

### 4. Train the classifier

```bash
python src/train.py
```

Trains three candidates (Random Forest, Gradient Boosting, Logistic Regression) with 5-fold cross-validation, picks the best, and saves it to `model.pkl`. A report is written to `assets/model_report.txt`.

### 5. Make a prediction (CLI)

```bash
# Interactive mode
python src/predict.py

# One-liner
python src/predict.py --hour 8 --day Monday --car 40 --bike 5 --bus 3 --truck 10
```

### 6. Run the API server

```bash
uvicorn app:app --reload
```

API docs → [http://localhost:8000/docs](http://localhost:8000/docs)

```
POST /predict
{
  "hour": 8,
  "minute": 0,
  "day": "Monday",
  "car_count": 40,
  "bike_count": 5,
  "bus_count": 3,
  "truck_count": 10
}
```

---

## Dashboard panels

| Panel | Description |
|-------|-------------|
| **KPI strip** | Total vehicles, avg per interval, peak hour, heavy-traffic % |
| **Hourly line chart** | Avg vehicles by hour — filterable by day of week |
| **Day × hour heatmap** | 7×24 grid coloured by avg vehicle count |
| **Traffic situation bars** | Heavy / High / Normal / Low distribution |
| **Vehicle mix donut** | Cars / Bikes / Buses / Trucks proportion |
| **Daily trend bars** | Average per interval for each calendar date |
| **Insights** | Key findings pulled from the EDA |

---

## Key findings

- **Double rush peaks** — 6–8 AM (avg ~164) and 4–5 PM (avg ~177). Overnight (10 PM–4 AM) drops to ~42.
- **Friday anomaly** — peak shifts to midday (9 AM–2 PM, avg ~207) instead of the evening.
- **Cars dominate** at 60%; trucks are second at 17%, higher than buses (12%) and bikes (11%) combined.
- **61% normal, 19% heavy** — congestion concentrated in rush windows; low conditions mainly overnight.

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Data processing | `pandas`, `numpy` |
| EDA plots | `matplotlib`, `seaborn` |
| ML | `scikit-learn` |
| Dashboard | Vanilla JS + [Chart.js](https://www.chartjs.org/) |
| API | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |

---
