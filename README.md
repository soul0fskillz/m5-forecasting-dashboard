<div align="center">

<img src="https://img.shields.io/badge/M5-Demand%20Forecasting-2563eb?style=for-the-badge&logoColor=white" height="40"/>

# M5 Demand Forecasting Dashboard

### AI-powered retail inventory intelligence — predict demand, prevent stockouts, automate reorders

<br/>

[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-2563eb?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Groq](https://img.shields.io/badge/Groq-Qwen3--32B-f97316?style=for-the-badge)](https://groq.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ed?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)

<br/>

> Built in **2 weeks** by **Krishna Sonji** and **Shweta Bankar**

[![GitHub](https://img.shields.io/badge/GitHub-Thir13een-181717?style=flat-square&logo=github)](https://github.com/Thir13een)
[![GitHub](https://img.shields.io/badge/GitHub-shwetabankar54-181717?style=flat-square&logo=github)](https://github.com/shwetabankar54)

</div>

---

<div align="center">

| | |
|:---:|:---:|
| **58 million** training rows | **30,490** SKUs forecasted |
| **4** LightGBM models | **9–12 hrs** total training time |
| **RMSSE 0.60** average accuracy | **28-day** rolling forecast horizon |
| **3** US states · 10 stores | **End-to-end** in 2 weeks |

</div>

---

![Dashboard](assets/dashboard.jpg)

---

## What is this?

This project applies machine learning to the real-world **M5 Forecasting** dataset (Walmart retail sales across California, Texas, and Wisconsin) to build a fully operational inventory management system across 10 stores and 30,490 products.

Given historical sales data, the system:
- Predicts demand for the next **1, 7, 14, and 28 days** per SKU using LightGBM
- Computes **safety stock** buffers based on demand volatility and a 7-day lead time
- Flags items that need **immediate reordering** (CRITICAL) or attention soon (WARNING)
- Answers inventory questions via a **natural language AI assistant** (Qwen3-32B on Groq)
- Accepts new sales CSVs to **re-run the full forecasting pipeline** on demand

---

## Features

| Feature | Description |
|---|---|
| **Inventory Dashboard** | Sortable, filterable grid of all 30,490 SKUs with priority status |
| **Inline Forecast View** | Click any row to see a 28-day demand chart and key metrics inline |
| **AI Inventory Assistant** | Chat with Qwen3-32B, pre-loaded with live inventory context |
| **CSV / Parquet Upload** | Upload new sales data to re-run the full pipeline |
| **Safety Stock Calculation** | Per-SKU reorder quantities accounting for demand volatility |
| **Stockout Timelines** | Days until empty, reorder point, and urgency for every product |

---

## The Model

### Why LightGBM over DeepAR and Transformers?

When approaching demand forecasting at scale, deep learning models like DeepAR, Temporal Fusion Transformer (TFT), and N-BEATS are often the first instinct. They can learn complex temporal patterns but come with real trade-offs at this scale:

| | LightGBM | DeepAR / TFT / N-BEATS |
|---|---|---|
| **GPU required** | No, runs on CPU | Yes, very slow without GPU |
| **Training time** | 9–12 hrs (58M rows, 4 models) | Days at this scale |
| **Inference speed** | Milliseconds | Seconds |
| **Model size** | ~200–325 MB per horizon | Often larger |
| **Tabular features** | Native and excellent | Awkward to integrate |
| **External regressors** | Easy | Complex |
| **Interpretability** | Feature importance built-in | Black box |
| **Retail tabular data** | Consistently strong | Often doesn't outperform GBDT |

The M5 competition validated this. Top-ranked public solutions heavily used gradient boosting (LightGBM / XGBoost), not deep learning. For high-volume retail tabular data with rich feature engineering, tree-based models consistently match or beat transformers at a fraction of the infrastructure cost.

---

### Data Scale and Preprocessing

Before any model could be trained, the raw M5 dataset required extensive preprocessing to build a training-ready feature matrix.

The M5 dataset contains daily sales for 30,490 items across 10 stores over roughly 5.4 years (1,941 days). After melting the wide-format sales matrix into long format, joining calendar features, price data, SNAP flags, and event indicators, and computing all lag and rolling window features per item, the training dataset grew to:

```
30,490 items  ×  ~1,900 training days  =  ~58,000,000 rows
```

Each row carries over 30 features including lags (1–364 days), rolling means and standard deviations across 5 window sizes, cyclical time encodings, price signals, and event flags.

This scale introduced several real engineering challenges:

- **Memory.** The full feature matrix at 58M rows × 30+ float32 features approaches 7 GB. Loading it naively crashes on most machines. We used `.parquet` with selective column loading (`pd.read_parquet(..., columns=[...])`) and `float32` dtypes throughout.
- **Lag computation.** Computing `y_lag_364` (a 364-day lookback) across 30,490 grouped time series requires vectorised `groupby().shift()` operations. A naive loop would run for hours.
- **Lookahead prevention.** All rolling features are computed on `y_lag_1` (the previous day's sales), not the current value, ensuring zero data leakage into any target horizon.
- **Training time.** Each of the 4 LightGBM models trained on the full 58M rows. Total training time across all 4 models was **9–12 hours**.

---

### Direct Multi-Horizon Forecasting with Anchor Points

Training 28 separate models (one per forecast day) would be expensive and hard to maintain. Instead, we trained **4 anchor models** at strategically chosen horizons:

```
Day 1       Day 7       Day 14      Day 28
  │           │           │           │
 h=1         h=7        h=14        h=28
  ●───────────●───────────●───────────●
  ↑           ↑           ↑           ↑
 Model 1    Model 2    Model 3    Model 4
```

Each model is a direct forecaster that predicts the actual sales value at that specific future day (`y_t+h`). The anchor horizons (1, 7, 14, 28) were chosen to align with natural demand cycles: daily fluctuation, weekly rhythm, bi-weekly patterns, and monthly trends.

For the 24 intermediate days between anchors, we do not apply naive linear interpolation on the raw predictions. Raw predictions encode the day-of-week seasonal pattern (Mondays sell differently from Saturdays), so interpolating them directly would mix trend and seasonality and produce distorted forecasts.

Instead we apply a three-step **deseason → interpolate → reseason** procedure using empirically learned per-item weekday weights.

#### Step 1: Learn per-item day-of-week weights

For each item, look back 56 days from the forecast origin and compute mean sales per weekday, then normalise by the item's overall mean:

$$w_{i,k} = \frac{\bar{y}_{i,k}}{\bar{y}_i + \epsilon}, \quad k \in \{0,1,...,6\}$$

where $\bar{y}_{i,k}$ is the mean sales for item $i$ on weekday $k$ over the past 56 days, and $\bar{y}_i$ is the overall mean across all weekdays. $\epsilon = 10^{-6}$ prevents division by zero.

This produces a 7-element shape vector per item representing the relative demand multiplier for each day of week. A FOODS item might have $w_{\text{Saturday}} = 1.8$ and $w_{\text{Tuesday}} = 0.7$, reflecting a weekend spike.

#### Step 2: Deseason the anchor predictions

Each anchor prediction captures both the underlying trend and the seasonality of that specific day. To interpolate only the trend, divide out the weekday effect:

$$\tilde{y}_{i,a} = \frac{\hat{y}_{i,a}}{w_{i,\text{wday}(a)}}$$

where $\hat{y}_{i,a}$ is the model prediction at anchor position $a \in \{0, 6, 13, 27\}$ (days 1, 7, 14, 28), and $w_{i,\text{wday}(a)}$ is the weekday weight for the day of week that anchor falls on. This gives $\tilde{y}_{i,a}$, the deseasoned base trend at each anchor.

#### Step 3: Interpolate the base trend

With seasonality removed, the base trend between adjacent anchors $a$ and $b$ is interpolated linearly:

$$\tilde{y}_{i,d} = \tilde{y}_{i,a}\,(1 - r) + \tilde{y}_{i,b}\,r, \qquad r = \frac{d - a}{b - a}$$

for every intermediate day $d \in [a, b]$, applied across three segments:

| Segment | Anchor positions (0-indexed) |
|---|---|
| Days 1 → 7 | $a = 0,\ b = 6$ |
| Days 7 → 14 | $a = 6,\ b = 13$ |
| Days 14 → 28 | $a = 13,\ b = 27$ |

#### Step 4: Reseason the interpolated values

Multiply each day's deseasoned estimate back by its own weekday weight:

$$\hat{y}_{i,d} = \tilde{y}_{i,d} \times w_{i,\text{wday}(d)}$$

This reapplies the correct seasonal multiplier for each future date, giving a forecast that respects both the trend trajectory and the weekly demand rhythm of each item. For anchor days, deseasoning and reseasoning cancel out so the final value always equals the original model prediction.

```
                   anchor             interpolated (base trend × weekday weight)
                     ↓                          ↓
Day  1 (Mon):  0.94               (h=1 anchor, returns exactly to model prediction)
Day  2 (Tue):  base_trend × 0.65  (slow day, weight pulls forecast down)
Day  3 (Wed):  base_trend × 0.72
Day  4 (Thu):  base_trend × 0.75
Day  5 (Fri):  base_trend × 1.20  (picks up toward weekend)
Day  6 (Sat):  base_trend × 1.85  (weekend spike fully restored by weight)
Day  7 (Sun):  1.21               (h=7 anchor, returns exactly to model prediction)
...
Day 14:        1.08               (h=14 anchor)
...
Day 28:        1.17               (h=28 anchor)
```

---

### Model Performance

RMSSE (Root Mean Squared Scaled Error) measures accuracy relative to a naive baseline. A score below 1.0 means the model beats the naive forecast.

| Horizon | RMSSE |
|---|---|
| Day 1 | **0.52** |
| Day 7 | **0.63** |
| Day 14 | **0.61** |
| Day 28 | **0.66** |
| **Average (4 anchors)** | **0.60** |

The 4 anchor points individually achieve strong RMSSE (0.52–0.66). Intermediate days are not separately evaluated since they come from the interpolation procedure rather than direct model predictions.

---

### Feature Engineering

Each model is trained on a rich set of features computed from historical sales:

**Lag features** capture autocorrelation at different time scales:

| Lag | Captures |
|---|---|
| 1 day | Yesterday's sales |
| 7 days | Same day last week |
| 14 days | Two weeks ago |
| 28 days | Four weeks ago |
| 56, 84 days | 2–3 month patterns |
| 182 days | Half-year seasonality |
| 364 days | Year-over-year patterns |

**Rolling window features** are computed on `y_lag_1` to prevent data leakage:

| Feature | Windows |
|---|---|
| Rolling mean | 7, 28, 56, 182, 364 days |
| Rolling std | 28, 56 days |
| Rolling median | 28 days |
| Zero-sale rate | 28 days |
| Non-zero sale count | 28 days |
| Days since last sale | N/A |

**Calendar features** use sin/cos cyclical encodings to avoid ordinal artifacts:
- Day of week: `wday_sin`, `wday_cos`
- Month: `month_sin`, `month_cos`

**External features:**
- SNAP days (US government food assistance program, drives FOODS category spikes)
- Event type: Cultural, National, Religious, Sporting, None

**Inventory calculations applied post-inference:**
```
safety_stock    = Z × σ_lead_time           (Z = 1.282, 90% service level)
sigma_lead_time = σ_daily × √(lead_time)    (lead_time = 7 days)
reorder_point   = demand_lead_time + safety_stock
order_qty       = max(0, demand_28d + safety_stock − current_stock)
```

---

## How to Use the App

### Dashboard Overview

Open the app at `http://localhost:3000` to land on the Inventory Dashboard:

```
┌─────────────────────────────────────────────────────┬──────────────────────┐
│  Never run out. Always know what to order.          │   [Upload CSV]       │
│  AI-powered forecasts across 30,490 products        │                      │
│  Viewing: Mon, Mar 2, 2026  |  Forecast: 28 days    │                      │
├─────────────────────────────────────────────────────┴──────────────────────┤
│  [!] 4,231 CRITICAL items need immediate reordering                        │
├────────────┬────────────┬────────────┬────────────┬────────────────────────┤
│ Total SKUs │  Critical  │  Warning   │    OK      │   Total Units to Order │
├────────────┴────────────┴────────────┴────────────┴────────────────────────┤
│  [Filters: Store | Category | Status | Search]  [Sort by: Units to Order] │
├──────────┬────────────┬──────────┬──────────┬────────┬──────────┬──────────┤
│ Item ID  │  Store     │ Category │ Dept     │ Status │ Order Qty│Days Left │
│ (click any row to expand)                                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

### Filtering and Sorting

Use the filter bar to narrow down items:
- **Store** — CA-1, CA-2, CA-3, CA-4, TX-1, TX-2, TX-3, WI-1, WI-2, WI-3
- **Category** — FOODS, HOBBIES, HOUSEHOLD
- **Status** — CRITICAL (order now), WARNING (order soon), OK (well stocked)
- **Search** — type any item ID prefix to find specific SKUs
- **Sort** — by Units to Order, Days Until Stockout, or Item ID

### Inline Row Expansion

Click any row to expand it inline and see the full forecast:

```
┌─────────────────────────────────────────────────────────────────────┐
│  FOODS_1_001_CA_1  ● CRITICAL  CA-1 · FOODS                   [✕] │
├──────────────────────────────────┬──────────────────────────────────┤
│                                  │  Avg Daily Sales    0.04/day     │
│   [28-day demand bar chart]      │  28-day Demand      1.17 units   │
│   Blue bars = forecast           │  Safety Buffer      0.09 units   │
│   Red line  = reorder point      │  Reorder Point      0.37 units   │
│                                  │  Stock on Hand      0.59 units   │
│                                  │  Days Until Empty   14.0 days    │
├──────────────────────────────────┴──────────────────────────────────┤
│  ● Recommended Order: 1 unit                                        │
└─────────────────────────────────────────────────────────────────────┘
```

Click the same row again (or the ✕) to collapse. Clicking a different row switches to that item.

### AI Chat Assistant

The right panel hosts an AI assistant pre-loaded with your live inventory data. Try asking:

> *"What are the top 10 most urgent items to order?"*
> *"Which store needs the most attention right now?"*
> *"Break down critical items by category"*
> *"Which items will stock out within 3 days?"*

The assistant has access to store breakdowns, category summaries, and top critical items, so answers are specific to your actual stock state.

### Uploading New Sales Data

Click **Upload sales data** in the top-right corner to re-run the pipeline with fresh data. The backend will:
1. Parse and validate your CSV
2. Run feature engineering (lags, rolling windows, calendar features)
3. Run all 4 LightGBM models
4. Recompute inventory metrics and priorities
5. Refresh the dashboard automatically

Upload takes **30–60 seconds** depending on the number of rows.

---

## CSV File Format

The upload endpoint accepts `.csv` or `.parquet` files. Your file must contain these columns:

| Column | Type | Description | Example |
|---|---|---|---|
| `id` | string | Item + Store identifier | `FOODS_1_001_CA_1` |
| `d` | string | Day index (M5 format) | `d_1914` |
| `sales` | float | Units sold that day | `2.0` |
| `sell_price` | float | Selling price | `1.99` |
| `sell_price_isna` | int (0/1) | 1 if price is missing | `0` |
| `snap` | int (0/1) | SNAP benefit day flag | `1` |
| `wday_sin` | float | Cyclical day-of-week (sin) | `0.782` |
| `wday_cos` | float | Cyclical day-of-week (cos) | `0.623` |
| `month_sin` | float | Cyclical month (sin) | `0.500` |
| `month_cos` | float | Cyclical month (cos) | `0.866` |
| `is_event` | int (0/1) | Any event this day | `0` |
| `event_cultural` | int (0/1) | Cultural event flag | `0` |
| `event_national` | int (0/1) | National event flag | `0` |
| `event_none` | int (0/1) | No event (1 = no event) | `1` |
| `event_religious` | int (0/1) | Religious event flag | `0` |
| `event_sporting` | int (0/1) | Sporting event flag | `0` |

**Example rows:**

```csv
id,d,sales,sell_price,sell_price_isna,snap,wday_sin,wday_cos,month_sin,month_cos,is_event,event_cultural,event_national,event_none,event_religious,event_sporting
FOODS_1_001_CA_1,d_1913,1.0,1.99,0,0,0.782,0.623,0.5,0.866,0,0,0,1,0,0
FOODS_1_001_CA_1,d_1914,2.0,1.99,0,1,0.975,0.223,0.5,0.866,0,0,0,1,0,0
HOBBIES_1_001_CA_1,d_1913,0.0,4.49,0,0,0.782,0.623,0.5,0.866,0,0,0,1,0,0
```

**Cyclical encoding formulas:**

```python
import numpy as np

# Day of week (0=Monday, 6=Sunday)
wday_sin = np.sin(2 * np.pi * weekday / 7)
wday_cos = np.cos(2 * np.pi * weekday / 7)

# Month (1=January, 12=December)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
```

Include at least the **last 364 days** of data per item for full lag coverage. Shorter histories still work (the pipeline uses `min_periods=1`) but lag features will be `NaN` for early rows, which reduces forecast accuracy.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Thir13een/m5-forecasting-dashboard.git
cd m5-forecasting-dashboard
```

### 2. Download the pre-trained models

Download all 5 files from the [v1.0 release](https://github.com/Thir13een/m5-forecasting-dashboard/releases/tag/v1.0) into a single local folder:

| File | Size | Link |
|---|---|---|
| `lgb_direct_h01.txt` | 195 MB | [Download](https://github.com/Thir13een/m5-forecasting-dashboard/releases/download/v1.0/lgb_direct_h01.txt) |
| `lgb_direct_h07.txt` | 304 MB | [Download](https://github.com/Thir13een/m5-forecasting-dashboard/releases/download/v1.0/lgb_direct_h07.txt) |
| `lgb_direct_h14.txt` | 325 MB | [Download](https://github.com/Thir13een/m5-forecasting-dashboard/releases/download/v1.0/lgb_direct_h14.txt) |
| `lgb_direct_h28.txt` | 320 MB | [Download](https://github.com/Thir13een/m5-forecasting-dashboard/releases/download/v1.0/lgb_direct_h28.txt) |
| `feature_cols.csv` | 1 KB | [Download](https://github.com/Thir13een/m5-forecasting-dashboard/releases/download/v1.0/feature_cols.csv) |

### 3. Set up environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
# Absolute path to the folder containing your downloaded model files
MODELS_PATH=/path/to/your/models/folder

# Free API key from https://console.groq.com
GROQ_API_KEY=gsk_your_key_here

# Leave as-is for local Docker setup
DATABASE_URL=postgresql://m5:m5pass@postgres:5432/m5db
```

### 4. Run with Docker

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| **Dashboard** | http://localhost:3000 |
| **API docs** | http://localhost:8000/docs |

First startup takes 2–3 minutes while Docker builds the images and loads all 4 LightGBM models into memory.

---

## Project Structure

```
m5-forecasting-dashboard/
│
├── backend/                        # FastAPI application
│   ├── app/
│   │   ├── main.py                 # App entrypoint, model loading on startup
│   │   ├── config.py               # Environment variable config
│   │   ├── state.py                # In-memory inventory/forecast state
│   │   ├── routers/
│   │   │   ├── inventory.py        # GET /inventory — filterable, paginated SKU list
│   │   │   ├── forecast.py         # GET /forecast/{item_id} — 28-day chart data
│   │   │   ├── upload.py           # POST /upload — CSV ingestion + pipeline run
│   │   │   ├── chat.py             # POST /chat/stream — SSE streaming chat
│   │   │   └── health.py           # GET /health — status + last_updated timestamp
│   │   └── services/
│   │       ├── model_store.py      # LightGBM model loader + predict()
│   │       ├── feature_engineering.py  # Lags, rolling windows, calendar features
│   │       ├── inference.py        # Run pipeline → forecast + inventory DataFrames
│   │       ├── chat_service.py     # Groq streaming + system prompt builder
│   │       └── demo_loader.py      # Load pre-computed CSVs on cold start
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                       # Next.js 14 application
│   ├── app/
│   │   ├── layout.tsx              # Root layout — header + chat panel sidebar
│   │   ├── globals.css             # Global styles + keyframes
│   │   ├── page.tsx                # Redirect to /inventory
│   │   └── inventory/
│   │       └── page.tsx            # Main inventory page
│   ├── components/
│   │   ├── InventoryGrid.tsx       # Table + inline row expansion + ExpandedRowPanel
│   │   ├── DemandChart.tsx         # 28-day bar chart (recharts)
│   │   ├── ChatWindow.tsx          # SSE streaming chat UI
│   │   ├── FileUpload.tsx          # Drag-and-drop CSV uploader
│   │   ├── InventoryFilters.tsx    # Filter bar (store, category, status, search)
│   │   ├── StatsCards.tsx          # Summary stat cards (total, critical, warning, OK)
│   │   ├── AlertBanner.tsx         # Red/yellow banner for critical/warning counts
│   │   └── ScrollArea.tsx          # Custom scrollable container
│   ├── lib/
│   │   ├── api.ts                  # Typed API client (fetch + SSE streaming)
│   │   ├── types.ts                # TypeScript interfaces
│   │   └── utils.ts                # Shared helpers
│   ├── next.config.mjs
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   └── Dockerfile
│
├── infrastructure/
│   └── init.sql                    # PostgreSQL schema (pipeline_runs table)
│
├── docker-compose.yml              # Orchestrates postgres + backend + frontend
├── .env.example                    # Environment variable template
└── README.md
```

---

## Engineering Highlights

This project covers the full ML engineering stack, not just model training.

**Scale.** 58 million training rows across 30,490 time series. Handling this requires deliberate choices around memory management, vectorised operations, and file formats (Parquet over CSV, float32 over float64). Most ML projects don't operate at this row count.

**No data leakage.** Every rolling feature is computed on `y_lag_1` (the previous day's sales), not the current day. This is a common mistake in time series ML that inflates validation scores. We enforced strict temporal discipline throughout the pipeline.

**Efficient multi-horizon design.** Instead of training 28 models, we designed a 4-anchor system using a seasonality-aware deseason → interpolate → reseason procedure. This gave a 7× reduction in training compute while preserving accuracy at key business horizons (day 1, week 1, week 2, month 1).

**Business logic on top of ML.** Raw model predictions alone aren't useful to an operations team. We layered safety stock calculations (90% service level, square-root lead-time scaling), reorder point formulas, and priority classification (CRITICAL / WARNING / OK) on top of the model outputs to produce actionable recommendations.

**Full-stack deployment.** The model isn't a notebook. It's served via a FastAPI backend with SSE streaming, persisted in PostgreSQL, and presented through a Next.js dashboard with real-time AI chat. Every component runs in Docker Compose for one-command deployment.

**LLM context injection.** The AI assistant's system prompt is dynamically built from live inventory data including store breakdowns, category summaries, and top critical items. It answers questions about your actual stock state, not generic retail knowledge.

### Skills Demonstrated

| Area | What's covered |
|---|---|
| **Machine Learning** | LightGBM, direct multi-step forecasting, time series feature engineering, RMSSE evaluation, safety stock modelling |
| **Data Engineering** | 58M-row Parquet pipelines, grouped lag/rolling transforms, lookahead prevention, memory-efficient dtypes |
| **Backend** | FastAPI, async Python, SSE streaming, file upload pipeline, PostgreSQL with SQLAlchemy |
| **Frontend** | Next.js 14 App Router, TypeScript, React state management, SSE client, inline data visualisation |
| **AI/LLM** | Groq API, dynamic system prompt construction, streaming token output, reasoning-block filtering |
| **DevOps** | Docker Compose multi-service orchestration, environment variable management, model artifact hosting |

---

## Tech Stack

### Machine Learning
[![LightGBM](https://img.shields.io/badge/LightGBM-Direct%20Multi--Horizon-2563eb?style=flat-square)](https://lightgbm.readthedocs.io/)
[![pandas](https://img.shields.io/badge/pandas-Feature%20Engineering-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![numpy](https://img.shields.io/badge/numpy-Numerical%20Ops-013243?style=flat-square&logo=numpy)](https://numpy.org/)

### Backend
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20%2B%20SSE-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Pipeline%20Logs-336791?style=flat-square&logo=postgresql)](https://www.postgresql.org/)
[![Groq](https://img.shields.io/badge/Groq-Qwen3--32B%20Chat-f97316?style=flat-square)](https://groq.com/)

### Frontend
[![Next.js](https://img.shields.io/badge/Next.js-14%20App%20Router-000000?style=flat-square&logo=next.js)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6?style=flat-square&logo=typescript)](https://www.typescriptlang.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952b3?style=flat-square&logo=bootstrap)](https://getbootstrap.com/)
[![ReactMarkdown](https://img.shields.io/badge/ReactMarkdown-Chat%20Rendering-61dafb?style=flat-square&logo=react)](https://github.com/remarkjs/react-markdown)

### Infrastructure
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ed?style=flat-square&logo=docker)](https://www.docker.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Releases%20(Models)-181717?style=flat-square&logo=github)](https://github.com/Thir13een/m5-forecasting-dashboard/releases)

---

<div align="center">

Built in **2 weeks** by

**Krishna Sonji** &nbsp;·&nbsp; **Shweta Bankar**

[![GitHub](https://img.shields.io/badge/GitHub-Thir13een-181717?style=flat-square&logo=github)](https://github.com/Thir13een)
&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-shwetabankar54-181717?style=flat-square&logo=github)](https://github.com/shwetabankar54)

</div>
