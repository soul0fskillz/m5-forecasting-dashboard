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

## What is this?

This project applies machine learning to the real-world **M5 Forecasting** dataset — Walmart retail sales data spanning **10 stores** across California, Texas, and Wisconsin — to build a fully operational inventory management system.

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
| **Inline Forecast View** | Click any row — 28-day demand chart + key metrics expand inline |
| **AI Inventory Assistant** | Chat with Qwen3-32B, pre-loaded with live inventory context |
| **CSV / Parquet Upload** | Upload new sales data to re-run the full pipeline |
| **Safety Stock Calculation** | Per-SKU reorder quantities accounting for demand volatility |
| **Stockout Timelines** | Days until empty, reorder point, and urgency for every product |

---

## The Model — Why LightGBM?

### LightGBM vs. DeepAR / Transformers

When approaching demand forecasting at scale, there are two common paths:

**Deep learning models** like DeepAR, Temporal Fusion Transformer (TFT), and N-BEATS are often the first instinct. They can learn complex temporal patterns and handle uncertainty natively. But they come with significant trade-offs:

| | LightGBM | DeepAR / TFT / N-BEATS |
|---|---|---|
| **GPU required** | No — runs on CPU | Yes — slow without GPU |
| **Training time** | 9–12 hrs (58M rows, 4 models) | Days at this scale |
| **Inference speed** | Milliseconds | Seconds |
| **Model size** | ~200–325 MB per horizon | Often larger |
| **Tabular features** | Excellent — native | Awkward to integrate |
| **External regressors** | Easy | Complex |
| **Interpretability** | Feature importance built-in | Black box |
| **Retail tabular data** | Consistently strong | Often doesn't outperform GBDT |

The M5 competition itself validated this — the top-ranked public solutions heavily used **gradient boosting (LightGBM / XGBoost)**, not deep learning. For high-volume, high-dimensionality retail tabular data with rich feature engineering, tree-based models consistently match or beat transformers — with a fraction of the infrastructure cost.

Our choice: **LightGBM with a direct multi-horizon strategy**.

---

### Data Scale & Preprocessing

Before a single model could be trained, the raw M5 dataset required extensive preprocessing to construct a training-ready feature matrix.

The M5 dataset contains daily sales for 30,490 items across 10 stores over ~5.4 years (1,941 days). After melting the wide-format sales matrix into long format, joining calendar features, price data, SNAP flags, and event indicators, and then computing all lag and rolling window features **per item**, the resulting training dataset grew to:

```
30,490 items  ×  ~1,900 training days  =  ~58,000,000 rows
```

Each row carries over 30 features — lags (1–364 days), rolling means and standard deviations across 5 window sizes, cyclical time encodings, price signals, and event flags.

**This scale introduced several engineering challenges:**

- **Memory** — the full feature matrix at 58M rows × 30+ float32 features approaches ~7 GB. Loading it naively crashes on most machines. We used `.parquet` with selective column loading (`pd.read_parquet(..., columns=[...])`) and `float32` dtypes to keep memory under control.
- **Lag computation** — computing `y_lag_364` (a 364-day lookback) across 30,490 grouped time series requires careful `groupby().shift()` operations. A naive loop would take hours; vectorised pandas operations kept it feasible.
- **Lookahead prevention** — all rolling features are computed on `y_lag_1` (yesterday's sales), not the current day's value, ensuring zero data leakage into any target horizon.
- **Training time** — each of the 4 LightGBM models trained on 58M rows with thousands of trees. Total training time across all 4 models: **9–12 hours**.

---

### Direct Multi-Horizon Forecasting with Anchor Points

The naive approach to 28-day forecasting is to train **28 separate models** — one per forecast day. That's expensive, slow to train, and hard to maintain.

Instead, we trained **4 anchor models** at strategically chosen horizons:

```
Day 1       Day 7       Day 14      Day 28
  │           │           │           │
 h=1         h=7        h=14        h=28
  ●───────────●───────────●───────────●
  ↑           ↑           ↑           ↑
 Model 1    Model 2    Model 3    Model 4
```

Each model is a **direct forecaster** — it predicts the actual sales value at that specific future day (`y_t+h`), not a cumulative total. The anchor horizons (1, 7, 14, 28) were chosen to align with natural demand cycles: daily fluctuation, weekly rhythm, bi-weekly patterns, and monthly trends.

**For the 24 intermediate days** that lie between anchors, we apply **linear interpolation** between the adjacent anchor predictions:

```
Days 2–6  →  lerp(pred_h01, pred_h07)   — between day-1 and week-1 anchors
Days 8–13 →  lerp(pred_h07, pred_h14)   — between week-1 and week-2 anchors
Days 15–27→  lerp(pred_h14, pred_h28)   — between week-2 and month anchors
```

For example, for a given SKU with predictions `h1=0.94`, `h7=1.21`, `h14=1.08`, `h28=1.17`:

```
Day  1:  0.94   (anchor)
Day  2:  0.98   (lerp)
Day  3:  1.03   (lerp)
Day  4:  1.07   (lerp)
Day  5:  1.12   (lerp)
Day  6:  1.16   (lerp)
Day  7:  1.21   (anchor)
Day  8:  1.19   (lerp)
...
Day 14:  1.08   (anchor)
...
Day 28:  1.17   (anchor)
```

This gives a smooth demand curve across all 28 days using only 4 trained models — a clean trade-off between model count and forecast granularity.

---

### Model Performance

RMSSE (Root Mean Squared Scaled Error) measures forecast accuracy relative to a naive baseline. A score below 1.0 means the model beats the naive forecast.

| Horizon | RMSSE |
|---|---|
| Day 1 | **0.52** |
| Day 7 | **0.63** |
| Day 14 | **0.61** |
| Day 28 | **0.66** |
| **Average (4 anchors)** | **0.60** |

The anchor points individually achieve strong RMSSE (0.52–0.66). The interpolated intermediate days are not separately evaluated — their error depends on how linearly demand evolves between anchors. In practice, this is a reasonable assumption for weekly retail patterns.

---

### Feature Engineering

Each model is trained on a rich set of features computed from historical sales:

**Lag features** — direct past sales values used to capture autocorrelation:

| Lag | Captures |
|---|---|
| 1 day | Yesterday's sales |
| 7 days | Same day last week |
| 14 days | Two weeks ago |
| 28 days | Four weeks ago |
| 56, 84 days | 2–3 month patterns |
| 182 days | Half-year seasonality |
| 364 days | Year-over-year patterns |

**Rolling window features** — computed on `y_lag_1` to prevent data leakage:

| Feature | Windows |
|---|---|
| Rolling mean | 7, 28, 56, 182, 364 days |
| Rolling std | 28, 56 days |
| Rolling median | 28 days |
| Zero-sale rate | 28 days |
| Non-zero sale count | 28 days |
| Days since last sale | — |

**Calendar features** — cyclical encodings (sin/cos) to avoid ordinal artifacts:
- Day of week: `wday_sin`, `wday_cos`
- Month: `month_sin`, `month_cos`

**Event and external features:**
- SNAP days (US government food assistance — drives FOODS spikes)
- Event type: Cultural, National, Religious, Sporting, None

**Inventory calculations** (post-inference):
```
safety_stock    = Z × σ_lead_time           (Z = 1.282, 90% service level)
sigma_lead_time = σ_daily × √(lead_time)    (lead_time = 7 days)
reorder_point   = demand_lead_time + safety_stock
order_qty       = max(0, demand_28d + safety_stock − current_stock)
```

---

## How to Use the App

### Dashboard Overview

When you open the app at `http://localhost:3000`, you land on the **Inventory Dashboard**:

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
│ (click to expand)                                                          │
└──────────────────────────────────────────────────────────────────────────-─┘
```

### Filtering and Sorting

Use the filter bar to narrow down items:
- **Store** — select one or more stores (CA-1, CA-2, CA-3, CA-4, TX-1, TX-2, TX-3, WI-1, WI-2, WI-3)
- **Category** — FOODS, HOBBIES, HOUSEHOLD
- **Status** — CRITICAL (order now), WARNING (order soon), OK (well stocked)
- **Search** — type an item ID prefix to find specific SKUs
- **Sort** — by Units to Order, Days Until Stockout, or Item ID

### Inline Row Expansion

Click any row to expand it inline:

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

- Click the same row again (or the ✕) to collapse
- Click a different row to switch

### AI Chat Assistant

The right panel hosts an AI assistant pre-loaded with your live inventory data. You can ask:

> *"What are the top 10 most urgent items to order?"*
> *"Which store needs the most attention right now?"*
> *"Break down critical items by category"*
> *"Which items will stock out within 3 days?"*

The assistant uses the full inventory context — store breakdowns, category summaries, top critical items — to give accurate, structured answers with tables.

### Uploading New Sales Data

Click **Upload sales data** in the top-right corner to re-run the full forecasting pipeline with fresh data. The backend will:
1. Parse and validate your CSV
2. Run feature engineering (lags, rolling windows, calendar features)
3. Run all 4 LightGBM models
4. Recompute inventory metrics and priorities
5. Refresh the dashboard automatically

Upload takes **30–60 seconds** depending on the number of rows.

---

## CSV File Format

The upload endpoint accepts `.csv` or `.parquet` files. Your file **must contain these columns**:

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

# Day of week (0=Monday … 6=Sunday)
wday_sin = np.sin(2 * np.pi * weekday / 7)
wday_cos = np.cos(2 * np.pi * weekday / 7)

# Month (1=January … 12=December)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
```

**Tip:** Include at least the **last 364 days** of data per item for full lag coverage. The feature engineering pipeline uses `min_periods=1` so shorter histories will still work, but lag features will be `NaN` for early rows — this can reduce forecast accuracy for items with limited history.

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

First startup takes ~2–3 minutes while Docker builds the images and loads all 4 LightGBM models into memory.

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
│   │   ├── page.tsx                # Redirect → /inventory
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

This project was deliberately built to demonstrate end-to-end ML engineering — not just model training, but the full journey from raw data to a production-ready application.

### What makes this non-trivial

**Scale** — 58 million training rows across 30,490 time series. Most ML tutorials work with thousands of rows. Handling 58M rows requires deliberate choices around memory management, vectorised operations, and file formats (Parquet over CSV, float32 over float64).

**No data leakage** — every rolling feature is computed on `y_lag_1` (the previous day's sales), not the current day. This is a common mistake in time series ML that inflates validation scores. We enforced strict temporal discipline throughout.

**Efficient multi-horizon design** — instead of training 28 models (one per forecast day), we designed a 4-anchor system that covers the full 28-day horizon with linear interpolation. This reduced training compute by 7× while preserving accuracy at key business horizons (day 1, week 1, week 2, month 1).

**Business logic on top of ML** — raw predictions alone aren't useful to operations teams. We layered safety stock calculations (90% service level, square-root lead-time scaling), reorder point formulas, and priority classification (CRITICAL / WARNING / OK) on top of the model outputs to produce actionable recommendations.

**Full-stack deployment** — the model doesn't live in a notebook. It's served via a FastAPI backend with SSE streaming, persisted in PostgreSQL, and presented through a Next.js dashboard with real-time AI chat. Every component runs in Docker Compose for one-command deployment.

**LLM integration with context injection** — the AI assistant isn't generic. Its system prompt is dynamically built from live inventory data — store breakdowns, category summaries, top critical items — so it answers questions accurately about your specific stock state, not generic retail knowledge.

### Skills demonstrated

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
[![GitHub](https://img.shields.io/badge/GitHub-Releases%20%28Models%29-181717?style=flat-square&logo=github)](https://github.com/Thir13een/m5-forecasting-dashboard/releases)

---

<div align="center">

Built in **2 weeks** by

**Krishna Sonji** &nbsp;·&nbsp; **Shweta Bankar**

[![GitHub](https://img.shields.io/badge/GitHub-Thir13een-181717?style=flat-square&logo=github)](https://github.com/Thir13een)
&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-shwetabankar54-181717?style=flat-square&logo=github)](https://github.com/shwetabankar54)

</div>
