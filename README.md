# M5 Demand Forecasting Dashboard

Retail inventory management system powered by LightGBM demand forecasting. Predicts sales across **30,490 products** in 10 stores and generates reorder recommendations, safety stock calculations, and stockout timelines.

![Stack](https://img.shields.io/badge/Next.js-14-black) ![Stack](https://img.shields.io/badge/FastAPI-0.111-green) ![Stack](https://img.shields.io/badge/LightGBM-RMSSE_0.98-blue) ![Stack](https://img.shields.io/badge/Groq-Qwen3--32B-orange)

---

## Features

- **Inventory dashboard** — sortable/filterable grid of all SKUs with reorder status (CRITICAL / WARNING / OK)
- **Inline row expansion** — click any row to see a 28-day demand forecast chart + key metrics
- **AI assistant** — chat interface powered by Qwen3-32B via Groq, pre-loaded with live inventory context
- **CSV upload** — upload new sales data to re-run forecasts on demand
- **Safety stock** — 90% service level (Z = 1.282), 7-day lead time, per-SKU demand volatility

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React, Bootstrap 5, ReactMarkdown |
| Backend | FastAPI, LightGBM, pandas, numpy |
| Database | PostgreSQL 16 |
| AI Chat | Groq API — Qwen3-32B |
| Deployment | Docker Compose |

---

## Getting Started

### 1. Download the models

Download all 5 files from the [v1.0 release](https://github.com/Thir13een/m5-forecasting-dashboard/releases/tag/v1.0) into a local folder (e.g. `~/models/m5`):

| File | Size |
|---|---|
| [lgb_direct_h01.txt](https://github.com/Thir13een/m5-forecasting-dashboard/releases/download/v1.0/lgb_direct_h01.txt) | 195 MB |
| [lgb_direct_h07.txt](https://github.com/Thir13een/m5-forecasting-dashboard/releases/download/v1.0/lgb_direct_h07.txt) | 304 MB |
| [lgb_direct_h14.txt](https://github.com/Thir13een/m5-forecasting-dashboard/releases/download/v1.0/lgb_direct_h14.txt) | 325 MB |
| [lgb_direct_h28.txt](https://github.com/Thir13een/m5-forecasting-dashboard/releases/download/v1.0/lgb_direct_h28.txt) | 320 MB |
| [feature_cols.csv](https://github.com/Thir13een/m5-forecasting-dashboard/releases/download/v1.0/feature_cols.csv) | 1 KB |

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
MODELS_PATH=/absolute/path/to/your/models/folder
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://m5:m5pass@postgres:5432/m5db
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Run with Docker

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs

---

## Project Structure

```
m5-forecasting-dashboard/
├── backend/
│   ├── app/
│   │   ├── routers/       # FastAPI route handlers
│   │   └── services/      # LightGBM inference, feature engineering, chat
│   └── requirements.txt
├── frontend/
│   ├── app/               # Next.js pages
│   ├── components/        # UI components
│   └── lib/               # API client, types, utils
├── infrastructure/
│   └── init.sql           # PostgreSQL schema
├── docker-compose.yml
└── .env.example
```

---

## Model Details

- **Algorithm:** LightGBM (direct multi-horizon)
- **Horizons:** 1, 7, 14, 28 days
- **Dataset:** M5 Forecasting competition (Walmart sales data)
- **Coverage:** 30,490 SKUs across CA-1–4, TX-1–3, WI-1–3
- **Accuracy:** RMSSE 0.98
- **Features:** lag features, rolling means/std, day-of-week, month cyclicals, SNAP flags, event indicators
