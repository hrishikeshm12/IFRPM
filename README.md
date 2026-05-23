# IFRPM — Intelligent Fleet Risk & Predictive Maintenance

![Banner](docs/banner.png)

> AI-powered aircraft fleet health monitoring with dynamic RUL prediction, risk scoring, and automated alerting — Arizona State University · Team Kansas

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-psycopg2-336791?logo=postgresql&logoColor=white)](https://postgresql.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [RUL Prediction Engine](#rul-prediction-engine)
- [Alerting Thresholds](#alerting-thresholds)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Contributing](#contributing)

---

## Overview

IFRPM is an intelligent fleet management system that combines real-time component telemetry with a dynamic multi-model ML ensemble to predict **Remaining Useful Life (RUL)** of aircraft components. The system surfaces risk scores, triggers automated maintenance alerts, and integrates external weather data as a predictive feature — all through a clean FastAPI REST interface and a React dashboard.

The ML engine is designed for **flexibility**: drop any scikit-learn `.pkl` or Keras `.h5` model into the models directory and it is automatically discovered, loaded, and included in the ensemble at startup — no code changes required.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                 React Dashboard (src/ifrpm-dashboard.jsx)         │
│      Fleet overview · Component health · RUL gauges · Alerts     │
└───────────────────────────┬──────────────────────────────────────┘
                            │ HTTP/REST
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│              FastAPI Application (Uvicorn — port 8000)            │
│                                                                   │
│  ┌───────────────┐  ┌────────────────┐  ┌─────────────────────┐  │
│  │    Routers    │  │    Services    │  │    ML Engine        │  │
│  │               │  │                │  │                     │  │
│  │  /aircraft    │  │  RUL Service   │  │  Dynamic Ensemble   │  │
│  │  /fleet       │  │  Risk Service  │  │  (.pkl + .h5 auto-  │  │
│  │  /alerts      │  │  Weather Svc   │  │   discovery)        │  │
│  │  /rul         │  │  Health Index  │  │  Feature Pipeline   │  │
│  └───────────────┘  └────────────────┘  └─────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │            SQLAlchemy ORM + psycopg2 (PostgreSQL)          │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│           External Weather API (feature enrichment)               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Features

- **Dynamic RUL Ensemble** — Loads all `.pkl` (joblib) and `.h5` (Keras/TensorFlow) models at startup; averages predictions across the ensemble for a single, robust estimate
- **Three-tier Alert System** — Critical / High / Medium thresholds with automated alert generation when RUL drops below configured limits
- **Fleet-wide Risk Dashboard** — Per-component and fleet-level risk aggregation with configurable thresholds
- **Weather Feature Integration** — External weather data injected into the risk model feature pipeline
- **Health Index** — Composite health score derived from sensor readings and ML outputs
- **Hot-swappable Models** — Add or remove model files without restarting; the ensemble auto-updates on next server start
- **Type-safe API** — Full Pydantic v2 validation on every request and response
- **Database-backed State** — Aircraft, component, and alert records persisted in PostgreSQL via SQLAlchemy ORM

---

## RUL Prediction Engine

Located in `backend/app/ml/`, the engine follows a three-stage pipeline:

```
Raw sensor telemetry
        │
        ▼
┌───────────────────────────────────────┐
│  Feature Engineering                  │
│  · Sensor normalization               │
│  · Rolling statistics (mean, std)     │
│  · Weather feature injection          │
│  · Health index calculation           │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  Model Auto-Discovery                 │
│  · Scans MODEL_DIR for *.pkl, *.h5    │
│  · Loads joblib (scikit-learn)        │
│  · Loads Keras weights (HDF5)         │
│  · Skips corrupt files gracefully     │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  Averaged Ensemble Prediction         │
│  · All loaded models predict          │
│  · Outputs averaged to single RUL     │
│  · Risk tier classification applied   │
└───────────────────────────────────────┘
```

**Adding a new model:** Drop a trained `.pkl` or `.h5` file into the `models/` directory and restart the server — it will be included in the ensemble automatically.

---

## Alerting Thresholds

Configurable via environment variables:

| Threshold | Default RUL Value | Action |
|---|---|---|
| `RUL_CRITICAL_THRESHOLD` | **≤ 10** | Immediate maintenance required |
| `RUL_HIGH_THRESHOLD` | **≤ 30** | Schedule maintenance soon |
| `RUL_MEDIUM_THRESHOLD` | **≤ 80** | Monitor closely |
| Above medium | > 80 | Normal operation |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| ML — Ensemble | scikit-learn, TensorFlow 2.x, joblib |
| Data Processing | pandas, NumPy, SciPy |
| Database | PostgreSQL + SQLAlchemy + psycopg2-binary |
| Validation | Pydantic v2, pydantic-settings |
| HTTP Client | httpx |
| Frontend Dashboard | React (JSX) |

---

## Project Structure

```
IFRPM/
├── backend/
│   ├── .env.example               # Environment variable template
│   └── app/
│       ├── main.py                # FastAPI app entrypoint
│       ├── config.py              # Settings (DATABASE_URL, MODEL_DIR, thresholds)
│       ├── database.py            # SQLAlchemy session & engine setup
│       ├── seed.py                # Database seeding script
│       ├── ml/                    # ML engine (model loader, ensemble, feature engineering)
│       ├── models/                # SQLAlchemy ORM models
│       ├── routers/               # FastAPI route handlers (aircraft, fleet, alerts, RUL)
│       ├── schemas/               # Pydantic request/response schemas
│       ├── services/              # Business logic (RUL, risk, weather, health index)
│       └── utils/                 # Helpers and utilities
│
├── src/
│   ├── ifrpm-dashboard.jsx        # React dashboard (31.5 KB)
│   └── dashboard/                 # Dashboard sub-components
│
├── docs/                          # Architecture docs
├── phase_1/                       # Phase 1 research & prototyping
├── quad_chart.py                  # Quad-chart visualization script
├── Dashboard.xlsx                 # Fleet planning dashboard
├── IFRPM_Quad_Chart.pdf           # Project overview quad chart
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL running locally (or via Docker)
- Node.js 18+ (for the React dashboard)

### 1. Clone the repo

```bash
git clone https://github.com/hrishikeshm12/IFRPM.git
cd IFRPM
```

### 2. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:

```env
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/ifrpm_dev
MODEL_DIR=../../models
RUL_CRITICAL_THRESHOLD=10
RUL_HIGH_THRESHOLD=30
RUL_MEDIUM_THRESHOLD=80
WEATHER_API_KEY=your_weather_api_key
```

### 4. Set up the database

```bash
# Create the database
createdb ifrpm_dev

# Seed initial data
python backend/app/seed.py
```

### 5. Start the API server

```bash
uvicorn backend.app.main:app --reload --port 8000
```

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### 6. Launch the dashboard (optional)

```bash
cd src
npm install
npm start
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/aircraft` | List all registered aircraft |
| `POST` | `/aircraft` | Register a new aircraft |
| `GET` | `/aircraft/{id}` | Get aircraft details |
| `GET` | `/fleet/risk` | Fleet-wide risk summary |
| `GET` | `/fleet/health` | Overall fleet health index |
| `POST` | `/rul/predict` | Predict RUL for a component |
| `GET` | `/alerts` | Fetch all active alerts |
| `GET` | `/alerts/{aircraft_id}` | Alerts for a specific aircraft |
| `PUT` | `/alerts/{id}/resolve` | Resolve an alert |

> Full interactive docs at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## Dashboard

The React dashboard (`src/ifrpm-dashboard.jsx`) provides:

- **Fleet Overview** — All aircraft at-a-glance with color-coded health status
- **RUL Gauges** — Per-component RUL visualization with threshold markers (Critical / High / Medium)
- **Risk Heatmap** — Fleet-wide component risk visualization
- **Alert Feed** — Real-time active alerts with severity classification
- **Historical Trends** — RUL degradation curves for tracked components

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit changes following conventional commits: `git commit -m "feat: add ..."`
4. Run tests and ensure linting passes
5. Open a pull request against `main`

---

## License

MIT
