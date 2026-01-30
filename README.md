# Sahayog Setu
# 360° Livelihood Grid - Bridging the gap from rights to assets

A comprehensive solution bridging the gap between rural workers and employment opportunities in the VB-G RAM G era.

## 🎯 Project Overview

Sahayog Setu addresses three critical gaps in India's rural employment landscape:

1. **The "Agricultural Pause" Gap** - Income continuity during harvest seasons
2. **The "Normative Allocation" Gap** - Fair budget distribution throughout the year
3. **The "Viksit Plan" Gap** - Technical support for village planning

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Interface Layer                          │
│              (Telephony/IVR - Twilio/Exotel)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Application Logic Layer                     │
│                    (FastAPI Backend)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Workers   │  │    Jobs     │  │   Farmers   │         │
│  │   Router    │  │   Router    │  │   Router    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Data & Intelligence Layer                      │
│         (Supabase PostgreSQL + Hash Chain Ledger)           │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Supabase account with database set up

### Setup

1. **Clone and navigate to the project**
   ```bash
   cd sahayog-setu
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\Activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   # Copy example and fill in your Supabase credentials
   cp .env.example .env
   ```

5. **Run the server**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

6. **Open API docs**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 📁 Project Structure

```
sahayog-setu/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Environment configuration
│   ├── database.py          # Supabase DB connection
│   ├── models/              # SQLAlchemy ORM models
│   ├── schemas/             # Pydantic schemas
│   ├── routers/             # API route handlers
│   │   ├── health.py        # Health checks
│   │   ├── workers.py       # Worker endpoints
│   │   ├── jobs.py          # Government job endpoints
│   │   └── farmers.py       # Private demand endpoints
│   ├── services/            # Business logic
│   └── utils/               # Helper utilities
├── database/
│   └── schema.sql           # PostgreSQL schema
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## 🔌 API Endpoints

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API & database health check |
| GET | `/health/db` | Detailed database status |

### Workers (Mazdoor Mitra)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/workers` | Register new worker |
| GET | `/api/v1/workers` | List workers (filterable) |
| GET | `/api/v1/workers/{id}` | Get worker details |
| PATCH | `/api/v1/workers/{id}` | Update worker |
| PUT | `/api/v1/workers/{id}/availability` | Toggle availability |

### Government Jobs
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/government/projects` | Create project |
| GET | `/api/v1/government/projects` | List projects |
| GET | `/api/v1/government/status/{village}` | Check work status |
| PUT | `/api/v1/government/projects/{id}/status` | Update status (ACTIVE/PAUSED) |

### Private Demand (Harvest Hero)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/farmers` | Register farmer |
| POST | `/api/v1/private-demand` | Post work demand |
| GET | `/api/v1/private-demand` | List demands |
| GET | `/api/v1/private-demand/village/{code}/opportunities` | Get village opportunities |

## 🌾 Key Feature: Harvest Hero

When government work is **PAUSED** (e.g., during harvest season), the system automatically:

1. Detects the pause via `/government/status/{village}`
2. Routes workers to private job opportunities
3. Matches based on village proximity and need score
4. Logs all allocations on the Fairness Ledger

## 📊 Modules

| Module | Target User | Purpose |
|--------|-------------|---------|
| **Mazdoor Mitra** | Workers | Voice-first job matching |
| **Gram Sahayak** | Sarpanch/Leaders | Smart plan generation |
| **Drishti Dashboard** | Block Officers | Budget forecasting & fairness monitoring |

## 🛡️ Data Integrity

All job allocations are recorded with:
- **Hash Chain**: Immutable audit trail
- **Need Score Snapshots**: For bias detection
- **Timestamps**: Full transaction history

## 📝 License

This project is part of the VB-G RAM G initiative.

---

Built with ❤️ for India's rural workforce
