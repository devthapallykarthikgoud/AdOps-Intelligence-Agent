# 🤖 AdOps Intelligence Agent

> An end-to-end AI-powered advertising campaign monitoring and alerting platform.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-purple?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 📌 Overview

**AdOps Intelligence Agent** automatically detects underperforming ad campaigns, performs deep KPI analysis, generates LLM-powered optimization recommendations, and sends automated business alerts — all without manual intervention.

Built for AdTech companies, media buying teams, and programmatic advertising platforms.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **KPI Analysis** | Automatically evaluates CTR, CPM, Fill Rate, and Revenue |
| 🚨 **Severity Classification** | Tags campaigns as `OK`, `MEDIUM`, `HIGH`, or `CRITICAL` |
| 🧠 **LangGraph AI Workflow** | Multi-step agent pipeline for intelligent reasoning |
| 🔍 **RAG Optimization Retrieval** | Fetches relevant strategies via FAISS vector search |
| 🤖 **Groq LLM Reasoning** | Root-cause analysis & recommendations via Llama 3.1 |
| 📧 **Automated Email Alerts** | Sends critical campaign alerts via Gmail + n8n |
| ☁️ **Google Drive Trigger** | Fully automated pipeline on CSV upload |
| 📁 **Exportable Reports** | CSV and JSON export support |

---

## 🏗️ System Architecture

```
┌────────────────────┐
│ Google Drive Upload │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   n8n Automation   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  FastAPI AI Backend│
└─────────┬──────────┘
          │
     ┌────┴────┐
     ▼         ▼
┌──────────┐  ┌───────────────┐
│KPI       │  │RAG Retrieval  │
│Analyzer  │  │(FAISS + Embed)│
└──────────┘  └───────────────┘
          │
          ▼
┌────────────────────┐
│  Groq LLM Reasoning│
│  (Llama 3.1)       │
└────────────────────┘
          │
          ▼
┌────────────────────┐
│  AI Recommendations│
└────────────────────┘
          │
          ▼
┌────────────────────┐
│  Gmail Alert System│
└────────────────────┘
```

---

## 🧠 LangGraph AI Workflow

The agent pipeline is orchestrated using **LangGraph** across four sequential nodes:

```
KPI Analysis  →  RAG Retrieval  →  LLM Reasoning  →  Recommendation Generation
```

### Node Breakdown

**1. KPI Analysis Node**
Detects anomalies in CTR, CPM, and Fill Rate. Assigns a severity level to each campaign.

**2. RAG Retrieval Node**
Queries a custom FAISS-indexed knowledge base using semantic search to surface relevant optimization strategies.

**3. LLM Node (Groq / Llama 3.1)**
Analyzes the campaign data and retrieval context to generate:
- Root cause explanation
- Business impact summary
- Actionable optimization steps

**4. Recommendation Generation**
Outputs structured recommendations ready for export or alert delivery.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| AI Orchestration | LangGraph |
| LLM | Groq API · Llama 3.1 |
| Vector Search | FAISS |
| Automation | n8n |
| File Trigger | Google Drive |
| Alerts | Gmail API |
| Data Processing | Pandas |
| Deployment | Render · Streamlit Cloud |
| Config | python-dotenv |

---

## 📂 Project Structure

```
AdOps-Intelligence-Agent/
│
├── app/
│   ├── agent.py          # LangGraph agent definition
│   ├── ai_engine.py      # Core AI pipeline logic
│   ├── ingest.py         # CSV ingestion & preprocessing
│   ├── llm.py            # Groq LLM integration
│   ├── models.py         # Pydantic data models
│   ├── rag.py            # FAISS vector retrieval
│   └── __init__.py
│
├── data/
│   ├── campaign_data.csv     # Sample campaign input
│   └── knowledge_base.txt    # RAG knowledge base
│
├── reports/                  # Generated reports output
│
├── api.py                    # FastAPI application
├── streamlit_app.py          # Streamlit frontend
├── requirements.txt
├── .env                      # Environment variables (not committed)
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AdOps-Intelligence-Agent.git
cd AdOps-Intelligence-Agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

**Activate:**

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.

---

## ▶️ Running the Application

### Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

### FastAPI Backend

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

API docs will be available at: `http://localhost:8000/docs`

---

## 🌐 API Reference

### `POST /analyze`

Upload a campaign CSV file for AI analysis.

**Request:** `multipart/form-data` — CSV file

**Response:**
```json
{
  "campaigns": [
    {
      "campaign_id": "C001",
      "campaign_name": "Nike Summer Sale",
      "severity": "HIGH",
      "alert_level": "⚠ HIGH",
      "recommendations": "...",
      "root_cause": "...",
      "business_impact": "..."
    }
  ]
}
```

---

## 📊 Sample Campaign CSV

```csv
campaign_id,campaign_name,impressions,clicks,revenue,fill_rate,ctr,cpm
C001,Nike Summer Sale,100000,180,70,65,0.18,0.70
C002,Amazon Prime Day,120000,150,60,55,0.15,0.50
```

---

## 🔁 n8n Automation Workflow

The end-to-end automation flow is:

```
Google Drive Trigger
        ↓
Download Uploaded CSV
        ↓
HTTP Request → FastAPI /analyze
        ↓
Parse AI Response
        ↓
Send Gmail Alert (Critical Campaigns)
```

This enables **zero-touch monitoring** — the data team simply drops a CSV into Google Drive and alerts are sent automatically.

---

## ☁️ Deployment

| Component | Platform |
|---|---|
| Frontend (Streamlit) | Streamlit Cloud |
| Backend (FastAPI) | Render |

---

## 🚀 Roadmap

- [ ] Slack & WhatsApp notifications
- [ ] Real-time campaign dashboards
- [ ] Predictive anomaly detection
- [ ] AI memory agents for historical context
- [ ] PostgreSQL integration for persistent storage
- [ ] Multi-client / multi-tenant support
- [ ] Docker & Kubernetes deployment
- [ ] Historical trend analysis

---

## 🎯 Use Cases

This platform is designed for:

- AdTech companies monitoring programmatic campaigns
- Media buying teams tracking performance at scale
- Revenue operations teams managing publisher inventory
- Marketing analytics teams needing automated alerts

---

## 👨‍💻 Author

**Devathapally Uma Karthikeya**

AI/ML · Full Stack · Agentic AI · Automation Systems

---

## ⭐ Support

If you find this project useful, give it a ⭐ on GitHub — it helps others discover it!
