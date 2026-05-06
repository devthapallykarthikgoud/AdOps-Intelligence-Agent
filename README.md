# AdOps Intelligence Agent

AI-powered campaign monitoring and optimization platform built using LangGraph, RAG (Retrieval-Augmented Generation), FAISS vector search, Groq LLMs, and Streamlit.

This system analyzes advertising campaign performance data, detects KPI anomalies, retrieves optimization strategies using semantic search, generates AI-driven business explanations, and provides actionable recommendations for AdOps teams.

## Features

- AI Campaign Analysis
- LangGraph Workflow
- RAG with FAISS
- Groq Llama 3.1 Integration
- Dynamic Recommendations
- Confidence Scoring
- Interactive Dashboard
- CSV & JSON Export
- Planned n8n Automation

## Tech Stack

| Technology            | Purpose                   |
| --------------------- | ------------------------- |
| Python                | Core programming language |
| Streamlit             | Frontend dashboard        |
| LangGraph             | AI workflow orchestration |
| FAISS                 | Vector similarity search  |
| Sentence Transformers | Embeddings                |
| Groq API              | LLM inference             |
| Llama 3.1             | AI reasoning model        |
| Pandas                | Data processing           |
| Matplotlib            | Analytics charts          |


## Project Structure

```text
adops_agent/

├── app/
│   ├── __init__.py
│   ├── agent.py
│   ├── ai_engine.py
│   ├── ingest.py
│   ├── rag.py
│   ├── scoring.py
│   └── models.py
│
├── data/
│   ├── knowledge_base.txt
│   └── sample_campaigns.csv
│
├── reports/
│
├── incoming_csv/
│
├── streamlit_app.py
├── analyze.py
├── requirements.txt
├── .env
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Project

```bash
streamlit run streamlit_app.py
```

## Environment Variables

Create `.env`

```env
GROQ_API_KEY=your_groq_api_key
```

## Planned Automation

Data Team Uploads CSV
↓
n8n Detects File
↓
Runs AI Analysis
↓
Detects Critical Campaigns
↓
Sends Email Alerts

## Author

Umakarthikeya
