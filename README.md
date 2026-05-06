# AdOps Intelligence Agent

AI-powered campaign monitoring and optimization platform built using LangGraph, RAG (Retrieval-Augmented Generation), FAISS vector search, Groq LLMs, and Streamlit.

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

- Python
- Streamlit
- LangGraph
- FAISS
- Sentence Transformers
- Groq API
- Pandas
- Matplotlib

## Project Structure

```text
adops_agent/

├── app/
├── data/
├── reports/
├── incoming_csv/
├── streamlit_app.py
├── analyze.py
├── requirements.txt
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
