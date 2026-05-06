"""
api.py — FastAPI Layer for Automation
---------------------------------------
WHY THIS FILE EXISTS:
  n8n (and any external tool) cannot trigger Streamlit directly.
  Streamlit is a UI framework — it needs a human clicking buttons.

  This file adds a REST API layer that n8n CAN call:
    POST /analyze  → accepts CSV bytes → runs your LangGraph pipeline
                   → returns structured JSON

IMPORTANT:
  Your existing files (streamlit_app.py, agent.py, ai_engine.py,
  ingest.py, rag.py, models.py) are NOT changed at all.
  This file just imports and reuses them.

HOW IT WORKS WITH n8n:
  1. User uploads CSV to Google Drive
  2. n8n detects the new file (Google Drive trigger)
  3. n8n downloads the CSV bytes
  4. n8n POSTs the bytes to POST /analyze on this server
  5. This endpoint runs your pipeline and returns JSON
  6. n8n reads the JSON and sends Gmail

INTERVIEW TALKING POINT:
  "I kept the Streamlit UI and the automation API completely separate.
  The API reuses all existing pipeline logic — no duplication.
  This is the single-responsibility principle: Streamlit handles
  human interaction, FastAPI handles machine-to-machine communication."
"""

import io
import os
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Import YOUR existing modules (unchanged) ──────────────────────────────────
from app.ingest import load_campaign_data, FAISSIndex
from app.agent import run_pipeline
from app.models import Severity


# ─── Startup: load expensive resources once ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the FAISS index + embedding model ONCE when the server starts.
    Reused for every incoming request — avoids 5-second delay per call.
    """
    print("🚀 Starting AdOps API — loading FAISS index...")
    app.state.faiss_index = FAISSIndex("data/knowledge_base.txt")
    print("✅ FAISS index ready")
    yield
    print("👋 Shutting down")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AdOps Intelligence Agent API",
    description="Automated campaign analysis endpoint for n8n integration.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _analysis_to_dict(result) -> dict:
    """
    Converts your CampaignAnalysis dataclass → plain dict for JSON response.
    Handles the Severity enum serialization (e.g. Severity.CRITICAL → "CRITICAL").
    """
    return {
        "campaign_id":   result.campaign_id,
        "campaign_name": result.campaign_name,
        "severity":      result.severity.value if hasattr(result.severity, "value") else str(result.severity),
        "ctr":           round(result.ctr, 4),
        "cpm":           round(result.cpm, 2),
        "fill_rate":     round(result.fill_rate, 1),
        "issues":        [i.issue_type for i in result.issues],
        "explanation":   result.explanation or "",
        "recommendations": result.recommendations or [],
    }


def _build_email_body(results: list, today: str) -> str:
    """
    Builds a plain-text email body from pipeline results.
    n8n reads this string directly from the JSON and passes it to Gmail.
    No string manipulation needed inside the n8n workflow.
    """
    total      = len(results)
    critical   = [r for r in results if r["severity"] == "CRITICAL"]
    high       = [r for r in results if r["severity"] == "HIGH"]
    medium     = [r for r in results if r["severity"] == "MEDIUM"]
    ok         = [r for r in results if r["severity"] == "OK"]
    flagged    = [r for r in results if r["severity"] != "OK"]

    lines = [
        f"AdOps Daily Report — {today}",
        "=" * 50,
        "",
        "CAMPAIGN SUMMARY",
        f"  Total Analyzed : {total}",
        f"  🔴 Critical    : {len(critical)}",
        f"  🟠 High        : {len(high)}",
        f"  🟡 Medium      : {len(medium)}",
        f"  ✅ Healthy     : {len(ok)}",
        "",
    ]

    if flagged:
        lines += [
            "CAMPAIGNS REQUIRING ACTION",
            "-" * 50,
        ]
        for r in flagged:
            lines += [
                "",
                f"[{r['severity']}] {r['campaign_name']}  (ID: {r['campaign_id']})",
                f"  CTR: {r['ctr']:.4f}%  |  CPM: ${r['cpm']:.2f}  |  Fill Rate: {r['fill_rate']:.1f}%",
                f"  Issues: {', '.join(r['issues']) or 'None'}",
            ]
            if r["recommendations"]:
                lines.append(f"  → {r['recommendations'][0]}")
    else:
        lines.append("✅ All campaigns are healthy — no action required.")

    lines += [
        "",
        "─" * 50,
        "AdOps Intelligence Agent · Automated Report",
        f"Triggered by: Google Drive CSV upload",
    ]

    return "\n".join(lines)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Health check — used by Render to verify the server is running.
    Also confirms the FAISS index loaded successfully.
    """
    return {
        "status": "healthy",
        "faiss_docs": len(app.state.faiss_index.documents),
        "model": "llama-3.1-8b-instant (Groq)",
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    MAIN ENDPOINT — called by n8n when a new CSV is uploaded to Google Drive.

    Accepts: multipart CSV file upload
    Returns: full JSON report + pre-formatted email fields

    n8n workflow:
      Google Drive Trigger → Download File → POST /analyze → Gmail

    The response includes:
      - alert_level    : overall severity (for IF node in n8n)
      - email_subject  : ready-made subject line
      - email_body     : ready-made plain text (paste into Gmail node)
      - campaigns      : full list with details
      - urgent         : only flagged campaigns (for n8n looping)
    """

    # Validate file type
    if not (file.filename or "").endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted.")

    try:
        # Read uploaded CSV bytes
        raw_bytes = await file.read()

        # Use your existing load_campaign_data — wrap bytes as file-like object
        csv_file = io.StringIO(raw_bytes.decode("utf-8"))
        df = load_campaign_data(csv_file)

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CSV parsing failed: {str(e)}")

    # ── Run your LangGraph pipeline for each campaign row ─────────────────────
    from app.models import Campaign

    results_raw = []

    for _, row in df.iterrows():
        campaign = Campaign(
            campaign_id   = str(row["campaign_id"]),
            campaign_name = str(row["campaign_name"]),
            impressions   = float(row["impressions"]),
            clicks        = float(row["clicks"]),
            revenue       = float(row["revenue"]),
            fill_rate     = float(row["fill_rate"]),
            ctr           = float(row["ctr"]),
            cpm           = float(row["cpm"]),
        )
        analysis = run_pipeline(campaign, app.state.faiss_index)
        results_raw.append(analysis)

    # ── Convert to serializable dicts ─────────────────────────────────────────
    results = [_analysis_to_dict(r) for r in results_raw]

    # ── Build summary counts ───────────────────────────────────────────────────
    today        = date.today().strftime("%B %d, %Y")
    critical     = [r for r in results if r["severity"] == "CRITICAL"]
    high         = [r for r in results if r["severity"] == "HIGH"]
    medium       = [r for r in results if r["severity"] == "MEDIUM"]
    ok           = [r for r in results if r["severity"] == "OK"]
    urgent       = [r for r in results if r["severity"] in ("CRITICAL", "HIGH")]

    # Overall alert level (used by n8n IF node to decide email type)
    if critical:
        alert_level = "CRITICAL"
    elif high:
        alert_level = "HIGH"
    elif medium:
        alert_level = "MEDIUM"
    else:
        alert_level = "OK"

    email_subject = (
        f"[{alert_level}] AdOps Report — {today} | "
        f"{len(urgent)} urgent campaign(s) flagged"
        if urgent else
        f"[ALL CLEAR] AdOps Report — {today} | All campaigns healthy"
    )

    return {
        # ── For n8n IF node ──────────────────────────────────────────────────
        "alert_level":     alert_level,
        "has_urgent":      len(urgent) > 0,

        # ── Ready-made email fields ──────────────────────────────────────────
        "email_subject":   email_subject,
        "email_body":      _build_email_body(results, today),

        # ── Summary counts ───────────────────────────────────────────────────
        "total_campaigns": len(results),
        "critical_count":  len(critical),
        "high_count":      len(high),
        "medium_count":    len(medium),
        "ok_count":        len(ok),

        # ── Full data ────────────────────────────────────────────────────────
        "campaigns":       results,
        "urgent_campaigns": urgent,
    }
