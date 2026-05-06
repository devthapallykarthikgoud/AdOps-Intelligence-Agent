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

```python
import io
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.ingest import load_campaign_data, FAISSIndex
from app.agent import run_pipeline


# ─────────────────────────────────────────────
# APP LIFECYCLE
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("🚀 Loading FAISS Knowledge Base...")

    app.state.faiss_index = FAISSIndex(
        "data/knowledge_base.txt"
    )

    print("✅ FAISS Ready")

    yield

    print("👋 API Shutdown")


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(

    title="AdOps Intelligence Agent API",

    description="AI-powered AdOps monitoring API",

    version="2.0.0",

    lifespan=lifespan
)

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# CONVERT ANALYSIS TO DICT
# ─────────────────────────────────────────────

def analysis_to_dict(result):

    return {

        "campaign_id":
            result.campaign_id,

        "campaign_name":
            result.campaign_name,

        "severity":
            result.severity.value,

        "ctr":
            round(result.ctr, 4),

        "cpm":
            round(result.cpm, 2),

        "fill_rate":
            round(result.fill_rate, 1),

        "issues":
            [i.issue_type for i in result.issues],

        "explanation":
            result.explanation,

        "recommendations":
            result.recommendations
    }


# ─────────────────────────────────────────────
# HTML EMAIL GENERATOR
# ─────────────────────────────────────────────

def build_html_email(results, today):

    critical = [
        r for r in results
        if r["severity"] == "CRITICAL"
    ]

    high = [
        r for r in results
        if r["severity"] == "HIGH"
    ]

    medium = [
        r for r in results
        if r["severity"] == "MEDIUM"
    ]

    ok = [
        r for r in results
        if r["severity"] == "OK"
    ]

    # ─────────────────────────
    # Campaign Cards
    # ─────────────────────────

    campaign_cards = ""

    for r in results:

        severity = r["severity"]

        severity_color = {

            "CRITICAL": "#ef4444",

            "HIGH": "#f97316",

            "MEDIUM": "#eab308",

            "OK": "#22c55e"

        }.get(severity, "#3b82f6")

        issues = ", ".join(
            r["issues"]
        ) if r["issues"] else "No Issues"

        recommendations_html = ""

        for rec in r["recommendations"]:

            recommendations_html += f"""
            <li style="
                margin-bottom:8px;
                line-height:1.6;
            ">
                {rec}
            </li>
            """

        campaign_cards += f"""

        <div style="
            background:#111827;
            border-left:6px solid {severity_color};
            border-radius:16px;
            padding:24px;
            margin-bottom:28px;
        ">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin-bottom:20px;
            ">

                <h2 style="
                    color:white;
                    margin:0;
                    font-size:24px;
                ">
                    {r['campaign_name']}
                </h2>

                <span style="
                    background:{severity_color};
                    color:white;
                    padding:10px 18px;
                    border-radius:999px;
                    font-size:12px;
                    font-weight:bold;
                    letter-spacing:1px;
                ">
                    {severity}
                </span>

            </div>

            <table style="
                width:100%;
                border-collapse:collapse;
                margin-bottom:24px;
            ">

                <tr>

                    <td style="
                        background:#1f2937;
                        color:#9ca3af;
                        padding:14px;
                        font-weight:bold;
                    ">
                        CTR
                    </td>

                    <td style="
                        background:#0f172a;
                        color:white;
                        padding:14px;
                    ">
                        {r['ctr']}%
                    </td>

                </tr>

                <tr>

                    <td style="
                        background:#1f2937;
                        color:#9ca3af;
                        padding:14px;
                        font-weight:bold;
                    ">
                        CPM
                    </td>

                    <td style="
                        background:#0f172a;
                        color:white;
                        padding:14px;
                    ">
                        ${r['cpm']}
                    </td>

                </tr>

                <tr>

                    <td style="
                        background:#1f2937;
                        color:#9ca3af;
                        padding:14px;
                        font-weight:bold;
                    ">
                        Fill Rate
                    </td>

                    <td style="
                        background:#0f172a;
                        color:white;
                        padding:14px;
                    ">
                        {r['fill_rate']}%
                    </td>

                </tr>

            </table>

            <div style="
                background:#0f172a;
                padding:18px;
                border-radius:12px;
                margin-bottom:20px;
            ">

                <h3 style="
                    color:#60a5fa;
                    margin-top:0;
                ">
                    Detected Issues
                </h3>

                <p style="
                    color:#d1d5db;
                    line-height:1.8;
                ">
                    {issues}
                </p>

            </div>

            <div style="
                background:#0f172a;
                padding:18px;
                border-radius:12px;
                margin-bottom:20px;
            ">

                <h3 style="
                    color:#c084fc;
                    margin-top:0;
                ">
                    AI Explanation
                </h3>

                <p style="
                    color:#d1d5db;
                    line-height:1.8;
                ">
                    {r['explanation']}
                </p>

            </div>

            <div style="
                background:#0f172a;
                padding:18px;
                border-radius:12px;
            ">

                <h3 style="
                    color:#34d399;
                    margin-top:0;
                ">
                    Recommendations
                </h3>

                <ul style="
                    color:#d1d5db;
                    line-height:1.8;
                    padding-left:20px;
                ">
                    {recommendations_html}
                </ul>

            </div>

        </div>
        """

    # ─────────────────────────
    # Final HTML
    # ─────────────────────────

    html = f"""

    <html>

    <body style="
        background:#020617;
        padding:40px;
        font-family:Arial,sans-serif;
    ">

        <div style="
            max-width:1000px;
            margin:auto;
            background:#0f172a;
            border-radius:24px;
            padding:40px;
        ">

            <h1 style="
                color:white;
                text-align:center;
                margin-bottom:12px;
            ">
                📊 AdOps AI Monitoring Report
            </h1>

            <p style="
                color:#94a3b8;
                text-align:center;
                margin-bottom:40px;
            ">
                Generated on {today}
            </p>

            <div style="
                display:flex;
                justify-content:space-between;
                gap:16px;
                margin-bottom:40px;
            ">

                <div style="
                    flex:1;
                    background:#7f1d1d;
                    color:white;
                    padding:20px;
                    border-radius:14px;
                    text-align:center;
                ">
                    🚨 Critical
                    <h2>{len(critical)}</h2>
                </div>

                <div style="
                    flex:1;
                    background:#7c2d12;
                    color:white;
                    padding:20px;
                    border-radius:14px;
                    text-align:center;
                ">
                    ⚠ High
                    <h2>{len(high)}</h2>
                </div>

                <div style="
                    flex:1;
                    background:#713f12;
                    color:white;
                    padding:20px;
                    border-radius:14px;
                    text-align:center;
                ">
                    🟡 Medium
                    <h2>{len(medium)}</h2>
                </div>

                <div style="
                    flex:1;
                    background:#14532d;
                    color:white;
                    padding:20px;
                    border-radius:14px;
                    text-align:center;
                ">
                    ✅ Healthy
                    <h2>{len(ok)}</h2>
                </div>

            </div>

            {campaign_cards}

            <div style="
                text-align:center;
                color:#64748b;
                margin-top:50px;
                font-size:14px;
            ">

                AdOps Intelligence Agent
                <br>
                Automated AI Campaign Monitoring System

            </div>

        </div>

    </body>

    </html>
    """

    return html


# ─────────────────────────────────────────────
# HEALTH ROUTE
# ─────────────────────────────────────────────

@app.get("/health")
async def health():

    return {

        "status": "healthy",

        "knowledge_base_docs":
            len(app.state.faiss_index.documents),

        "model":
            "llama-3.1-8b-instant"
    }


# ─────────────────────────────────────────────
# ANALYZE ROUTE
# ─────────────────────────────────────────────

@app.post("/analyze")
async def analyze(

    file: UploadFile = File(...)
):

    if not file.filename.endswith(".csv"):

        raise HTTPException(

            status_code=400,

            detail="Only CSV files supported"
        )

    try:

        raw_bytes = await file.read()

        csv_file = io.StringIO(
            raw_bytes.decode("utf-8")
        )

        df = load_campaign_data(csv_file)

    except Exception as e:

        raise HTTPException(

            status_code=422,

            detail=f"CSV parsing failed: {str(e)}"
        )

    # ─────────────────────────
    # Run AI Pipeline
    # ─────────────────────────

    from app.models import Campaign

    results_raw = []

    for _, row in df.iterrows():

        campaign = Campaign(

            campaign_id=str(
                row["campaign_id"]
            ),

            campaign_name=str(
                row["campaign_name"]
            ),

            impressions=float(
                row["impressions"]
            ),

            clicks=float(
                row["clicks"]
            ),

            revenue=float(
                row["revenue"]
            ),

            fill_rate=float(
                row["fill_rate"]
            ),

            ctr=float(
                row["ctr"]
            ),

            cpm=float(
                row["cpm"]
            )
        )

        analysis = run_pipeline(
            campaign,
            app.state.faiss_index
        )

        results_raw.append(
            analysis
        )

    # ─────────────────────────
    # Convert Results
    # ─────────────────────────

    results = [

        analysis_to_dict(r)

        for r in results_raw
    ]

    today = date.today().strftime(
        "%B %d, %Y"
    )

    critical = [
        r for r in results
        if r["severity"] == "CRITICAL"
    ]

    high = [
        r for r in results
        if r["severity"] == "HIGH"
    ]

    medium = [
        r for r in results
        if r["severity"] == "MEDIUM"
    ]

    ok = [
        r for r in results
        if r["severity"] == "OK"
    ]

    urgent = [
        r for r in results
        if r["severity"] in [
            "CRITICAL",
            "HIGH"
        ]
    ]

    # ─────────────────────────
    # Overall Alert Level
    # ─────────────────────────

    if critical:

        alert_level = "CRITICAL"

    elif high:

        alert_level = "HIGH"

    elif medium:

        alert_level = "MEDIUM"

    else:

        alert_level = "OK"

    # ─────────────────────────
    # Subject
    # ─────────────────────────

    if urgent:

        email_subject = (
            f"[{alert_level}] "
            f"AdOps AI Report — "
            f"{today} | "
            f"{len(urgent)} urgent campaign(s)"
        )

    else:

        email_subject = (
            f"[ALL CLEAR] "
            f"AdOps AI Report — "
            f"{today}"
        )

    # ─────────────────────────
    # HTML Email
    # ─────────────────────────

    html_email = build_html_email(
        results,
        today
    )

    # ─────────────────────────
    # Final Response
    # ─────────────────────────

    return {

        "alert_level":
            alert_level,

        "has_urgent":
            len(urgent) > 0,

        "email_subject":
            email_subject,

        "email_body":
            html_email,

        "total_campaigns":
            len(results),

        "critical_count":
            len(critical),

        "high_count":
            len(high),

        "medium_count":
            len(medium),

        "ok_count":
            len(ok),

        "campaigns":
            results,

        "urgent_campaigns":
            urgent
    }
```
