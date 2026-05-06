import os

from dotenv import load_dotenv

from groq import Groq

from app.models import CampaignAnalysis

load_dotenv()

# ─────────────────────────────────────────────
# Initialize Groq Client
# ─────────────────────────────────────────────

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_NAME = "llama-3.1-8b-instant"


# ─────────────────────────────────────────────
# Generate AI Explanation
# ─────────────────────────────────────────────

def generate_explanation(
    analysis: CampaignAnalysis
):

    # Healthy campaigns
    if analysis.severity == "OK":

        return """
Campaign performance is healthy.

All KPIs are operating within acceptable thresholds.
No immediate optimization actions are required.
Continue monitoring campaign trends and audience engagement.
"""

    # ─────────────────────────
    # ISSUE TEXT
    # ─────────────────────────

    issues_text = "\n".join([

        f"- {issue.issue_type} "
        f"(Current: {issue.metric_value}, "
        f"Threshold: {issue.threshold})"

        for issue in analysis.issues
    ])

    # ─────────────────────────
    # RAG CONTEXT
    # ─────────────────────────

    rag_context = "\n\n".join(
        analysis.rag_context
    )

    # ─────────────────────────
    # PROMPT
    # ─────────────────────────

    prompt = f"""
You are a senior AdTech performance strategist.

Analyze this advertising campaign carefully.

CAMPAIGN DETAILS:
- Campaign Name: {analysis.campaign_name}
- Severity Level: {analysis.severity}
- CTR: {analysis.ctr:.2f}%
- CPM: ${analysis.cpm:.2f}
- Fill Rate: {analysis.fill_rate:.2f}%

DETECTED ISSUES:
{issues_text}

RETRIEVED OPTIMIZATION KNOWLEDGE:
{rag_context}

YOUR TASK:

Generate a highly specific business analysis.

Requirements:

1. Explain WHY the campaign is underperforming
2. Explain the BUSINESS IMPACT
3. Give SPECIFIC optimization actions
4. Use campaign metrics in your reasoning
5. Avoid generic recommendations
6. Different campaigns should have different explanations
7. Mention targeting, creatives, bidding, inventory, or audience issues where relevant
8. Keep response professional and concise
9. Make recommendations data-driven
FORMAT:

Root Cause:
...

Business Impact:
...

Recommendations:
1.
2.
3.
"""

    # ─────────────────────────
    # LLM CALL
    # ─────────────────────────

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[

            {
                "role": "system",
                "content": (
                    "You are an expert AdOps optimization strategist "
                    "specialized in programmatic advertising, "
                    "campaign analytics, audience targeting, "
                    "yield optimization, and monetization."
                )
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.7,

        max_tokens=500
    )

    return response.choices[0].message.content.strip()
# ─────────────────────────────────────────────
# EXTRACT RECOMMENDATIONS
# ─────────────────────────────────────────────

def extract_recommendations(
    explanation: str
):

    recommendations = []

    lines = explanation.split("\n")

    for line in lines:

        clean = line.strip()

        # Detect numbered recommendations
        if (
            clean.startswith("1.")
            or clean.startswith("2.")
            or clean.startswith("3.")
            or clean.startswith("-")
        ):

            recommendations.append(
                clean
            )

    return recommendations