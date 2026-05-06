from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    END
)

from app.models import (
    Campaign,
    CampaignAnalysis,
    Issue,
    Severity
)

from app.rag import (
    retrieve_strategies
)

from app.ai_engine import (
    generate_explanation,
    extract_recommendations
)


# ─────────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):

    campaign: Campaign

    faiss_index: object

    analysis: CampaignAnalysis


# ─────────────────────────────────────────────
# KPI ANALYSIS NODE
# ─────────────────────────────────────────────

def kpi_node(
    state: AgentState
):

    campaign = state["campaign"]

    issues = []

    # ─────────────────────────
    # LOW CTR DETECTION
    # ─────────────────────────

    if campaign.ctr < 0.30:

        issues.append(

            Issue(

                issue_type="Low CTR",

                metric_value=campaign.ctr,

                threshold=0.30,

                description=(
                    "CTR below healthy benchmark"
                )
            )
        )

    # ─────────────────────────
    # LOW CPM DETECTION
    # ─────────────────────────

    if campaign.cpm < 1.00:

        issues.append(

            Issue(

                issue_type="Low CPM",

                metric_value=campaign.cpm,

                threshold=1.00,

                description=(
                    "CPM below monetization target"
                )
            )
        )

    # ─────────────────────────
    # LOW FILL RATE DETECTION
    # ─────────────────────────

    if campaign.fill_rate < 70:

        issues.append(

            Issue(

                issue_type="Low Fill Rate",

                metric_value=campaign.fill_rate,

                threshold=70,

                description=(
                    "Fill rate below acceptable level"
                )
            )
        )

    # ─────────────────────────
    # SEVERITY SCORING
    # ─────────────────────────

    issue_count = len(issues)

    if issue_count == 0:

        severity = Severity.OK

    elif issue_count == 1:

        severity = Severity.MEDIUM

    elif issue_count == 2:

        severity = Severity.HIGH

    else:

        severity = Severity.CRITICAL

    # ─────────────────────────
    # CREATE ANALYSIS OBJECT
    # ─────────────────────────

    analysis = CampaignAnalysis(

        campaign_id=campaign.campaign_id,

        campaign_name=campaign.campaign_name,

        ctr=campaign.ctr,

        cpm=campaign.cpm,

        fill_rate=campaign.fill_rate,

        severity=severity,

        issues=issues,

        rag_context=[],

        explanation=None,

        recommendations=[]
    )

    return {
        "analysis": analysis
    }


# ─────────────────────────────────────────────
# RAG RETRIEVAL NODE
# ─────────────────────────────────────────────

def rag_node(
    state: AgentState
):

    analysis = state["analysis"]

    # Skip healthy campaigns
    if analysis.severity == Severity.OK:

        analysis.rag_context = []

        return {
            "analysis": analysis
        }

    # Retrieve strategies
    strategies = retrieve_strategies(

        faiss_index=state["faiss_index"],

        issues=analysis.issues,

        top_k=5
    )

    analysis.rag_context = strategies

    return {
        "analysis": analysis
    }


# ─────────────────────────────────────────────
# LLM REASONING NODE
# ─────────────────────────────────────────────

def llm_node(
    state: AgentState
):

    analysis = state["analysis"]

    # Skip healthy campaigns
    if analysis.severity == Severity.OK:

        analysis.explanation = (
            "Campaign performance is healthy. "
            "No optimization actions required."
        )

        analysis.recommendations = []

        return {
            "analysis": analysis
        }

    # ─────────────────────────
    # GENERATE AI EXPLANATION
    # ─────────────────────────

    explanation = generate_explanation(
        analysis
    )

    analysis.explanation = explanation

    # ─────────────────────────
    # EXTRACT RECOMMENDATIONS
    # ─────────────────────────

    recommendations = extract_recommendations(
        explanation
    )

    # Fallback recommendations
    if not recommendations:

        recommendations = [

            "Review campaign targeting settings",

            "Optimize creatives and messaging",

            "Analyze inventory quality and bid strategy"
        ]

    analysis.recommendations = recommendations

    return {
        "analysis": analysis
    }


# ─────────────────────────────────────────────
# BUILD LANGGRAPH
# ─────────────────────────────────────────────

def build_graph():

    graph = StateGraph(
        AgentState
    )

    # ─────────────────────────
    # ADD NODES
    # ─────────────────────────

    graph.add_node(
        "kpi_node",
        kpi_node
    )

    graph.add_node(
        "rag_node",
        rag_node
    )

    graph.add_node(
        "llm_node",
        llm_node
    )

    # ─────────────────────────
    # DEFINE FLOW
    # ─────────────────────────

    graph.set_entry_point(
        "kpi_node"
    )

    graph.add_edge(
        "kpi_node",
        "rag_node"
    )

    graph.add_edge(
        "rag_node",
        "llm_node"
    )

    graph.add_edge(
        "llm_node",
        END
    )

    # Compile graph
    return graph.compile()


# ─────────────────────────────────────────────
# INITIALIZE GRAPH
# ─────────────────────────────────────────────

GRAPH = build_graph()


# ─────────────────────────────────────────────
# RUN FULL PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(

    campaign: Campaign,

    faiss_index
):

    initial_state = {

        "campaign": campaign,

        "faiss_index": faiss_index,

        "analysis": None
    }

    final_state = GRAPH.invoke(
        initial_state
    )

    return final_state["analysis"]