from app.ingest import FAISSIndex
from app.models import Issue


# ─────────────────────────────────────────────
# BUILD QUERY
# ─────────────────────────────────────────────

def build_query(
    issues: list[Issue]
):

    if not issues:
        return ""

    issue_names = [
        issue.issue_type
        for issue in issues
    ]

    # Multi-issue query
    return " and ".join(issue_names)


# ─────────────────────────────────────────────
# SCORE STRATEGY RELEVANCE
# ─────────────────────────────────────────────

def calculate_relevance(

    strategy_issue: str,

    detected_issues: list[str]
):

    score = 0

    # Exact multi-issue match
    if strategy_issue.lower() == " and ".join(
        detected_issues
    ).lower():

        score += 10

    # Partial matches
    for issue in detected_issues:

        if issue.lower() in strategy_issue.lower():

            score += 3

    return score


# ─────────────────────────────────────────────
# RETRIEVE STRATEGIES
# ─────────────────────────────────────────────

def retrieve_strategies(

    faiss_index: FAISSIndex,

    issues: list[Issue],

    top_k=5
):

    if not issues:
        return []

    # Build query
    query = build_query(
        issues
    )

    # Get issue names
    detected_issues = [

        issue.issue_type

        for issue in issues
    ]

    # Retrieve more candidates
    raw_results = faiss_index.search(

        query=query,

        top_k=10
    )

    # ─────────────────────────
    # SCORE RESULTS
    # ─────────────────────────

    scored_results = []

    for doc in raw_results:

        relevance_score = calculate_relevance(

            strategy_issue=doc["issue"],

            detected_issues=detected_issues
        )

        scored_results.append({

            "doc": doc,

            "score": relevance_score
        })

    # Sort by relevance
    scored_results = sorted(

        scored_results,

        key=lambda x: x["score"],

        reverse=True
    )

    # ─────────────────────────
    # DEDUPLICATION
    # ─────────────────────────

    unique_strategies = set()

    final_results = []

    for item in scored_results:

        doc = item["doc"]

        strategy = doc["strategy"]

        # Skip duplicates
        if strategy in unique_strategies:
            continue

        unique_strategies.add(strategy)

        formatted = (

            f"Issue: {doc['issue']}\n"

            f"Strategy: {doc['strategy']}\n"

            f"Explanation: {doc['explanation']}"
        )

        final_results.append(formatted)

        # Limit final results
        if len(final_results) >= top_k:
            break

    return final_results


# ─────────────────────────────────────────────
# FORMAT RAG CONTEXT
# ─────────────────────────────────────────────

def format_rag_context(

    strategies: list[str]
):

    if not strategies:

        return (
            "No optimization strategies found."
        )

    return "\n\n".join(strategies)