import re
import pandas as pd

CTR_THRESHOLD = 0.3
CPM_THRESHOLD = 1.0
FILL_RATE_THRESHOLD = 70.0


# ─────────────────────────────────────────────
# Load and Clean CSV  (unchanged)
# ─────────────────────────────────────────────

def load_campaign_data(csv_path):

    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(how="all")

    numeric_cols = ["impressions", "clicks", "revenue", "fill_rate"]
    df[numeric_cols] = df[numeric_cols].fillna(0)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["ctr"] = df.apply(
        lambda row: round((row["clicks"] / row["impressions"]) * 100, 4)
        if row["impressions"] > 0 else 0,
        axis=1
    )

    df["cpm"] = df.apply(
        lambda row: round((row["revenue"] / row["impressions"]) * 1000, 2)
        if row["impressions"] > 0 else 0,
        axis=1
    )

    return df


# ─────────────────────────────────────────────
# FAISSIndex — Keyword-based replacement
# ─────────────────────────────────────────────
#
# ROOT CAUSE OF OOM:
#   sentence-transformers("all-MiniLM-L6-v2") downloads and loads
#   a 90MB neural network model into RAM at import time.
#   Combined with FastAPI + pandas + groq, this exceeded Render's
#   512MB free tier limit before the server even started.
#
# THE FIX:
#   Replace FAISS + sentence-transformers with keyword scoring.
#   For a knowledge base of 12-15 entries, keyword matching is
#   equally accurate and uses ~5MB RAM instead of ~400MB.
#
# ZERO CHANGES NEEDED elsewhere — same class name, same .search()
# interface, same return format. rag.py and agent.py unchanged.
# ─────────────────────────────────────────────

class FAISSIndex:
    """
    Drop-in replacement for the FAISS + sentence-transformers index.

    Identical interface:
      index = FAISSIndex("data/knowledge_base.txt")
      results = index.search("Low CTR and Low CPM", top_k=3)
      # returns list of dicts: [{issue, strategy, explanation}, ...]

    Uses keyword scoring instead of vector embeddings.
    RAM usage: ~5MB vs ~400MB for sentence-transformers.
    Startup time: instant vs ~8 seconds for model download.
    """

    def __init__(self, knowledge_base_path):
        self.documents = []
        self._load(knowledge_base_path)
        print(f"✅ Knowledge base loaded: {len(self.documents)} entries")

    def _load(self, path):
        """Parses knowledge_base.txt — supports [DOCUMENT] and [HEADER] formats."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Support both separator styles
        if "[DOCUMENT]" in content:
            sections = content.split("[DOCUMENT]")
        else:
            sections = re.split(r"\[[\w_]+\]", content)

        for section in sections:
            section = section.strip()
            if not section or section.startswith("#"):
                continue

            doc = {}
            for line in section.split("\n"):
                line = line.strip()
                if line.startswith("issue:"):
                    doc["issue"] = line.replace("issue:", "").strip()
                elif line.startswith("strategy:"):
                    doc["strategy"] = line.replace("strategy:", "").strip()
                elif line.startswith("explanation:"):
                    doc["explanation"] = line.replace("explanation:", "").strip()

            if "issue" in doc and "strategy" in doc:
                self.documents.append(doc)

    def search(self, query: str, top_k: int = 3) -> list:
        """
        Keyword-based relevance scoring.

        Scoring:
          +10  doc's issue type found in query  (e.g. "Low CTR" in "Low CTR and Low CPM")
          +5   individual issue word matches
          +2   query words found in strategy or explanation text
        """
        if not query or not self.documents:
            return self.documents[:top_k]

        query_lower = query.lower()
        scored = []

        for doc in self.documents:
            score = 0
            doc_issue       = doc.get("issue", "").lower()
            doc_strategy    = doc.get("strategy", "").lower()
            doc_explanation = doc.get("explanation", "").lower()

            # Exact issue phrase in query
            if doc_issue in query_lower:
                score += 10

            # Individual words from issue in query
            for word in doc_issue.split():
                if len(word) > 3 and word in query_lower:
                    score += 5

            # Query words in strategy / explanation
            for word in query_lower.split():
                if len(word) > 3:
                    if word in doc_strategy or word in doc_explanation:
                        score += 2

            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
