import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

CTR_THRESHOLD = 0.3
CPM_THRESHOLD = 1.0
FILL_RATE_THRESHOLD = 70.0


# ─────────────────────────────────────────────
# Load and Clean CSV
# ─────────────────────────────────────────────

def load_campaign_data(csv_path):

    df = pd.read_csv(csv_path)

    # clean column names
    df.columns = df.columns.str.strip().str.lower()

    # remove empty rows
    df = df.dropna(how="all")

    # fill missing numeric values
    numeric_cols = [
        "impressions",
        "clicks",
        "revenue",
        "fill_rate"
    ]

    df[numeric_cols] = df[numeric_cols].fillna(0)

    # convert to numbers
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # calculate CTR
    df["ctr"] = df.apply(
        lambda row:
        round((row["clicks"] / row["impressions"]) * 100, 4)
        if row["impressions"] > 0 else 0,
        axis=1
    )

    # calculate CPM
    df["cpm"] = df.apply(
        lambda row:
        round((row["revenue"] / row["impressions"]) * 1000, 2)
        if row["impressions"] > 0 else 0,
        axis=1
    )

    return df


# ─────────────────────────────────────────────
# FAISS Vector Database
# ─────────────────────────────────────────────

class FAISSIndex:

    def __init__(self, knowledge_base_path):

        # embedding model
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.documents = []

        self.index = None

        self.build_index(knowledge_base_path)

    # ─────────────────────────────────────────

    def load_knowledge_base(self, path):

        documents = []

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = content.split("[DOCUMENT]")

        for section in sections:

            section = section.strip()

            if not section:
                continue

            lines = section.split("\n")

            doc = {}

            for line in lines:

                if line.startswith("issue:"):
                    doc["issue"] = line.replace("issue:", "").strip()

                elif line.startswith("strategy:"):
                    doc["strategy"] = line.replace("strategy:", "").strip()

                elif line.startswith("explanation:"):
                    doc["explanation"] = line.replace("explanation:", "").strip()

            if "issue" in doc:
                documents.append(doc)

        return documents

    # ─────────────────────────────────────────

    def build_index(self, path):

        self.documents = self.load_knowledge_base(path)

        texts = []

        for doc in self.documents:

            text = (
                f"{doc['issue']} "
                f"{doc['strategy']} "
                f"{doc['explanation']}"
            )

            texts.append(text)

        embeddings = self.model.encode(texts)

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

    # ─────────────────────────────────────────

    def search(self, query, top_k=3):

        query_embedding = self.model.encode([query])

        query_embedding = np.array(
            query_embedding,
            dtype="float32"
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx in indices[0]:

            if idx < len(self.documents):
                results.append(self.documents[idx])

        return results