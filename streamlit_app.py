import streamlit as st
import pandas as pd
import json

from app.ingest import (
    load_campaign_data,
    FAISSIndex
)

from app.models import Campaign

from app.agent import run_pipeline


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AdOps AI Agent",
    page_icon="📊",
    layout="wide"
)


# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────

st.title("📊 AdOps Intelligence Agent")

st.markdown("""
AI-powered campaign monitoring system using:

- LangGraph
- RAG (FAISS)
- Groq Llama 3.1
- KPI Analysis
""")


# ─────────────────────────────────────────────
# LOAD FAISS KNOWLEDGE BASE
# ─────────────────────────────────────────────

@st.cache_resource
def load_faiss():

    return FAISSIndex(
        "data/knowledge_base.txt"
    )


faiss_index = load_faiss()


# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload Campaign CSV",
    type=["csv"]
)


# ─────────────────────────────────────────────
# MAIN PROCESSING
# ─────────────────────────────────────────────

if uploaded_file:

    # Load CSV
    df = load_campaign_data(
        uploaded_file
    )

    st.subheader("📁 Campaign Data")

    st.dataframe(
        df,
        use_container_width=True
    )

    # Run Button
    if st.button("🚀 Run AI Analysis"):

        results = []

        progress_bar = st.progress(0)

        status_text = st.empty()

        total = len(df)

        # ─────────────────────────
        # PROCESS CAMPAIGNS
        # ─────────────────────────

        for idx, row in df.iterrows():

            status_text.text(
                f"Processing {row['campaign_name']}..."
            )

            # Create Campaign object
            campaign = Campaign(

                campaign_id=row["campaign_id"],

                campaign_name=row["campaign_name"],

                impressions=row["impressions"],

                clicks=row["clicks"],

                revenue=row["revenue"],

                fill_rate=row["fill_rate"],

                ctr=row["ctr"],

                cpm=row["cpm"]
            )

            # Run AI pipeline
            analysis = run_pipeline(
                campaign,
                faiss_index
            )

            results.append(analysis)

            progress = int(
                ((idx + 1) / total) * 100
            )

            progress_bar.progress(progress)

        status_text.success(
            "✅ Analysis completed successfully!"
        )

        # ─────────────────────────
        # SUMMARY METRICS
        # ─────────────────────────

        st.subheader("📈 Analysis Summary")

        critical_count = len([
            r for r in results
            if str(r.severity) == "Severity.CRITICAL"
        ])

        high_count = len([
            r for r in results
            if str(r.severity) == "Severity.HIGH"
        ])

        medium_count = len([
            r for r in results
            if str(r.severity) == "Severity.MEDIUM"
        ])

        ok_count = len([
            r for r in results
            if str(r.severity) == "Severity.OK"
        ])

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🚨 Critical",
                critical_count
            )

        with col2:
            st.metric(
                "⚠ High",
                high_count
            )

        with col3:
            st.metric(
                "🟡 Medium",
                medium_count
            )

        with col4:
            st.metric(
                "✅ Healthy",
                ok_count
            )

        # ─────────────────────────
        # CAMPAIGN RESULTS
        # ─────────────────────────

        st.subheader("🧠 Campaign Analysis")

        for result in results:

            with st.expander(
                f"{result.campaign_name} — {result.severity}"
            ):

                # KPIs
                st.write(
                    f"CTR: {result.ctr:.2f}%"
                )

                st.write(
                    f"CPM: ${result.cpm:.2f}"
                )

                st.write(
                    f"Fill Rate: {result.fill_rate:.2f}%"
                )

                # ─────────────────────
                # ISSUES
                # ─────────────────────

                st.markdown("### 🚨 Issues")

                if result.issues:

                    for issue in result.issues:

                        st.write(
                            f"- {issue.issue_type}"
                        )

                else:

                    st.success(
                        "No issues detected"
                    )

                # ─────────────────────
                # AI EXPLANATION
                # ─────────────────────

                st.markdown(
                    "### 🤖 AI Explanation"
                )

                st.info(
                    result.explanation
                )

                # ─────────────────────
                # RECOMMENDATIONS
                # ─────────────────────

                st.markdown(
                    "### ✅ Recommendations"
                )

                if result.recommendations:

                    for rec in result.recommendations:

                        st.success(rec)

                else:

                    st.write(
                        "No recommendations needed."
                    )
        # ─────────────────────────
        # EXPORT REPORTS
        # ─────────────────────────

        st.subheader("📥 Export Reports")

        export_data = []

        for result in results:

            export_data.append({

                "campaign_name": result.campaign_name,

                "severity": str(result.severity),

                "ctr": result.ctr,

                "cpm": result.cpm,

                "fill_rate": result.fill_rate,

                "issues": ", ".join([
                    issue.issue_type
                    for issue in result.issues
                ]),

                "recommendations": "\n".join(
                    result.recommendations
                ),

                "explanation": result.explanation
            })

        export_df = pd.DataFrame(
            export_data
        )

        # ─────────────────────────
        # DOWNLOAD BUTTONS
        # ─────────────────────────

        col1, col2 = st.columns(2)

        # CSV DOWNLOAD
        with col1:

            csv = export_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                label="⬇ Download CSV Report",

                data=csv,

                file_name="adops_analysis_report.csv",

                mime="text/csv",

                key="csv_report_button"
            )

        # JSON DOWNLOAD
        with col2:

            json_data = json.dumps(
                export_data,
                indent=2
            )

            st.download_button(

                label="⬇ Download JSON Report",

                data=json_data,

                file_name="adops_analysis_report.json",

                mime="application/json",

                key="json_report_button"
            )