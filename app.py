import json
import tempfile

import streamlit as st

from analyzer import AnalyzerConfig, ResumeAnalyzer

def render_items(items):
    if items:
        st.write(", ".join(items))
    else:
        st.write("None detected")

def render_recommendations(recommendations):
    if recommendations:
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.write("No recommendations available.")


def get_match_label(score):
    if score >= 0.75:
        return "🟢 Strong Match"
    elif score >= 0.45:
        return "🟡 Moderate Match"
    else:
        return "🔴 Weak Match"
    

def render_score_metrics(report):
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Final Score", report["final_score"])
    col2.metric("Semantic", report["semantic_similarity"])
    col3.metric("Required Skills", report["required_skill_match"])
    col4.metric("Preferred Skills", report["preferred_skill_match"])
    col5.metric("Experience Fit", report["experience_profile_fit"])
    st.markdown(f"### {get_match_label(report['final_score'])}")


def render_skill_sections(report):
    st.subheader("✅ Required Skills Match")
    render_items(report["matched_required_skills"])

    st.subheader("❌ Missing Required Skills")
    render_items(report["missing_required_skills"])

    st.subheader("⭐ Preferred Skills Match")
    render_items(report["matched_preferred_skills"])

    st.subheader("⚠️ Missing Preferred Skills")
    render_items(report["missing_preferred_skills"])

    st.subheader("💡 Recommendations")
    render_recommendations(report["recommendations"])


def render_explainability(report):
    st.subheader("🔎 Top Matching Terms (Explainability)")
    if report["top_matching_terms"]:
        st.table(
            [{"term": t, "importance": s} for t, s in report["top_matching_terms"]]
        )
    else:
        st.write("No strong overlapping TF-IDF terms found.")


@st.cache_resource
def load_analyzer(config):
    return ResumeAnalyzer(config)


st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 Resume Analyzer (Structured NLP Matching)")
st.write(
    "Upload a resume PDF, paste one or more job descriptions, and get a structured match report with required skills, preferred skills, and experience fit."
)

with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "Embedding model",
        options=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0,
    )

    st.caption(
        "Final score combines semantic similarity, required skills, preferred skills, and experience/profile fit."
    )

# Streamlit UI for uploading resume and entering job descriptions
uploaded_pdf = st.file_uploader("Upload Resume PDF", type=["pdf"])

job_text = st.text_area(
    "Paste Job Description(s)",
    height=220,
    help="For batch mode, separate multiple job descriptions using: ===JOB===",
)

run = st.button("Analyze", type="primary", use_container_width=True)

if run:
    if not uploaded_pdf or not job_text.strip():
        st.error("Please upload a PDF and paste a job description.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    config = AnalyzerConfig(model_name=model_name)
    analyzer = load_analyzer(config)

    with st.spinner("Analyzing..."):
        job_blocks = [j.strip() for j in job_text.split("===JOB===") if j.strip()]
        reports = [analyzer.analyze_pdf(pdf_path, jb) for jb in job_blocks]
    
    if len(reports) == 1:
        report = reports[0]

        render_score_metrics(report)

        st.divider()

        left, right = st.columns([1, 1])

        with left:
            render_skill_sections(report)

        with right:
            render_explainability(report)

        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="Download Report (JSON)",
            data=report_json,
            file_name="resume_match_report.json",
            mime="application/json",
        )

        with st.expander("Debug / Details"):
            st.json(report)

    else:
        st.subheader("Batch Results")

        batch_rows = []
        for i, report in enumerate(reports, start=1):
            batch_rows.append(
                {
                    "Job": f"Job {i}",
                    "Final Score": report["final_score"],
                    "Semantic": report["semantic_similarity"],
                    "Required Skills": report["required_skill_match"],
                    "Preferred Skills": report["preferred_skill_match"],
                    "Experience Fit": report["experience_profile_fit"],
                    "Matched Skills": report["matched_count"],
                    "Missing Skills": report["missing_count"],
                }
            )

        st.dataframe(batch_rows, use_container_width=True)

        best_idx = max(range(len(reports)), key=lambda i: reports[i]["final_score"])
        best_report = reports[best_idx]

        st.success(
            f"Best match: Job {best_idx + 1} with Final Score = {best_report['final_score']}"
        )

        render_score_metrics(best_report)

        st.divider()

        left, right = st.columns([1, 1])

        with left:
            render_skill_sections(best_report)

        with right:
            render_explainability(best_report)

        batch_json = json.dumps(reports, indent=2)
        st.download_button(
            label="Download Batch Report (JSON)",
            data=batch_json,
            file_name="resume_batch_match_report.json",
            mime="application/json",
        )

        st.divider()
        st.subheader("Detailed Reports for All Jobs")

        for i, report in enumerate(reports, start=1):
            with st.expander(f"Job {i} Detailed Report"):
                render_score_metrics(report)

                left_detail, right_detail = st.columns([1, 1])

                with left_detail:
                    render_skill_sections(report)

                with right_detail:
                    render_explainability(report)

        with st.expander("Debug / Details"):
            st.json(reports)