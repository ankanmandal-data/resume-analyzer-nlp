import json
import tempfile

import streamlit as st

from analyzer import AnalyzerConfig, ResumeAnalyzer, get_match_label


# ── Helpers ────────────────────────────────────────────────────────────────────

def render_items(items):
    if items:
        st.write(", ".join(items))
    else:
        st.write("None detected")


def render_recommendations(recommendations):
    for rec in recommendations:
        st.markdown(f"- {rec}")


def match_emoji(label: str) -> str:
    return {"Strong Match": "🟢", "Moderate Match": "🟡", "Weak Match": "🔴"}.get(label, "")


def _score_breakdown_table(report):
    """Score breakdown table — extracted so it can be rendered both inside and outside expanders."""
    st.markdown(
        """
| Component | Weight | Your Score |
|---|---|---|
| Semantic Similarity | 40% | `{}` |
| Required Skill Match | 30% | `{}` |
| Preferred Skill Match | 15% | `{}` |
| Experience / Profile Fit | 15% | `{}` |

**Formula:** `Final = 0.40 × Semantic + 0.30 × Required + 0.15 × Preferred + 0.15 × Experience`

Scores are normalized between 0 and 1. All components are approximate signals —
the final score is intended as a starting point for review, not a definitive ranking.
        """.format(
            report["semantic_similarity"],
            report["required_skill_match"],
            report["preferred_skill_match"],
            report["experience_profile_fit"],
        )
    )


def render_score_metrics(report):
    """Full score metrics block — use at top level (not inside another expander)."""
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Final Score",      report["final_score"])
    col2.metric("Semantic",         report["semantic_similarity"])
    col3.metric("TF-IDF Baseline",  report.get("tfidf_baseline_similarity", 0))
    col4.metric("Required Skills",  report["required_skill_match"])
    col5.metric("Preferred Skills", report["preferred_skill_match"])
    col6.metric("Experience Fit",   report["experience_profile_fit"])

    label = report.get("match_label", get_match_label(report["final_score"]))
    st.markdown(f"### {match_emoji(label)} {label}")
    st.progress(min(report["final_score"], 1.0))

    yoe = report.get("years_of_experience_detected")
    if yoe:
        st.caption(f"📅 Years of experience detected in resume: approximately {yoe}")

    # Nested expander is fine here — this function is called at top level
    with st.expander("How this score is calculated"):
        _score_breakdown_table(report)


def render_score_metrics_flat(report):
    """
    Same as render_score_metrics but WITHOUT the inner expander.
    Use this when already inside a st.expander — Streamlit doesn't allow nesting.
    """
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Final Score",      report["final_score"])
    col2.metric("Semantic",         report["semantic_similarity"])
    col3.metric("TF-IDF Baseline",  report.get("tfidf_baseline_similarity", 0))
    col4.metric("Required Skills",  report["required_skill_match"])
    col5.metric("Preferred Skills", report["preferred_skill_match"])
    col6.metric("Experience Fit",   report["experience_profile_fit"])

    label = report.get("match_label", get_match_label(report["final_score"]))
    st.markdown(f"### {match_emoji(label)} {label}")
    st.progress(min(report["final_score"], 1.0))

    yoe = report.get("years_of_experience_detected")
    if yoe:
        st.caption(f"📅 Years of experience detected in resume: approximately {yoe}")

    # No expander here — render the table directly
    _score_breakdown_table(report)


def render_skill_sections(report):
    st.subheader("✅ Required Skills — Matched")
    render_items(report["matched_required_skills"])

    st.subheader("❌ Required Skills — Missing")
    render_items(report["missing_required_skills"])

    st.subheader("⭐ Preferred Skills — Matched")
    render_items(report["matched_preferred_skills"])

    st.subheader("⚠️ Preferred Skills — Not Detected")
    render_items(report["missing_preferred_skills"])

    st.subheader("💡 Suggestions")
    render_recommendations(report["recommendations"])


def render_explainability(report):
    st.subheader("🔎 Score Explainability")

    tfidf   = report.get("tfidf_baseline_similarity", 0)
    sem     = report["semantic_similarity"]
    lift    = report.get("semantic_lift_over_tfidf", 0)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("TF-IDF Baseline",  tfidf)
    col_b.metric("Semantic Score",   sem)
    col_c.metric("Semantic Lift",    lift,
                 delta=f"+{lift}" if lift > 0 else str(lift),
                 delta_color="normal")

    if lift > 0.15:
        st.caption(
            "The embedding model scores noticeably higher than the keyword baseline here, "
            "which suggests it may be detecting relevant context or paraphrased skills "
            "that keyword matching would miss."
        )
    elif lift > 0:
        st.caption(
            "The embedding score is slightly above the keyword baseline. "
            "Both approaches seem to be picking up on similar signals."
        )
    else:
        st.caption(
            "The semantic score is close to or below the keyword baseline. "
            "This can happen when the resume and JD share a lot of exact terminology."
        )

    st.subheader("Top Matching Terms")
    if report["top_matching_terms"]:
        st.table([{"term": t, "tfidf weight": round(s, 4)}
                  for t, s in report["top_matching_terms"]])
        st.caption(
            "These terms appear in both the resume and job description with high TF-IDF weight. "
            "They are likely contributing to the match score, though TF-IDF weight reflects "
            "term importance in the document, not direct causal impact on the final score."
        )
    else:
        st.write("No strong overlapping terms found.")


# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_analyzer(model_name: str) -> ResumeAnalyzer:
    return ResumeAnalyzer(AnalyzerConfig(model_name=model_name))


# ── Page layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 Resume Analyzer")
st.write(
    "Upload a resume PDF and paste one or more job descriptions to get a structured "
    "match report — including skill coverage, semantic similarity, and an explainability "
    "breakdown comparing keyword vs semantic matching."
)

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Embedding model",
        options=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0,
    )
    st.caption(
        "**all-MiniLM-L6-v2** — faster, lighter, generally good for sentence-level tasks.  \n"
        "**all-mpnet-base-v2** — slower but tends to produce slightly better embeddings."
    )
    st.divider()
    st.markdown(
        "**Score thresholds (approximate)**  \n"
        "🟢 Strong — ≥ 0.75  \n"
        "🟡 Moderate — 0.45–0.74  \n"
        "🔴 Weak — < 0.45"
    )

uploaded_pdf = st.file_uploader("Upload Resume PDF", type=["pdf"])
job_text = st.text_area(
    "Paste Job Description(s)",
    height=220,
    help="For batch mode, separate multiple job descriptions with ===JOB===",
)
run = st.button("Analyze", type="primary", use_container_width=True)


# ── Analysis ───────────────────────────────────────────────────────────────────

if run:
    if not uploaded_pdf or not job_text.strip():
        st.error("Please upload a resume PDF and paste at least one job description.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    analyzer = load_analyzer(model_name)

    job_blocks = [j.strip() for j in job_text.split("===JOB===") if j.strip()]

    with st.spinner(
        f"Analyzing {'1 job description' if len(job_blocks) == 1 else f'{len(job_blocks)} job descriptions'}..."
    ):
        reports = [analyzer.analyze_pdf(pdf_path, jb) for jb in job_blocks]

    # ── Single job ─────────────────────────────────────────────────────────────
    if len(reports) == 1:
        report = reports[0]
        render_score_metrics(report)   # top-level — expander inside is fine
        st.divider()

        left, right = st.columns(2)
        with left:
            render_skill_sections(report)
        with right:
            render_explainability(report)

        st.download_button(
            label="⬇️ Download Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="resume_match_report.json",
            mime="application/json",
        )
        with st.expander("Full JSON output"):
            st.json(report)

    # ── Batch mode ─────────────────────────────────────────────────────────────
    else:
        st.subheader("Batch Results")

        batch_rows = []
        for i, r in enumerate(reports, start=1):
            label = r.get("match_label", get_match_label(r["final_score"]))
            batch_rows.append({
                "Job":              f"Job {i}",
                "Match":            f"{match_emoji(label)} {label}",
                "Final Score":      r["final_score"],
                "Semantic":         r["semantic_similarity"],
                "TF-IDF Baseline":  r.get("tfidf_baseline_similarity", 0),
                "Semantic Lift":    r.get("semantic_lift_over_tfidf", 0),
                "Required Skills":  r["required_skill_match"],
                "Preferred Skills": r["preferred_skill_match"],
                "Experience Fit":   r["experience_profile_fit"],
                "Matched":          r["matched_count"],
                "Missing":          r["missing_count"],
            })

        st.dataframe(batch_rows, use_container_width=True)

        best_idx = max(range(len(reports)), key=lambda i: reports[i]["final_score"])
        best = reports[best_idx]
        best_label = best.get("match_label", get_match_label(best["final_score"]))
        st.success(
            f"Highest scoring: Job {best_idx + 1} — "
            f"{match_emoji(best_label)} {best_label} (Score: {best['final_score']})"
        )

        render_score_metrics(best)    # top-level — expander inside is fine
        st.divider()

        left, right = st.columns(2)
        with left:
            render_skill_sections(best)
        with right:
            render_explainability(best)

        st.download_button(
            label="⬇️ Download Batch Report (JSON)",
            data=json.dumps(reports, indent=2),
            file_name="resume_batch_report.json",
            mime="application/json",
        )

        st.divider()
        st.subheader("All Job Reports")
        for i, r in enumerate(reports, start=1):
            label = r.get("match_label", get_match_label(r["final_score"]))
            with st.expander(f"Job {i} — {match_emoji(label)} {label}  (Score: {r['final_score']})"):
                # Using _flat variant here — Streamlit doesn't allow expanders inside expanders
                render_score_metrics_flat(r)
                ld, rd = st.columns(2)
                with ld:
                    render_skill_sections(r)
                with rd:
                    render_explainability(r)

        with st.expander("Full JSON output"):
            st.json(reports)