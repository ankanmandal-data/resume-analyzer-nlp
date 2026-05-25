import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from analyzer import AnalyzerConfig, ResumeAnalyzer, get_match_label

# Manually labeled test cases covering a range of match levels.
# Resume and JD texts are simplified skill lists to keep the evaluation
# reproducible without real PDF files. Scores on real-world resumes
# may differ due to formatting variability and richer context.
EVALUATION_CASES = [
    {
        "case_name": "Strong NLP/Data Science Match",
        "resume_text": """
            Python, SQL, NLP, machine learning, scikit-learn, TensorFlow, pandas, numpy,
            matplotlib, seaborn, classification, regression, statistics
        """,
        "job_text": """
            Looking for a Data Scientist / NLP Engineer with Python, SQL, NLP, machine learning,
            scikit-learn, TensorFlow, pandas, numpy, statistics, and model evaluation.
        """,
        "expected_level": "High",
    },
    {
        "case_name": "Moderate Data Science Match",
        "resume_text": """
            Python, pandas, numpy, regression, classification, matplotlib
        """,
        "job_text": """
            Looking for a Data Scientist with Python, SQL, machine learning, NLP,
            scikit-learn, pandas, numpy, and statistics.
        """,
        "expected_level": "Medium",
    },
    {
        "case_name": "Weak Match — Frontend Developer",
        "resume_text": """
            HTML, CSS, JavaScript, frontend, React, UI design
        """,
        "job_text": """
            Looking for a Data Scientist / NLP Engineer with Python, SQL, NLP, machine learning,
            scikit-learn, TensorFlow, and statistics.
        """,
        "expected_level": "Low",
    },
    {
        "case_name": "Strong Data Analyst Match",
        "resume_text": """
            SQL, Excel, Tableau, Power BI, data analysis, reporting, dashboarding,
            KPIs, stakeholder communication, data visualization
        """,
        "job_text": """
            Looking for a Data Analyst with SQL, Excel, Tableau, Power BI,
            dashboards, KPIs, reporting, and stakeholder communication.
        """,
        "expected_level": "High",
    },
    {
        "case_name": "Strong BI Analyst Match",
        "resume_text": """
            SQL, Tableau, Power BI, Excel, dashboarding, reporting, KPIs,
            data visualization, stakeholder communication, business analytics
        """,
        "job_text": """
            Requirements: SQL, Tableau, Power BI, Excel, dashboards, reporting,
            KPIs, and stakeholder communication.
            Preferred: business analytics and data storytelling.
        """,
        "expected_level": "High",
    },
    {
        "case_name": "Strong Data Engineer Match",
        "resume_text": """
            SQL, Python, ETL, data pipelines, data wrangling, Snowflake,
            PostgreSQL, AWS, reporting, data modeling
        """,
        "job_text": """
            Requirements: SQL, Python, ETL, data pipelines, data wrangling,
            Snowflake, and data modeling.
            Preferred: AWS, PostgreSQL, and reporting experience.
        """,
        "expected_level": "High",
    },
    {
        "case_name": "Moderate Data Analyst Match",
        "resume_text": """
            SQL, Excel, reporting, data cleaning, dashboards, communication
        """,
        "job_text": """
            Requirements: SQL, Excel, Tableau, Power BI, KPIs, dashboarding,
            and stakeholder communication.
            Preferred: data storytelling and business analytics.
        """,
        "expected_level": "Medium",
    },
    {
        "case_name": "Moderate BI Analyst Match",
        "resume_text": """
            Excel, Power BI, reporting, dashboarding, metrics, communication
        """,
        "job_text": """
            Requirements: SQL, Tableau, Power BI, Excel, KPIs, dashboarding,
            and data visualization.
            Preferred: DAX, Power Query, and stakeholder communication.
        """,
        "expected_level": "Medium",
    },
    {
        "case_name": "Moderate ML Analyst Match",
        "resume_text": """
            Python, pandas, numpy, statistics, data analysis, regression,
            matplotlib, seaborn
        """,
        "job_text": """
            Requirements: Python, SQL, machine learning, statistics,
            scikit-learn, model evaluation, and data visualization.
            Preferred: TensorFlow, NLP, and cloud experience.
        """,
        "expected_level": "Medium",
    },
    {
        "case_name": "Weak Match — Marketing",
        "resume_text": """
            social media marketing, content writing, SEO, campaign management,
            branding, customer engagement
        """,
        "job_text": """
            Requirements: SQL, Python, Tableau, Excel, dashboarding, KPIs,
            reporting, and data analysis.
            Preferred: Power BI and stakeholder communication.
        """,
        "expected_level": "Low",
    },
    {
        "case_name": "Weak Match — Frontend (DA JD)",
        "resume_text": """
            JavaScript, React, HTML, CSS, frontend development, UI design,
            responsive web design
        """,
        "job_text": """
            Requirements: SQL, Excel, Tableau, Power BI, reporting,
            dashboarding, KPIs, and data visualization.
            Preferred: business analytics and data storytelling.
        """,
        "expected_level": "Low",
    },
    {
        "case_name": "Weak Match — HR Coordinator",
        "resume_text": """
            recruiting, onboarding, employee relations, HR policies,
            payroll coordination, interview scheduling
        """,
        "job_text": """
            Requirements: Python, SQL, machine learning, NLP, scikit-learn,
            TensorFlow, model evaluation, and statistics.
            Preferred: cloud workflows and data pipelines.
        """,
        "expected_level": "Low",
    },
]


def run_evaluation():
    config = AnalyzerConfig()
    analyzer = ResumeAnalyzer(config)

    rows = []
    for case in EVALUATION_CASES:
        report = analyzer.analyze_texts(case["resume_text"], case["job_text"])
        predicted = get_match_label(report["final_score"]).split()[0]  # "High"/"Medium"/"Low" only
        # Normalize label to match expected format
        label_map = {"Strong": "High", "Moderate": "Medium", "Weak": "Low"}
        predicted_level = label_map.get(predicted, predicted)

        rows.append({
            "case_name":                 case["case_name"],
            "expected_level":            case["expected_level"],
            "predicted_level":           predicted_level,
            "semantic_similarity":       report["semantic_similarity"],
            "tfidf_baseline_similarity": report["tfidf_baseline_similarity"],
            "semantic_lift_over_tfidf":  report["semantic_lift_over_tfidf"],
            "required_skill_match":      report["required_skill_match"],
            "preferred_skill_match":     report["preferred_skill_match"],
            "experience_profile_fit":    report["experience_profile_fit"],
            "skill_overlap":             report["skill_overlap"],
            "final_score":               report["final_score"],
            "matched_count":             report["matched_count"],
            "missing_count":             report["missing_count"],
        })

    df = pd.DataFrame(rows)
    print(df[["case_name", "expected_level", "predicted_level", "final_score"]].to_string())

    accuracy = (df["expected_level"] == df["predicted_level"]).mean()
    print(f"\nAccuracy: {accuracy:.2f} ({int(accuracy * len(df))}/{len(df)} correct)")
    print(
        "\nNote: evaluation uses simplified skill-list test cases rather than real resumes. "
        "Results on real-world PDFs may vary."
    )

    print("\nClassification Report:")
    print(classification_report(
        df["expected_level"],
        df["predicted_level"],
        labels=["Low", "Medium", "High"],
    ))

    cm = confusion_matrix(
        df["expected_level"],
        df["predicted_level"],
        labels=["Low", "Medium", "High"],
    )
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Low", "Actual Medium", "Actual High"],
        columns=["Predicted Low", "Predicted Medium", "Predicted High"],
    )
    print("\nConfusion Matrix:")
    print(cm_df)

    df.to_csv("evaluation_results.csv", index=False)
    cm_df.to_csv("evaluation_confusion_matrix.csv")
    print("\nSaved: evaluation_results.csv, evaluation_confusion_matrix.csv")


if __name__ == "__main__":
    run_evaluation()
