import pandas as pd
from analyzer import AnalyzerConfig, ResumeAnalyzer

# Evaluation script to test model performance on labeled examples
# Uses classification metrics to validate scoring consistency
# sample evaluation cases
evaluation_cases = [
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
        "expected_level": "High"
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
        "expected_level": "Medium"
    },
    {
        "case_name": "Weak Match",
        "resume_text": """
        HTML, CSS, JavaScript, frontend, React, UI design
        """,
        "job_text": """
        Looking for a Data Scientist / NLP Engineer with Python, SQL, NLP, machine learning,
        scikit-learn, TensorFlow, and statistics.
        """,
        "expected_level": "Low"
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
        "expected_level": "High"
    },
    {
        "case_name": "Strong Business Intelligence Analyst Match",
        "resume_text": """
        SQL, Tableau, Power BI, Excel, dashboarding, reporting, KPIs,
        data visualization, stakeholder communication, business analytics
        """,
        "job_text": """
        Requirements: SQL, Tableau, Power BI, Excel, dashboards, reporting,
        KPIs, and stakeholder communication.
        Preferred: business analytics and data storytelling.
        """,
        "expected_level": "High"
    },
    {
        "case_name": "Strong Data Engineer Analyst Match",
        "resume_text": """
        SQL, Python, ETL, data pipelines, data wrangling, Snowflake,
        PostgreSQL, AWS, reporting, data modeling
        """,
        "job_text": """
        Requirements: SQL, Python, ETL, data pipelines, data wrangling,
        Snowflake, and data modeling.
        Preferred: AWS, PostgreSQL, and reporting experience.
        """,
        "expected_level": "High"
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
        "expected_level": "Medium"
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
        "expected_level": "Medium"
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
        "expected_level": "Medium"
    },
    {
        "case_name": "Weak Marketing Match",
        "resume_text": """
        social media marketing, content writing, SEO, campaign management,
        branding, customer engagement
        """,
        "job_text": """
        Requirements: SQL, Python, Tableau, Excel, dashboarding, KPIs,
        reporting, and data analysis.
        Preferred: Power BI and stakeholder communication.
        """,
        "expected_level": "Low"
    },
    {
        "case_name": "Weak Frontend Developer Match",
        "resume_text": """
        JavaScript, React, HTML, CSS, frontend development, UI design,
        responsive web design
        """,
        "job_text": """
        Requirements: SQL, Excel, Tableau, Power BI, reporting,
        dashboarding, KPIs, and data visualization.
        Preferred: business analytics and data storytelling.
        """,
        "expected_level": "Low"
    },
    {
        "case_name": "Weak HR Coordinator Match",
        "resume_text": """
        recruiting, onboarding, employee relations, HR policies,
        payroll coordination, interview scheduling
        """,
        "job_text": """
        Requirements: Python, SQL, machine learning, NLP, scikit-learn,
        TensorFlow, model evaluation, and statistics.
        Preferred: cloud workflows and data pipelines.
        """,
        "expected_level": "Low"
    }
]

def label_score(score):
    if score >= 0.75:
        return "High"
    elif score >= 0.45:
        return "Medium"
    return "Low"

config = AnalyzerConfig()
analyzer = ResumeAnalyzer(config)

results = []

for case in evaluation_cases:
    report = analyzer.analyze_texts(case["resume_text"], case["job_text"])
    predicted_level = label_score(report["final_score"])

    results.append({
        "case_name": case["case_name"],
        "expected_level": case["expected_level"],
        "predicted_level": predicted_level,
        "semantic_similarity": report["semantic_similarity"],
        "skill_overlap": report["skill_overlap"],
        "final_score": report["final_score"],
        "matched_count": report["matched_count"],
        "missing_count": report["missing_count"]
    })

df = pd.DataFrame(results)
print(df)

accuracy = (df["expected_level"] == df["predicted_level"]).mean()
print(f"\nEvaluation Accuracy: {accuracy:.2f}")

df.to_csv("evaluation_results.csv", index=False)
print("\nSaved results to evaluation_results.csv")

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(
    df["expected_level"],
    df["predicted_level"],
    labels=["Low", "Medium", "High"]
))