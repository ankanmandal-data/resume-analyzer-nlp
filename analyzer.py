import re
import fitz
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------
# Text extraction + cleaning
# ----------------------------
def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def clean_text_basic(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_skills(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[/,_|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text




# ----------------------------
# Skill alias dictionary
# ----------------------------
# Map different variations of skills to a canonical form
# Example: "ml" → "machine learning"
SKILL_ALIASES = {
    "python": [r"\bpython\b", r"\bpython3\b"],
    "sql": [r"\bsql\b"],
    "machine learning": [r"\bmachine learning\b", r"\bml\b"],
    "deep learning": [r"\bdeep learning\b", r"\bdl\b", r"\bneural network(s)?\b"],
    "nlp": [r"\bnlp\b", r"\bnatural language processing\b"],
    "statistics": [r"\bstatistics\b", r"\bstatistical\b", r"\bstat(s)?\b"],
    "data analysis": [r"\bdata analysis\b", r"\bdata analyst\b", r"\banalytics\b"],
    "classification": [r"\bclassification\b", r"\bclassifier\b"],
    "regression": [r"\bregression\b"],
    "model evaluation": [r"\bmodel evaluation\b", r"\bevaluation\b", r"\bmetrics\b"],
    "scikit-learn": [r"\bscikit[-_ ]?learn\b", r"\bsklearn\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "tensorflow": [r"\btensorflow\b", r"\bkeras\b"],
    "pytorch": [r"\bpytorch\b", r"\btorch\b"],
    "matplotlib": [r"\bmatplotlib\b"],
    "seaborn": [r"\bseaborn\b"],
    "mysql": [r"\bmysql\b"],
    "git": [r"\bgit\b", r"\bgithub\b"],
    "jupyter": [r"\bjupyter\b", r"\bjupyter notebook\b"],
    "excel": [r"\bexcel\b", r"\bmicrosoft excel\b"],
    "tableau": [r"\btableau\b"],
    "power bi": [r"\bpower\s*bi\b"],
    "a/b testing": [r"\ba/b testing\b", r"\bab testing\b", r"\ba b testing\b"],
    "dashboarding": [r"\bdashboarding\b", r"\bdashboard(s)?\b"],
    "data visualization": [
        r"\bdata visualization\b",
        r"\bvisualization\b",
        r"\bvisualisation\b",
        r"\bmatplotlib\b",
        r"\bseaborn\b",
        r"\btableau\b",
        r"\bpower\s*bi\b",
    ],
    "data storytelling": [r"\bdata storytelling\b", r"\bstorytelling\b"],
    "business analytics": [r"\bbusiness analytics\b", r"\bbusiness analysis\b"],
    "predictive modeling": [
        r"\bpredictive modeling\b",
        r"\bpredictive modelling\b",
        r"\bpredictive models?\b",
    ],
    "data cleaning": [r"\bdata cleaning\b", r"\bcleaning\b"],
    "reporting": [r"\breporting\b", r"\breports?\b"],
    "api": [r"\bapi\b", r"\bapis\b"],
    "cloud": [r"\bcloud\b", r"\bcloud-based\b", r"\bcloud workflows?\b"],
    "spark": [r"\bspark\b", r"\bpyspark\b"],
    "aws": [r"\baws\b", r"\bamazon web services\b"],
    "s3": [r"\bs3\b", r"\bamazon s3\b"],
    "ec2": [r"\bec2\b"],
    "cloudfront": [r"\bcloudfront\b"],
    "model deployment": [r"\bmodel deployment\b", r"\bdeployment\b", r"\bdeploy\b"],
    "model monitoring": [r"\bmodel monitoring\b", r"\bmonitoring\b"],
    "mlops": [r"\bmlops\b", r"\bmachine learning operations\b"],
    "data pipelines": [r"\bdata pipelines?\b", r"\bpipelines?\b", r"\bbatch processing\b"],
    "rag": [r"\brag\b", r"\bretrieval[- ]augmented generation\b"],
    "llm": [r"\bllm\b", r"\blarge language model(s)?\b"],
    "communication": [r"\bcommunication\b", r"\bcommunicate\b"],
    "problem solving": [r"\bproblem solving\b", r"\bproblem-solving\b"],
    "pivot tables": [r"\bpivot table(s)?\b"],
    "vlookup": [r"\bvlookup\b"],
    "power query": [r"\bpower query\b"],
    "dax": [r"\bdax\b"],
    "google sheets": [r"\bgoogle sheets\b"],
    "looker": [r"\blooker\b"],
    "sql server": [r"\bsql server\b"],
    "postgresql": [r"\bpostgresql\b"],
    "bigquery": [r"\bbigquery\b"],
    "snowflake": [r"\bsnowflake\b"],
    "redshift": [r"\bredshift\b"],
    "etl": [r"\betl\b"],
    "data wrangling": [r"\bdata wrangling\b"],
    "data modeling": [r"\bdata modeling\b"],
    "kpi": [r"\bkpi(s)?\b"],
    "okr": [r"\bokr(s)?\b"],
    "stakeholder communication": [r"\bstakeholder communication\b"],
    "data-driven decision making": [r"\bdata[- ]driven decision making\b"]
}

def extract_skills_v2(text: str) -> Set[str]:
    text = normalize_for_skills(text)
    found = set()
    for skill, patterns in SKILL_ALIASES.items():
        for pat in patterns:
            if re.search(pat, text):
                found.add(skill)
                break
    return found


# ----------------------------
# Explainability via TF-IDF top terms
# ----------------------------
# Use TF-IDF to identify important overlapping terms
# Helps explain why the resume matches the job description
def top_tfidf_terms(resume_text: str, job_text: str, top_k: int = 15) -> List[Tuple[str, float]]:
    """
    Fit TF-IDF on [resume + job], then return top terms from job vector
    that also appear in resume vocabulary, as a quick explainability signal.
    """
    resume_text = clean_text_basic(resume_text)
    job_text = clean_text_basic(job_text)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform([resume_text, job_text])

    feature_names = vectorizer.get_feature_names_out()
    job_vec = X[1].toarray().ravel()

    # Top terms by tf-idf weight in job description
    top_idx = job_vec.argsort()[::-1][: top_k * 3]  # grab more, we will filter
    terms = []
    for idx in top_idx:
        term = feature_names[idx]
        score = float(job_vec[idx])
        if score <= 0:
            continue
        # keep terms that are likely present in resume too
        if term in resume_text:
            terms.append((term, score))
        if len(terms) >= top_k:
            break
    return terms

# ----------------------------
# Scoring
# ----------------------------
# Split job description into required and preferred sections
# This allows weighted scoring based on importance of skills
def split_job_description(job_text: str) -> dict:
    text = clean_text_basic(job_text)

    required_keywords = ["requirements", "required", "must have", "qualifications"]
    preferred_keywords = ["preferred", "nice to have", "plus", "bonus"]

    required_text = text
    preferred_text = ""

    for keyword in preferred_keywords:
        if keyword in text:
            parts = text.split(keyword, 1)
            required_text = parts[0]
            preferred_text = parts[1]
            break

    return {
        "required": required_text,
        "preferred": preferred_text if preferred_text else text
    }


def split_resume_sections(resume_text: str) -> dict:
    text = clean_text_basic(resume_text)

    sections = {
        "skills": "",
        "experience": "",
        "projects": "",
        "education": "",
        "full": text
    }

    headings = {
        "skills": ["skills", "technical skills"],
        "experience": ["experience", "work experience", "professional experience"],
        "projects": ["projects", "academic projects"],
        "education": ["education"]
    }

    for section, possible_headings in headings.items():
        for heading in possible_headings:
            if heading in text:
                start = text.find(heading)
                sections[section] = text[start:start + 1200]
                break

    return sections

# Compute semantic similarity between a resume section and the job description
# Used as a heuristic signal for experience/profile fit
def compute_section_similarity(model, section_text: str, job_text: str) -> float:
    if not section_text.strip() or not job_text.strip():
        return 0.0

    section_emb = model.encode([section_text])
    job_emb = model.encode([job_text])

    return float(cosine_similarity(section_emb, job_emb)[0][0])


def generate_recruiter_recommendations(
    missing_required,
    missing_preferred,
    required_score,
    preferred_score,
    experience_score
):
    recommendations = []

    if missing_required:
        recommendations.append(
            "Prioritize adding evidence for required skills: " + ", ".join(missing_required)
        )

    if missing_preferred:
        recommendations.append(
            "Consider adding preferred skills if applicable: " + ", ".join(missing_preferred)
        )

    if required_score < 0.6:
        recommendations.append(
            "Required skill match is low. Tailor the resume more closely to the core job requirements."
        )

    if preferred_score < 0.5:
        recommendations.append(
            "Preferred qualification match can be improved by adding relevant tools, projects, or domain experience."
        )

    if experience_score < 0.45:
        recommendations.append(
            "Experience/profile alignment appears weak. Add clearer project or work experience descriptions related to the role."
        )

    if not recommendations:
        recommendations.append(
            "The resume shows strong alignment with the job description. Minor tailoring may further improve the match."
        )

    return recommendations

# ----------------------------
# Model caching
# ----------------------------
@dataclass
class AnalyzerConfig:
    model_name: str = "all-MiniLM-L6-v2"
    w_similarity: float = 0.7
    w_skills: float = 0.3

class ResumeAnalyzer:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.model = SentenceTransformer(config.model_name)

    def analyze_texts(self, resume_text: str, job_text: str) -> Dict:
        resume_clean = clean_text_basic(resume_text)
        job_clean = clean_text_basic(job_text)

        # ----------------------------
        # Structured parsing
        # ----------------------------
        job_parts = split_job_description(job_clean)
        resume_sections = split_resume_sections(resume_clean)

        required_text = job_parts["required"]
        preferred_text = job_parts["preferred"]

        # ----------------------------
        # Global semantic similarity
        # ----------------------------
        resume_emb = self.model.encode([resume_clean])
        job_emb = self.model.encode([job_clean])
        semantic_score = float(cosine_similarity(resume_emb, job_emb)[0][0])

        # ----------------------------
        # Required vs preferred skill matching
        # ----------------------------
        # Compute overlap between resume skills and job description skills
        # Separate scores for required and preferred skills
        resume_skills = extract_skills_v2(resume_clean)

        required_skills = extract_skills_v2(required_text)
        preferred_skills = extract_skills_v2(preferred_text)

        matched_required = sorted(resume_skills.intersection(required_skills))
        missing_required = sorted(required_skills - resume_skills)

        matched_preferred = sorted(resume_skills.intersection(preferred_skills))
        missing_preferred = sorted(preferred_skills - resume_skills)

        required_score = (
            len(matched_required) / len(required_skills)
            if required_skills else 0.0
        )

        preferred_score = (
            len(matched_preferred) / len(preferred_skills)
            if preferred_skills else 0.0
        )

        # ----------------------------
        # Experience / profile-fit score
        # ----------------------------
        # Estimate candidate experience relevance based on keyword presence
        # This is a heuristic approximation of profile fit
        experience_text = (
            resume_sections["experience"]
            or resume_sections["projects"]
            or resume_sections["full"]
        )

        experience_score = compute_section_similarity(
            self.model,
            experience_text,
            job_clean
        )

        # ----------------------------
        # Recruiter-aware final score
        # ----------------------------
        final = (
            0.40 * semantic_score
            + 0.30 * required_score
            + 0.15 * preferred_score
            + 0.15 * experience_score
        )

        # ----------------------------
        # Backward-compatible skill summary
        # ----------------------------
        all_job_skills = required_skills.union(preferred_skills)
        matched_all = sorted(resume_skills.intersection(all_job_skills))
        missing_all = sorted(all_job_skills - resume_skills)

        skill_score = (
            len(matched_all) / len(all_job_skills)
            if all_job_skills else 0.0
        )

        # ----------------------------
        # Explainability
        # ----------------------------
        top_terms = top_tfidf_terms(resume_text, job_text, top_k=15)

        # ----------------------------
        # Recommendations
        # ----------------------------
        recommendations = generate_recruiter_recommendations(
            missing_required,
            missing_preferred,
            required_score,
            preferred_score,
            experience_score
        )

        return {
            "semantic_similarity": round(semantic_score, 4),
            "required_skill_match": round(required_score, 4),
            "preferred_skill_match": round(preferred_score, 4),
            "experience_profile_fit": round(experience_score, 4),
            "skill_overlap": round(skill_score, 4),
            "final_score": round(final, 4),

            "matched_required_skills": matched_required,
            "missing_required_skills": missing_required,
            "matched_preferred_skills": matched_preferred,
            "missing_preferred_skills": missing_preferred,

            "matched_skills": matched_all,
            "missing_skills": missing_all,
            "matched_count": len(matched_all),
            "missing_count": len(missing_all),

            "resume_skills_found": sorted(resume_skills),
            "required_skills_found": sorted(required_skills),
            "preferred_skills_found": sorted(preferred_skills),
            "job_skills_found": sorted(all_job_skills),

            "top_matching_terms": [(t, round(s, 4)) for t, s in top_terms],
            "recommendations": recommendations,

            "scoring_breakdown": {
                "semantic_similarity_weight": 0.40,
                "required_skill_weight": 0.30,
                "preferred_skill_weight": 0.15,
                "experience_profile_fit_weight": 0.15
            },

            "model_name": self.config.model_name
        }
    def analyze_pdf(self, pdf_path: str, job_text: str) -> Dict:
        resume_text = extract_text_from_pdf(pdf_path)
        return self.analyze_texts(resume_text, job_text)