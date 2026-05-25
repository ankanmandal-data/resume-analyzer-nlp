import re
import fitz
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def clean_text_basic(text: str) -> str:
    """Lowercase, collapse whitespace, strip newlines."""
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_skills(text: str) -> str:
    """Normalize text for skill matching — splits on common delimiters."""
    text = text.lower().replace("\n", " ")
    text = re.sub(r"[/,_|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Experience heuristic ───────────────────────────────────────────────────────

def extract_years_of_experience(text: str) -> Optional[int]:
    """
    Best-effort extraction of years of experience from resume text.
    Looks for patterns like '3 years', '5+ years experience', etc.
    Returns the highest number found, or None if nothing is detected.
    This is a rough heuristic — it doesn't distinguish between
    total career length and individual role durations.
    """
    text = text.lower()
    patterns = [
        r"(\d+)\+?\s*years?\s+of\s+experience",
        r"(\d+)\+?\s*years?\s+experience",
        r"(\d+)\+?\s*yrs?\s+experience",
        r"(\d+)\+?\s*years?\s+of\s+work",
    ]
    found = []
    for pat in patterns:
        for match in re.finditer(pat, text):
            val = int(match.group(1))
            if 0 < val < 40:   # sanity-check range
                found.append(val)
    return max(found) if found else None


# ── Skill dictionary ───────────────────────────────────────────────────────────
# Each key is the canonical skill name; values are regex patterns that match it.

SKILL_ALIASES = {
    "python":           [r"\bpython\b", r"\bpython3\b"],
    "sql":              [r"\bsql\b"],
    "machine learning": [r"\bmachine learning\b", r"\bml\b"],
    "deep learning":    [r"\bdeep learning\b", r"\bdl\b", r"\bneural network(s)?\b"],
    "nlp":              [r"\bnlp\b", r"\bnatural language processing\b"],
    "statistics":       [r"\bstatistics\b", r"\bstatistical\b", r"\bstat(s)?\b"],
    "data analysis":    [r"\bdata analysis\b", r"\bdata analyst\b", r"\banalytics\b"],
    "classification":   [r"\bclassification\b", r"\bclassifier\b"],
    "regression":       [r"\bregression\b"],
    "model evaluation": [r"\bmodel evaluation\b", r"\bevaluation\b", r"\bmetrics\b"],
    "scikit-learn":     [r"\bscikit[-_ ]?learn\b", r"\bsklearn\b"],
    "pandas":           [r"\bpandas\b"],
    "numpy":            [r"\bnumpy\b"],
    "tensorflow":       [r"\btensorflow\b", r"\bkeras\b"],
    "pytorch":          [r"\bpytorch\b", r"\btorch\b"],
    "matplotlib":       [r"\bmatplotlib\b"],
    "seaborn":          [r"\bseaborn\b"],
    "mysql":            [r"\bmysql\b"],
    "git":              [r"\bgit\b", r"\bgithub\b"],
    "jupyter":          [r"\bjupyter\b", r"\bjupyter notebook\b"],
    "excel":            [r"\bexcel\b", r"\bmicrosoft excel\b"],
    "tableau":          [r"\btableau\b"],
    "power bi":         [r"\bpower\s*bi\b"],
    "a/b testing":      [r"\ba/b testing\b", r"\bab testing\b", r"\ba b testing\b"],
    "dashboarding":     [r"\bdashboarding\b", r"\bdashboard(s)?\b"],
    "data visualization": [
        r"\bdata visualization\b", r"\bvisualization\b", r"\bvisualisation\b",
        r"\bmatplotlib\b", r"\bseaborn\b", r"\btableau\b", r"\bpower\s*bi\b",
    ],
    "data storytelling":   [r"\bdata storytelling\b", r"\bstorytelling\b"],
    "business analytics":  [r"\bbusiness analytics\b", r"\bbusiness analysis\b"],
    "predictive modeling": [r"\bpredictive modeling\b", r"\bpredictive modelling\b", r"\bpredictive models?\b"],
    "data cleaning":       [r"\bdata cleaning\b", r"\bcleaning\b"],
    "reporting":           [r"\breporting\b", r"\breports?\b"],
    "api":                 [r"\bapi\b", r"\bapis\b"],
    "cloud":               [r"\bcloud\b", r"\bcloud-based\b", r"\bcloud workflows?\b"],
    "spark":               [r"\bspark\b", r"\bpyspark\b"],
    "aws":                 [r"\baws\b", r"\bamazon web services\b"],
    "s3":                  [r"\bs3\b", r"\bamazon s3\b"],
    "ec2":                 [r"\bec2\b"],
    "cloudfront":          [r"\bcloudfront\b"],
    "model deployment":    [r"\bmodel deployment\b", r"\bdeployment\b", r"\bdeploy\b"],
    "model monitoring":    [r"\bmodel monitoring\b", r"\bmonitoring\b"],
    "mlops":               [r"\bmlops\b", r"\bmachine learning operations\b"],
    "data pipelines":      [r"\bdata pipelines?\b", r"\bpipelines?\b", r"\bbatch processing\b"],
    "rag":                 [r"\brag\b", r"\bretrieval[- ]augmented generation\b"],
    "llm":                 [r"\bllm\b", r"\blarge language model(s)?\b"],
    "communication":       [r"\bcommunication\b", r"\bcommunicate\b"],
    "problem solving":     [r"\bproblem solving\b", r"\bproblem-solving\b"],
    # Data Analyst / BI focused
    "pivot tables":        [r"\bpivot table(s)?\b"],
    "vlookup":             [r"\bvlookup\b"],
    "power query":         [r"\bpower query\b"],
    "dax":                 [r"\bdax\b"],
    "google sheets":       [r"\bgoogle sheets\b"],
    "looker":              [r"\blooker\b"],
    "sql server":          [r"\bsql server\b"],
    "postgresql":          [r"\bpostgresql\b"],
    "bigquery":            [r"\bbigquery\b"],
    "snowflake":           [r"\bsnowflake\b"],
    "redshift":            [r"\bredshift\b"],
    "etl":                 [r"\betl\b"],
    "data wrangling":      [r"\bdata wrangling\b"],
    "data modeling":       [r"\bdata modeling\b"],
    "kpi":                 [r"\bkpi(s)?\b"],
    "okr":                 [r"\bokr(s)?\b"],
    "stakeholder communication":   [r"\bstakeholder communication\b"],
    "data-driven decision making": [r"\bdata[- ]driven decision making\b"],
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


# ── Score label ────────────────────────────────────────────────────────────────

def get_match_label(score: float) -> str:
    """
    Maps a final score to a human-readable label.
    Thresholds are heuristic and may need adjustment depending on use case.
    """
    if score >= 0.75:
        return "Strong Match"
    elif score >= 0.45:
        return "Moderate Match"
    return "Weak Match"


# ── TF-IDF baseline ────────────────────────────────────────────────────────────

def compute_tfidf_similarity(resume_text: str, job_text: str) -> float:
    """
    Keyword-based similarity using TF-IDF cosine distance.
    Used as a baseline to compare against the semantic embedding score.
    A higher semantic score relative to this baseline suggests the
    embedding model is picking up on meaning beyond exact keyword overlap.
    """
    resume_text = clean_text_basic(resume_text)
    job_text = clean_text_basic(job_text)
    if not resume_text.strip() or not job_text.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform([resume_text, job_text])
    return float(cosine_similarity(X[0], X[1])[0][0])


# ── Explainability ─────────────────────────────────────────────────────────────

def top_tfidf_terms(resume_text: str, job_text: str, top_k: int = 15) -> List[Tuple[str, float]]:
    """
    Returns overlapping terms between resume and JD ranked by TF-IDF weight
    in the job description. Useful for explaining which keywords contributed
    most to the match — though this reflects term importance, not causation.
    """
    resume_text = clean_text_basic(resume_text)
    job_text = clean_text_basic(job_text)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform([resume_text, job_text])

    feature_names = vectorizer.get_feature_names_out()
    job_vec = X[1].toarray().ravel()

    top_idx = job_vec.argsort()[::-1][: top_k * 3]
    terms = []
    for idx in top_idx:
        term = feature_names[idx]
        score = float(job_vec[idx])
        if score <= 0:
            break
        if term in resume_text:
            terms.append((term, score))
        if len(terms) >= top_k:
            break
    return terms


# ── JD / Resume parsing ────────────────────────────────────────────────────────

def split_job_description(job_text: str) -> dict:
    """
    Splits JD into required and preferred sections using keyword detection.
    Falls back to treating the full text as required when no split is found.
    Works reasonably well for structured JDs; may not split correctly for
    non-standard formats.
    """
    text = clean_text_basic(job_text)
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
        "preferred": preferred_text if preferred_text else text,
    }


def split_resume_sections(resume_text: str) -> dict:
    """
    Extracts named sections from the resume using heading keyword detection.
    Takes up to 1200 characters from each section start. Works for most
    standard single-column resume formats; multi-column layouts may not
    parse correctly depending on how the PDF text is ordered.
    """
    text = clean_text_basic(resume_text)
    sections = {
        "skills": "", "experience": "", "projects": "", "education": "", "full": text
    }
    headings = {
        "skills":     ["skills", "technical skills"],
        "experience": ["experience", "work experience", "professional experience"],
        "projects":   ["projects", "academic projects"],
        "education":  ["education"],
    }
    for section, possible_headings in headings.items():
        for heading in possible_headings:
            if heading in text:
                start = text.find(heading)
                sections[section] = text[start: start + 1200]
                break
    return sections


def compute_section_similarity(model, section_text: str, job_text: str) -> float:
    """Cosine similarity between a resume section embedding and the full JD embedding."""
    if not section_text.strip() or not job_text.strip():
        return 0.0
    section_emb = model.encode([section_text])
    job_emb = model.encode([job_text])
    return float(cosine_similarity(section_emb, job_emb)[0][0])


# ── Recommendations ────────────────────────────────────────────────────────────

def generate_recruiter_recommendations(
    missing_required, missing_preferred,
    required_score, preferred_score, experience_score
) -> List[str]:
    recommendations = []

    if missing_required:
        recommendations.append(
            "Consider adding evidence for these required skills if applicable: "
            + ", ".join(missing_required)
        )
    if missing_preferred:
        recommendations.append(
            "These preferred skills are not currently detected — worth adding if you have relevant experience: "
            + ", ".join(missing_preferred)
        )
    if required_score < 0.6:
        recommendations.append(
            "Required skill coverage appears limited. Tailoring the resume more closely "
            "to the job's core requirements may improve the match."
        )
    if preferred_score < 0.5:
        recommendations.append(
            "Preferred qualification coverage could be improved by highlighting relevant "
            "tools, projects, or domain experience where applicable."
        )
    if experience_score < 0.45:
        recommendations.append(
            "The experience section may not align strongly with the role. "
            "Adding more specific descriptions of relevant projects or responsibilities could help."
        )
    if not recommendations:
        recommendations.append(
            "The resume appears to align well with this job description. "
            "Minor tailoring around specific terminology may further improve the match."
        )
    return recommendations


# ── Config & analyzer ─────────────────────────────────────────────────────────

@dataclass
class AnalyzerConfig:
    model_name: str = "all-MiniLM-L6-v2"


class ResumeAnalyzer:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.model = SentenceTransformer(config.model_name)

    def analyze_texts(self, resume_text: str, job_text: str) -> Dict:
        resume_clean = clean_text_basic(resume_text)
        job_clean = clean_text_basic(job_text)

        job_parts = split_job_description(job_clean)
        resume_sections = split_resume_sections(resume_clean)

        required_text = job_parts["required"]
        preferred_text = job_parts["preferred"]

        # Semantic similarity (primary signal)
        resume_emb = self.model.encode([resume_clean])
        job_emb = self.model.encode([job_clean])
        semantic_score = float(cosine_similarity(resume_emb, job_emb)[0][0])

        # TF-IDF baseline for comparison
        tfidf_baseline_score = compute_tfidf_similarity(resume_clean, job_clean)
        semantic_lift = round(semantic_score - tfidf_baseline_score, 4)

        # Skill matching
        resume_skills = extract_skills_v2(resume_clean)
        required_skills = extract_skills_v2(required_text)
        preferred_skills = extract_skills_v2(preferred_text)

        matched_required = sorted(resume_skills.intersection(required_skills))
        missing_required = sorted(required_skills - resume_skills)
        matched_preferred = sorted(resume_skills.intersection(preferred_skills))
        missing_preferred = sorted(preferred_skills - resume_skills)

        required_score = len(matched_required) / len(required_skills) if required_skills else 0.0
        preferred_score = len(matched_preferred) / len(preferred_skills) if preferred_skills else 0.0

        # Experience/profile fit
        experience_text = (
            resume_sections["experience"]
            or resume_sections["projects"]
            or resume_sections["full"]
        )
        experience_score = compute_section_similarity(self.model, experience_text, job_clean)

        # Years of experience (best-effort heuristic)
        years_detected = extract_years_of_experience(resume_text)

        # Weighted final score
        final = (
            0.40 * semantic_score
            + 0.30 * required_score
            + 0.15 * preferred_score
            + 0.15 * experience_score
        )

        # Combined skill summary
        all_job_skills = required_skills.union(preferred_skills)
        matched_all = sorted(resume_skills.intersection(all_job_skills))
        missing_all = sorted(all_job_skills - resume_skills)
        skill_score = len(matched_all) / len(all_job_skills) if all_job_skills else 0.0

        top_terms = top_tfidf_terms(resume_text, job_text, top_k=15)
        recommendations = generate_recruiter_recommendations(
            missing_required, missing_preferred,
            required_score, preferred_score, experience_score
        )
        match_label = get_match_label(final)

        return {
            "match_label":                match_label,
            "final_score":                round(final, 4),
            "semantic_similarity":        round(semantic_score, 4),
            "tfidf_baseline_similarity":  round(tfidf_baseline_score, 4),
            "semantic_lift_over_tfidf":   semantic_lift,
            "required_skill_match":       round(required_score, 4),
            "preferred_skill_match":      round(preferred_score, 4),
            "experience_profile_fit":     round(experience_score, 4),
            "skill_overlap":              round(skill_score, 4),

            "matched_required_skills":  matched_required,
            "missing_required_skills":  missing_required,
            "matched_preferred_skills": matched_preferred,
            "missing_preferred_skills": missing_preferred,
            "matched_skills":           matched_all,
            "missing_skills":           missing_all,
            "matched_count":            len(matched_all),
            "missing_count":            len(missing_all),

            "years_of_experience_detected": years_detected,

            "resume_skills_found":    sorted(resume_skills),
            "required_skills_found":  sorted(required_skills),
            "preferred_skills_found": sorted(preferred_skills),
            "job_skills_found":       sorted(all_job_skills),

            "top_matching_terms": [(t, round(s, 4)) for t, s in top_terms],
            "recommendations":    recommendations,

            "scoring_breakdown": {
                "semantic_similarity_weight":    0.40,
                "required_skill_weight":         0.30,
                "preferred_skill_weight":        0.15,
                "experience_profile_fit_weight": 0.15,
            },
            "model_name": self.config.model_name,
        }

    def analyze_pdf(self, pdf_path: str, job_text: str) -> Dict:
        resume_text = extract_text_from_pdf(pdf_path)
        return self.analyze_texts(resume_text, job_text)
