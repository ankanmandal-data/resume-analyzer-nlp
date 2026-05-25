# Resume Analyzer

A structured NLP pipeline that scores how well a resume matches a job description. It combines sentence embeddings, keyword-based skill matching, and section-level experience alignment into a single weighted score, with a TF-IDF baseline for comparison and term-level explainability.

Built with Sentence Transformers, scikit-learn, and Streamlit.

## Features

- PDF resume upload with automatic text extraction
- Single and batch job description comparison (`===JOB===` separator)
- Semantic similarity via pretrained Sentence Transformers (`all-MiniLM-L6-v2`)
- TF-IDF keyword baseline with semantic lift metric
- Separate required vs preferred skill matching (65+ skills across DS, DA, BI, DE roles)
- Section-level experience/profile fit scoring
- Best-effort years of experience detection from resume text
- Weighted final score with recruiter-style label: 🟢 Strong / 🟡 Moderate / 🔴 Weak
- Score breakdown expander showing per-component contributions
- TF-IDF term explainability with contextual commentary
- Actionable improvement suggestions
- Downloadable JSON report

## App Screenshot

![App Output](assets/app_output.png)

## Tech Stack

- Python 3.9+
- Streamlit
- Sentence Transformers
- scikit-learn
- PyMuPDF (fitz)
- NumPy / Pandas

## Project Structure

```
resume-analyzer/
├── analyzer.py                       # Core NLP pipeline
├── app.py                            # Streamlit UI
├── evaluation.py                     # Evaluation script
├── evaluation_results.csv            # Output from evaluation run
├── evaluation_confusion_matrix.csv   # Confusion matrix output
├── requirements.txt
├── README.md
├── assets/
│   ├── System_Architecture.png
│   ├── Scoring_System.png
│   ├── Explainability_Flow.png
│   └── app_output.png
├── sample_data/
│   ├── sample_resume.pdf
│   └── sample_job_description.txt
└── notebooks/
    └── prototype.ipynb               # Early exploration (reference only)
```

## How It Works

1. Extract raw text from the resume PDF using PyMuPDF
2. Normalize and clean both resume and job description text
3. Split the JD into required and preferred sections using keyword detection
4. Extract skills from both documents using an alias-based regex dictionary
5. Compute global semantic similarity using Sentence Transformer embeddings
6. Compute a TF-IDF keyword baseline and the semantic lift over it
7. Score required and preferred skill match separately
8. Estimate experience/profile fit using section-level cosine similarity
9. Attempt to extract years of experience using regex patterns
10. Combine all components into a weighted final score
11. Surface matched/missing skills, top TF-IDF terms, and suggestions

## System Architecture

![System Architecture](assets/System_Architecture.png)

## Scoring

Final score is a weighted combination of four components:

| Component | Weight |
|---|---|
| Semantic Similarity | 40% |
| Required Skill Match | 30% |
| Preferred Skill Match | 15% |
| Experience / Profile Fit | 15% |

```
Final Score = 0.40 × Semantic + 0.30 × Required + 0.15 × Preferred + 0.15 × Experience
```

All scores are normalized between 0 and 1. The weights are reasonable defaults based on
what matters most in typical screening, but they can be adjusted for different use cases.

![Scoring System](assets/Scoring_System.png)

## Explainability

Several signals help interpret the score:

- **TF-IDF baseline** — keyword overlap between resume and JD
- **Semantic similarity** — contextual similarity via Sentence Transformers
- **Semantic lift** — difference between the two; a positive lift suggests the
  embedding model is detecting relevant context that keyword matching alone would miss
- **Top matching terms** — the specific n-grams with the highest TF-IDF weight in both documents

This is intended to make the scoring more transparent and interpretable, not to imply
causal attribution — the final score is a heuristic, not a ground-truth ranking.

![Explainability Flow](assets/Explainability_Flow.png)

## Batch Mode

Separate multiple job descriptions with `===JOB===` to compare one resume against
several openings at once. The app ranks them by score and shows a side-by-side table.

## Evaluation

`evaluation.py` runs 12 labeled test cases and outputs accuracy, a classification report,
and a confusion matrix. Cases cover Data Science, Data Analyst, BI Analyst, Data Engineering,
and clearly mismatched roles (marketing, frontend, HR).

```bash
python evaluation.py
```

**Results on the 12-case test set:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Low | 1.00 | 1.00 | 1.00 |
| Medium | 1.00 | 1.00 | 1.00 |
| High | 1.00 | 1.00 | 1.00 |

Overall accuracy: **12/12 on this test set**

These results should be interpreted with caution — the test set is small (12 cases)
and uses simplified skill-list resumes rather than real PDFs. Performance on real-world
resumes will likely show more variance. The evaluation is primarily useful for
verifying that the scoring logic behaves consistently across match levels.

Thresholds: High ≥ 0.75 · Medium ≥ 0.45 · Low < 0.45

## How to Run

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sample files in `sample_data/` can be used to test the app immediately without
needing to provide your own resume.

To regenerate evaluation output:

```bash
python evaluation.py
```

## Limitations

- **Small evaluation set:** 12 test cases using simplified text, not real resumes.
  Results on real PDFs will vary, especially for complex or non-standard layouts.
- **PDF parsing:** Multi-column or scanned resumes may not extract cleanly.
- **Skill dictionary:** Manually maintained — new or niche tools need to be added manually.
- **JD section splitting:** Relies on keyword detection; non-standard JD formats may
  not split correctly into required/preferred sections.
- **Experience fit:** Section-level semantic similarity is a rough proxy —
  it doesn't model actual years of experience or role progression.
- **Years detection:** Regex-based heuristic; may over- or under-count in some cases.

## Future Work

- Reverse mode: rank multiple resumes against a single job description
- Structured experience parsing (years per role, company names, progression)
- LLM-based natural language explanation layer
- Domain-specific skill vocabulary extensions (Finance, Healthcare, etc.)
- Larger and more diverse evaluation set with real anonymized resumes

## Author

Ankan Mandal
