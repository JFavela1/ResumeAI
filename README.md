# ResumeAI — AgentMatch

AgentMatch is an AI-powered resume and job-description matching system. It uses semantic similarity and agentic AI to rank candidate-job fit, identify skill gaps, and generate actionable career recommendations.

---

## Project Structure

```
ResumeAI/
├── src/
│   ├── config.py            # Central config: paths, model names, hyperparameters
│   ├── data_loader.py       # Downloads raw dataset from Hugging Face
│   ├── preprocess.py        # Text cleaning (clean_text + normalize_text)
│   ├── baseline_tfidf.py    # TF-IDF cosine similarity baseline
│   ├── baseline_bm25.py     # BM25 keyword relevance baseline
│   ├── semantic_matcher.py  # SBERT semantic similarity
│   ├── skill_gap.py         # Skill gap analysis and recommendations
│   ├── evaluate.py          # Model comparison and plots
│   └── agent.py             # AgentMatch — LLM-powered orchestrator
├── data/
│   ├── build_dataset.py     # One-time pipeline: download → clean → save
│   └── processed/           # Generated CSVs and plots (git-ignored)
├── notebooks/               # Exploratory notebooks
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv resume
source resume/bin/activate      # Windows: resume\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key (required for the agent)
export OPENAI_API_KEY=sk-...
```

---

## Running the Pipeline

### Step 1 — Build the dataset (run once)
Downloads from Hugging Face, cleans, and saves to `data/processed/`.
```bash
python data/build_dataset.py
```

### Step 2 — Run baselines
Each script reads from `data/processed/clean_resume_job_pairs.csv` and writes its own results CSV.
```bash
cd src
python baseline_tfidf.py
python baseline_bm25.py
python semantic_matcher.py
```

### Step 3 — Analyze skill gaps
```bash
cd src
python skill_gap.py
```

### Step 4 — Evaluate and compare models
Generates a correlation comparison table and scatter/bar plots in `data/processed/`.
```bash
cd src
python evaluate.py
```

---

## AgentMatch — LLM Agent

The agent orchestrates all three matchers and the skill gap analyzer through OpenAI's function-calling API, then synthesizes a structured recommendation.

### Quick start

```python
from src.agent import analyze

resume = """
Jane Doe | Python Developer
Skills: Python, Django, PostgreSQL, scikit-learn, Docker
Experience: 3 years backend development, REST APIs, agile teams
"""

job_description = """
Senior Python Engineer
Requirements: 4+ years Python, Django or FastAPI, PostgreSQL,
machine learning experience, Docker, strong communication skills.
"""

result = analyze(resume, job_description)
print(result)
```

### Output format

```json
{
  "scores": {
    "tfidf_similarity": 0.61,
    "bm25_score": 3.84,
    "sbert_similarity": 0.79,
    "overall_fit_pct": 72
  },
  "skill_analysis": {
    "matched_skills": ["python", "django", "postgresql", "docker"],
    "missing_skills": ["machine learning", "fastapi"]
  },
  "fit_level": "Good",
  "recommendation": "Jane's profile is a strong keyword and semantic match for this role..."
}
```

### Fit levels

| Fit Level | overall_fit_pct |
|-----------|----------------|
| Strong | ≥ 70% |
| Good | 50–69% |
| Fair | 30–49% |
| Poor | < 30% |

### Running the built-in demo
```bash
cd src && python agent.py
```
Add `verbose=True` to `analyze()` to print each tool call as the agent works through it.

### Estimated API cost

The default orchestrator model is `gpt-4o-mini` in `src/config.py`. Pricing varies by model and token usage; see [OpenAI pricing](https://openai.com/pricing). Typical single analyses are on the order of a few cents with `gpt-4o-mini`.

To change model or token budget, edit `AGENT_MODEL` and `AGENT_MAX_TOKENS` in `src/config.py`.

---

## Dataset

[netsol/resume-score-details](https://huggingface.co/datasets/netsol/resume-score-details) — resume/job-description pairs with micro and macro match scores.

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| TF-IDF | Baseline | Bag-of-words cosine similarity |
| BM25 | Baseline | Probabilistic keyword retrieval |
| SBERT | Semantic | Sentence-level embeddings (`all-MiniLM-L6-v2`) |
| AgentMatch | Agent | OpenAI chat model with function calling |
