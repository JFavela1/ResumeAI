# ResumeAI — AgentMatch

AgentMatch is an AI-powered resume and job-description matching system. It uses semantic similarity and agentic AI to rank candidate-job fit, identify skill gaps, and generate actionable career recommendations.

---

## Project Structure

```
ResumeAI/
├── src/
│   ├── config.py            # Central config: paths, model names, hyperparameters
│   ├── data_loader.py       # Downloads raw dataset from Hugging Face
│   ├── preprocess.py        # Text cleaning and DataFrame builder
│   ├── baseline_tfidf.py    # TF-IDF cosine similarity baseline
│   ├── baseline_bm25.py     # BM25 similarity baseline
│   ├── semantic_matcher.py  # SBERT semantic similarity
│   ├── skill_gap.py         # Skill gap analysis and recommendations
│   ├── evaluate.py          # Model comparison and plots
│   └── agent.py             # Agentic AI orchestrator (in progress)
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
Generates a correlation comparison table and scatter/bar plots.
```bash
cd src
python evaluate.py
```

---

## Dataset

[netsol/resume-score-details](https://huggingface.co/datasets/netsol/resume-score-details) — resume/job-description pairs with micro and macro match scores.

---

## Models

| Model | Description |
|-------|-------------|
| TF-IDF | Bag-of-words cosine similarity |
| BM25 | Probabilistic keyword retrieval |
| SBERT | Sentence-level semantic embeddings (`all-MiniLM-L6-v2`) |
