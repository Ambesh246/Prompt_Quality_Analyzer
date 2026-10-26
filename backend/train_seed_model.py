"""Train a transparent bootstrap classifier from synthetic contrastive examples.

This is a development starter artifact only. Train the Colab notebook on BPO before
making claims about general prompt-quality prediction or deploying for real users.
"""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from app.ml.evaluation import classification_metrics
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    ("explain", "photosynthesis", "a first-year biology student", "key stages and why light matters", "three concise sections and one everyday example"),
    ("summarize", "the following article", "a busy reader", "the thesis, strongest evidence, and caveats", "five bullet points and a one-sentence takeaway"),
    ("write", "a professional follow-up email", "a recruiter", "thank them for the interview and ask about next steps", "a subject line and a polite 120-word email"),
    ("create", "a Python function to remove duplicate values", "a beginner programmer", "preserve the original order and explain complexity", "code, two tests, and a short explanation"),
    ("analyze", "monthly sales data", "a small business owner", "trends, outliers, and month-over-month changes", "a summary table and three actionable findings"),
    ("compare", "PostgreSQL and MySQL", "a student choosing a database", "performance, features, ease of use, and typical use cases", "a comparison table and a conditional recommendation"),
    ("draft", "a project proposal for a campus app", "a faculty review panel", "problem, target users, scope, risks, and success metrics", "clear headings and an implementation timeline"),
    ("debug", "a Python function that returns the wrong result", "a student", "the smallest reproducible cause and a minimal fix", "explanation, corrected code, and a regression test"),
    ("plan", "a two-week exam revision schedule", "an engineering student", "daily study blocks, revision, and practice questions", "a day-by-day table with realistic breaks"),
    ("translate", "a short customer support message", "a Hindi-speaking customer", "preserve the meaning and respectful tone", "natural Hindi with no added information"),
    ("design", "a landing page for a student portfolio", "a recruiter viewing it on mobile", "project evidence, skills, contact details, and readability", "a component outline with responsive layout notes"),
    ("research", "ways to reduce household energy use", "a renter with a limited budget", "low-cost options and likely trade-offs", "prioritized recommendations with evidence caveats"),
    ("evaluate", "a proposed machine-learning project", "a student team", "data availability, leakage, baseline, metrics, and deployment risk", "a rubric table and top three fixes"),
    ("write", "a short story about a lost robot", "young readers", "curiosity, friendship, and a hopeful ending", "about 500 words in a warm tone"),
    ("extract", "entities from a short paragraph", "a data analyst", "people, organizations, dates, and locations only when present", "valid JSON with a documented schema"),
    ("build", "a REST API for task management", "a small project team", "CRUD operations, validation, errors, and tests", "endpoint table and example requests"),
    ("explain", "gradient descent", "a student new to machine learning", "the intuition, update step, and a common pitfall", "plain language, a small numeric example, and key equations"),
    ("compare", "two internship offers", "a student deciding between them", "learning, role fit, commute, compensation, and growth", "a weighted decision matrix; ask for missing details"),
    ("create", "SQL queries for a library database", "a beginner", "joins, aggregate functions, and edge cases", "five queries with sample schema and expected results"),
    ("summarize", "a meeting transcript", "the project lead", "decisions, owners, deadlines, and unresolved questions", "sections for decisions, action items, and open issues"),
    ("write", "a LinkedIn project description", "a technical recruiter", "problem, contribution, stack, evidence, and measurable impact", "three concise bullets without invented results"),
    ("analyze", "a classification model's results", "a machine-learning student", "precision, recall, F1, class imbalance, and error patterns", "a metric table plus limitations and next experiments"),
    ("create", "a weekly meal plan", "a person with a limited budget", "balanced options and repeatable ingredients", "a seven-day table and consolidated shopping list"),
    ("draft", "a polite message requesting an extension", "a university instructor", "the reason at an appropriate level of detail and a specific new date", "a respectful email with a subject line"),
    ("review", "a React component", "a junior frontend developer", "correctness, accessibility, maintainability, and edge cases", "prioritized findings with code examples for critical issues"),
    ("teach", "binary search", "a student who knows arrays", "the loop invariant, runtime, and off-by-one errors", "a traced example and language-neutral pseudocode"),
    ("recommend", "a laptop for university coding", "a student", "budget, battery, RAM, portability, and software needs", "a shortlist with trade-offs; ask for the budget if missing"),
    ("outline", "a research report on prompt engineering", "a university evaluator", "research questions, methods, limitations, and evaluation", "a structured outline with suggested evidence types"),
    ("refactor", "a long Python script", "a developer maintaining it", "separation of concerns, naming, tests, and behavior preservation", "a refactoring plan before code changes"),
    ("calculate", "the monthly payment on a loan", "a borrower comparing options", "principal, annual interest rate, term, and assumptions", "the formula, a worked example, and a note about fees"),
    ("plan", "a small community coding workshop", "first-time participants", "learning goals, preparation, accessibility, and activities", "a two-hour agenda with materials and timings"),
    ("write", "a product description for a reusable bottle", "online shoppers", "capacity, materials, care, and realistic benefits", "a 100-word description without unsupported claims"),
    ("debug", "a SQL query that returns duplicate rows", "a data analyst", "join cardinality and the underlying data issue", "a diagnostic checklist and corrected query"),
    ("compare", "TF-IDF and sentence embeddings", "a student building a text classifier", "data needs, semantic handling, runtime, and interpretability", "a table and baseline recommendation"),
    ("summarize", "a long policy document", "a team deciding whether to adopt it", "obligations, exceptions, deadlines, and open questions", "an executive summary and a checklist with section references"),
    ("create", "unit tests for a date parser", "a Python developer", "valid inputs, invalid inputs, time zones, and boundaries", "pytest code grouped by behavior"),
    ("design", "a database schema for an attendance app", "a backend developer", "users, sessions, attendance events, uniqueness, and auditability", "an ER-style description and SQL table definitions"),
    ("explain", "overfitting in machine learning", "a beginner", "training versus validation performance and ways to detect it", "a short analogy, a simple example, and practical mitigations"),
    ("draft", "a sponsorship proposal for a college event", "a local business", "audience reach, benefits, deliverables, and follow-up", "a one-page outline with clear sponsorship tiers"),
    ("evaluate", "two prompt rewriting strategies", "a project supervisor", "intent preservation, factuality, length, and usefulness", "an evaluation matrix and a reproducible test plan"),
]


def build_examples():
    examples, labels, groups = [], [], []
    for i, (verb, subject, audience, scope, output) in enumerate(TASKS):
        weak_options = [
            f"help me with {subject}",
            f"{verb} {subject}",
            f"tell me something about {subject}",
        ]
        strong_options = [
            f"Please {verb} {subject} for {audience}. Focus on {scope}. Present the result as {output}. If essential information is missing, ask briefly rather than inventing facts.",
            f"Task: {verb} {subject}. Audience: {audience}. Scope: {scope}. Output: {output}. Preserve accuracy and state important assumptions.",
            f"{verb.capitalize()} {subject} for {audience}, covering {scope}. Use {output}. Do not invent facts; flag uncertainty and request critical missing information.",
        ]
        for j, (weak, strong) in enumerate(zip(weak_options, strong_options)):
            group_id = f"task-{i:03d}-variant-{j}"
            examples.extend([weak, strong])
            labels.extend([0, 1])
            groups.extend([group_id, group_id])
    return examples, labels, groups


def main():
    examples, labels, groups = build_examples()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(examples, labels, groups=groups))
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=25000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=30000, sublinear_tf=True)
    from sklearn.pipeline import FeatureUnion
    pipeline = Pipeline([
        ("features", FeatureUnion([("word", word), ("char", char)])),
        ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)),
    ])
    x_train = [examples[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    x_test = [examples[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    calculated_metrics = classification_metrics(y_test, predictions, probabilities)
    metrics = {
        "holdout_accuracy": calculated_metrics["accuracy"],
        "holdout_precision": calculated_metrics["precision"],
        "holdout_recall": calculated_metrics["recall"],
        "holdout_f1": calculated_metrics["f1"],
        "holdout_roc_auc": calculated_metrics["roc_auc"],
        "train_examples": len(train_idx),
        "holdout_examples": len(test_idx),
        "unique_task_variants": len(set(groups)),
    }
    version = "pqa-seed-synthetic-v0.1"
    joblib.dump(pipeline, MODEL_DIR / "prompt_preference.joblib", compress=3)
    metadata = {
        "model_version": version,
        "created_utc_date": date.today().isoformat(),
        "model_type": "word_and_character_tfidf_logistic_regression",
        "training_data": "synthetic contrastive seed prompts authored for development; NOT the BPO dataset",
        "label_semantics": "0=seed weak/original-style prompt; 1=seed explicit/structured prompt; estimated relative preference, not human ground-truth quality",
        "feature_config": {"word_ngrams": [1, 2], "char_ngrams": [3, 5], "max_word_features": 25000, "max_char_features": 30000},
        "random_seed": 42,
        "metrics": metrics,
        "limitations": [
            "This bootstrap artifact is trained on synthetic examples and is not suitable as evidence of real-world predictive performance.",
            "Labels indicate the construction style of seed pairs, not an objective universal prompt-quality score.",
            "The Colab notebook should replace this artifact after downloading and evaluating zai-org/BPO.",
            "The 0-100 preference signal is not a calibrated probability of successful LLM output.",
        ],
        "production_ready": False,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    for prompt in ["help me with my resume", "Please compare PostgreSQL and MySQL for a student choosing a database. Use a table and give a conditional recommendation."]:
        print(f"P(optimized-style)={pipeline.predict_proba([prompt])[0][1]:.3f} | {prompt}")

if __name__ == "__main__":
    main()
