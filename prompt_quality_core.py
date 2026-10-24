"""Stable shared functions for Prompt Quality Analyzer.

Keep this module next to app.py before loading the exported joblib model.
The fitted sklearn pipeline references extract_structural_features by module path.
"""
import re
import numpy as np

def contains_phrase(text, phrase):
    """Match a phrase with word boundaries; allow flexible whitespace between words."""
    text = str(text).lower()
    phrase = str(phrase).lower().strip()
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    pattern = r"(?<!\w)" + escaped + r"(?!\w)"
    return re.search(pattern, text) is not None


def contains_any(text, phrases):
    return int(any(contains_phrase(text, phrase) for phrase in phrases))


FEATURE_PHRASES = {
    "action_verb": ["explain", "summarize", "summarise", "translate", "write", "create", "build", "debug", "compare", "analyze", "analyse", "classify", "calculate", "design", "generate", "recommend", "plan", "extract", "evaluate", "review", "rewrite", "convert", "solve", "describe", "list", "draft", "what is", "what are", "who is", "who was", "why", "how do", "how does", "how to", "tell me", "give me", "can you", "help me", "find out"],
    "context": ["context", "background", "given", "based on", "following", "here is", "scenario", "for this task", "the problem is"],
    "audience": ["audience", "for beginners", "for a beginner", "for students", "for a student", "for experts", "for developers", "target reader", "second-year", "second year", "non-technical audience"],
    "format": ["output format", "table", "bullet points", "json", "markdown", "headings", "numbered steps", "numbered list", "csv", "code block", "paragraphs", "checklist"],
    "constraints": ["must", "should", "do not", "don't", "avoid", "limit", "maximum", "minimum", "exactly", "under", "at most", "no more than", "within", "word limit", "less than"],
    "examples": ["example", "examples", "sample", "test case", "test cases", "worked example", "illustration"],
    "success_criteria": ["success criteria", "acceptance criteria", "expected result", "expected output", "verify", "validate", "check correctness", "definition of done", "quality criteria", "evaluate against"],
    "role": ["act as", "you are a", "as an expert", "as a professional", "role-play as"],
    "input_data": ["input", "input data", "provided code", "following code", "dataset", "csv file", "document", "passage", "text:", "data:", "attached file", "below"],
    "programming_language": ["python", "javascript", "typescript", "java", "c++", "c#", "sql", "bash", "go", "rust", "r language", "pseudocode"],
    "expected_behavior": ["expected output", "should return", "must return", "return a", "output should", "expected behavior", "expected behaviour", "desired result"],
    "tests": ["test", "tests", "unit test", "test case", "edge case", "assert"],
    "edge_cases": ["edge case", "empty input", "invalid input", "error handling", "boundary condition", "missing values", "exceptions"],
    "tone": ["tone", "formal", "friendly", "professional", "persuasive", "humorous", "conversational", "academic", "casual", "empathetic"],
    "length": ["word limit", "words", "paragraphs", "concise", "brief", "under 500", "under 200", "length", "maximum length"],
    "purpose": ["purpose", "objective", "aim", "intended to", "to persuade", "to inform", "to explain", "to announce"],
    "required_content": ["include", "cover", "mention", "must include", "address", "discuss", "compare", "summarize", "summarise"],
    "target_language": ["target language", "into english", "into spanish", "into hindi", "into french", "into german", "into bengali", "to english", "to spanish", "to hindi", "to french", "to german", "to bengali"],
    "preserve_format": ["preserve formatting", "keep formatting", "maintain line breaks", "preserve tone", "keep the original format", "preserve the original meaning"],
    "scope": ["scope", "focus on", "between", "from 2020", "from 2021", "as of", "in the last", "region", "country", "time period", "timeframe"],
    "sources": ["cite sources", "sources", "references", "evidence", "peer-reviewed", "primary sources", "citations", "cite your sources"],
    "methodology": ["methodology", "method", "research design", "systematic review", "compare approaches", "criteria", "analyze the data", "analyse the data"],
    "uncertainty": ["assumptions", "limitations", "uncertainty", "caveats", "bias", "state assumptions", "if unsure"],
    "metrics": ["metric", "metrics", "accuracy", "precision", "recall", "f1", "rmse", "mae", "r2", "r-squared"],
    "validation": ["validate", "validation", "test set", "cross-validation", "sanity check", "verify", "quality check"]
}


def extract_structural_features(texts):
    rows = []
    for value in texts:
        text = str(value)
        lower = text.lower()
        words = re.findall(r"\b[\w'-]+\b", lower)
        word_count = len(words)
        char_count = len(text)
        sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
        unique_ratio = len(set(words)) / max(word_count, 1)
        avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
        digit_ratio = sum(ch.isdigit() for ch in text) / max(char_count, 1)
        punctuation_ratio = sum(ch in r"!#$%&'()*+,-./:;<=>?@[\]^_`{|}~" for ch in text) / max(char_count, 1)
        list_markers = int(bool(re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text)))
        signal_values = [contains_any(lower, FEATURE_PHRASES[k]) for k in FEATURE_PHRASES]
        rows.append([
            np.log1p(char_count), np.log1p(word_count), avg_word_len,
            unique_ratio, np.log1p(len(sentences)), np.log1p(text.count("?")),
            np.log1p(text.count("!")), np.log1p(text.count("\n")),
            digit_ratio, punctuation_ratio, list_markers,
            *signal_values
        ])
    return np.asarray(rows, dtype=float)


# Each task has its own checklist. Weights add to 100 for each task type.
RUBRICS = {
    "general": [
        ("task", "A clear action/task is identifiable", 20, ["explain", "summarize", "summarise", "write", "create", "compare", "analyze", "analyse", "translate", "debug", "calculate", "design", "generate", "recommend", "plan", "solve", "describe", "review", "draft", "list", "what is", "what are", "who is", "who was", "why", "how do", "how does", "how to", "tell me", "give me", "can you", "help me", "find out"]),
        ("context", "Relevant context/input is supplied", 15, FEATURE_PHRASES["context"] + ["text:", "data:", "code:"]),
        ("audience", "Audience or knowledge level is specified", 10, FEATURE_PHRASES["audience"]),
        ("format", "Desired output format is clear", 15, FEATURE_PHRASES["format"]),
        ("constraints", "Important constraints/requirements are stated", 15, FEATURE_PHRASES["constraints"]),
        ("examples", "Examples or samples are requested/provided when useful", 10, FEATURE_PHRASES["examples"]),
        ("success", "Success/verification criteria are stated", 15, FEATURE_PHRASES["success_criteria"]),
    ],
    "coding": [
        ("task", "Programming task is identifiable", 15, FEATURE_PHRASES["action_verb"]),
        ("language", "Programming language is specified", 15, FEATURE_PHRASES["programming_language"]),
        ("input", "Input, code, or problem details are provided", 15, FEATURE_PHRASES["input_data"] + ["function:", "bug:", "error:"]),
        ("behavior", "Expected behavior/output is stated", 15, FEATURE_PHRASES["expected_behavior"]),
        ("edges", "Edge cases or error handling are mentioned", 10, FEATURE_PHRASES["edge_cases"]),
        ("tests", "Tests or test cases are requested", 10, FEATURE_PHRASES["tests"]),
        ("constraints", "Constraints are included", 10, FEATURE_PHRASES["constraints"]),
        ("format", "Code/explanation output format is described", 10, FEATURE_PHRASES["format"] + ["provide code", "explain the code"]),
    ],
    "education": [
        ("task", "Learning task/topic is identifiable", 20, FEATURE_PHRASES["action_verb"]),
        ("audience", "Learner level/audience is specified", 20, FEATURE_PHRASES["audience"] + ["school student", "college student", "second year", "beginner level"]),
        ("context", "Topic or learning context is indicated", 10, FEATURE_PHRASES["context"] + ["topic", "chapter", "subject"]),
        ("examples", "Examples or worked examples are requested", 15, FEATURE_PHRASES["examples"]),
        ("format", "Teaching/output structure is specified", 10, FEATURE_PHRASES["format"] + ["step by step", "step-by-step"]),
        ("constraints", "Depth, length, or scope constraints are present", 10, FEATURE_PHRASES["constraints"] + FEATURE_PHRASES["length"]),
        ("success", "Practice/check-understanding criteria are included", 15, FEATURE_PHRASES["success_criteria"] + ["quiz", "practice questions", "check my understanding"]),
    ],
    "writing": [
        ("task", "Writing purpose/task is identifiable", 20, FEATURE_PHRASES["action_verb"] + FEATURE_PHRASES["purpose"]),
        ("audience", "Intended audience is specified", 15, FEATURE_PHRASES["audience"]),
        ("tone", "Tone or voice is specified", 10, FEATURE_PHRASES["tone"]),
        ("format", "Output format is specified", 10, FEATURE_PHRASES["format"]),
        ("length", "Length or size is specified", 10, FEATURE_PHRASES["length"] + FEATURE_PHRASES["constraints"]),
        ("content", "Required content is listed", 15, FEATURE_PHRASES["required_content"]),
        ("examples", "Style/sample reference is provided when helpful", 10, FEATURE_PHRASES["examples"] + ["in the style of", "match this style"]),
        ("constraints", "Other important constraints are stated", 10, FEATURE_PHRASES["constraints"]),
    ],
    "research": [
        ("task", "Research question/objective is identifiable", 15, FEATURE_PHRASES["action_verb"] + FEATURE_PHRASES["purpose"]),
        ("scope", "Scope, timeframe, population, or geography is defined", 15, FEATURE_PHRASES["scope"]),
        ("sources", "Evidence/source requirements are specified", 20, FEATURE_PHRASES["sources"]),
        ("method", "Method or comparison approach is described", 15, FEATURE_PHRASES["methodology"]),
        ("format", "Expected report structure is specified", 10, FEATURE_PHRASES["format"]),
        ("constraints", "Research constraints are included", 10, FEATURE_PHRASES["constraints"]),
        ("uncertainty", "Limitations, assumptions, or uncertainty are requested", 15, FEATURE_PHRASES["uncertainty"]),
    ],
    "data_analysis": [
        ("task", "Analysis objective is identifiable", 15, FEATURE_PHRASES["action_verb"] + FEATURE_PHRASES["purpose"]),
        ("data", "Dataset/input details are provided", 20, FEATURE_PHRASES["input_data"] + FEATURE_PHRASES["metrics"] + ["columns", "rows", "spreadsheet"]),
        ("metrics", "Variables or metrics are named", 15, FEATURE_PHRASES["metrics"]),
        ("assumptions", "Assumptions/constraints are addressed", 10, FEATURE_PHRASES["uncertainty"] + FEATURE_PHRASES["constraints"]),
        ("format", "Desired output/visualization format is specified", 10, FEATURE_PHRASES["format"] + ["chart", "plot", "graph", "dashboard"]),
        ("validation", "Validation or quality checks are requested", 15, FEATURE_PHRASES["validation"] + FEATURE_PHRASES["success_criteria"]),
        ("audience", "Audience/context is specified", 5, FEATURE_PHRASES["audience"]),
        ("scope", "Analysis scope is clear", 10, FEATURE_PHRASES["scope"] + ["focus on", "compare groups"]),
    ],
    "translation": [
        ("task", "Translation request is identifiable", 20, ["translate", "translation", "render in", "convert into"]),
        ("source", "Source text appears to be provided", 25, ["text:", "passage:", "translate", "following text", "quoted text"]),
        ("language", "Target language is specified", 25, FEATURE_PHRASES["target_language"] + ["into italian", "into portuguese", "into japanese", "into korean", "into arabic", "into urdu", "into marathi", "into tamil", "into telugu", "into gujarati", "into punjabi", "into chinese", "into russian"]),
        ("context", "Context or audience is specified when nuance matters", 10, FEATURE_PHRASES["context"] + FEATURE_PHRASES["audience"]),
        ("tone", "Tone/register requirements are specified", 10, FEATURE_PHRASES["tone"] + ["formal register", "informal register", "polite form"]),
        ("format", "Formatting/meaning preservation is specified", 5, FEATURE_PHRASES["preserve_format"] + FEATURE_PHRASES["format"]),
        ("constraints", "Translation constraints are specified", 5, FEATURE_PHRASES["constraints"]),
    ]
}

assert all(sum(row[2] for row in rows) == 100 for rows in RUBRICS.values())


SUGGESTIONS = {
    "task": "State the exact action and the result you want.",
    "context": "Add the relevant background, input, or situation; avoid making the reader guess essential details.",
    "audience": "Specify who the answer is for and the expected knowledge level, if that affects the answer.",
    "format": "Specify the preferred output format, such as steps, a table, bullets, code, or a short report.",
    "constraints": "Add relevant limits or requirements (for example, length, tools, scope, or things to avoid).",
    "examples": "Add an example or sample input/output if it would clarify what a good result looks like.",
    "success": "Define what a satisfactory answer should contain and how it should be checked.",
    "language": "Specify the programming language and, if relevant, its version or runtime.",
    "input": "Provide the relevant code/data/input and include any error message or sample where applicable.",
    "behavior": "Describe expected behaviour and output, including how success should look.",
    "edges": "Mention important edge cases, invalid inputs, and expected error handling.",
    "tests": "Ask for test cases covering normal and edge-case inputs.",
    "tone": "Specify the desired tone or voice.",
    "length": "Set a suitable length or level of detail.",
    "content": "List the key points or content that must be included.",
    "scope": "Bound the scope, timeframe, population, or geographical context.",
    "sources": "Specify acceptable evidence or source requirements and whether citations are needed.",
    "method": "State the analysis or comparison method, or ask the assistant to explain its method.",
    "uncertainty": "Ask the answer to state assumptions, limitations, and uncertainty rather than inventing facts.",
    "data": "Specify the dataset, columns, units, missing-value rules, or input files.",
    "metrics": "Name the outcome metrics or variables to analyze.",
    "assumptions": "State assumptions explicitly and explain how conclusions depend on them.",
    "validation": "Ask for sanity checks, validation steps, or a way to verify the result.",
    "source": "Clearly separate the source text from the instructions.",
    "format_preserve": "State whether original formatting, tone, and meaning must be preserved."
}


def get_suggestion(key, task_type):
    if task_type == "translation" and key == "language":
        return "Specify the target language (for example, Spanish, Hindi, or French)."
    if task_type == "translation" and key == "source":
        return "Clearly delimit the source text to translate, for example with quotation marks or a 'Text:' label."
    if task_type == "research" and key == "sources":
        return "Specify the evidence standard and whether citations or primary sources are required."
    if task_type == "coding" and key == "language":
        return "Specify the programming language and, if relevant, its version/runtime."
    return SUGGESTIONS.get(key, f"Review this dimension if it is relevant: {key}.")


def detect_criteria(prompt, task_type="general"):
    text = str(prompt).strip()
    normalized = re.sub(r"\s+", " ", text.lower())
    results = []
    for key, label, weight, phrases in RUBRICS[task_type]:
        detected = contains_any(normalized, phrases)
        # Translation source signal: quotes or text delimiter help; otherwise use a basic length check.
        if task_type == "translation" and key == "source":
            has_delimited_text = bool(re.search(r"""'[^']+'|"[^"]+"|(?:text|passage)\s*:""", text, flags=re.I | re.S))
            detected = detected and (has_delimited_text or len(re.findall(r"\b\w+\b", text)) >= 4)
        results.append({"key": key, "dimension": label, "weight": weight, "detected": bool(detected)})
    return results


def rubric_score(criteria_results):
    return float(sum(row["weight"] for row in criteria_results if row["detected"]))


def build_improved_prompt(prompt, task_type, criteria_results):
    original = str(prompt).strip()
    missing = sorted([row for row in criteria_results if not row["detected"]], key=lambda row: row["weight"], reverse=True)
    # Choose a concise task-specific skeleton that preserves the original request.
    if task_type == "coding":
        template = [
            "Original task (preserve its intent):",
            original,
            "\nInstructions:",
            "- Use the programming language/runtime specified in the original request; if missing and essential, ask or clearly state an assumption.",
            "- Explain the expected behaviour and provide a practical solution.",
            "- Cover relevant edge cases and error handling; include tests where useful.",
            "- Keep the response organized and avoid changing the requested functionality.",
        ]
    elif task_type == "education":
        template = [
            "Original learning request (preserve its intent):",
            original,
            "\nTeaching guidance:",
            "- Match the explanation to the audience and level stated in the original request; if unspecified, use a beginner-friendly level and say so.",
            "- Explain key ideas clearly, define necessary technical terms, and use a relevant example when it helps.",
            "- Organize the answer with clear sections and a short recap; do not add unrelated material.",
        ]
    elif task_type == "writing":
        template = [
            "Original writing task (preserve its intent):",
            original,
            "\nWriting guidance:",
            "- Follow any audience, tone, format, and length specified in the original request.",
            "- If an important writing preference is missing, make a reasonable minimal assumption or ask a concise question.",
            "- Include the requested key content, avoid unrelated additions, and produce a polished final draft.",
        ]
    elif task_type == "research":
        template = [
            "Original research request (preserve its intent):",
            original,
            "\nResearch guidance:",
            "- Clarify the scope and timeframe when they materially affect the answer.",
            "- Support factual claims with reliable sources when available; do not fabricate citations.",
            "- Separate evidence from inference, explain the method or comparison criteria, and state uncertainty and limitations.",
            "- Organize findings with a concise conclusion and relevant caveats.",
        ]
    elif task_type == "data_analysis":
        template = [
            "Original data-analysis request (preserve its intent):",
            original,
            "\nAnalysis guidance:",
            "- Identify the data/input, analytical goal, relevant variables, and evaluation metrics.",
            "- State assumptions, units, missing-data handling, and limitations where relevant.",
            "- Use appropriate tables or visualizations, validate results, and distinguish observed results from interpretation.",
        ]
    elif task_type == "translation":
        template = [
            "Original translation request (preserve its intent):",
            original,
            "\nTranslation guidance:",
            "- Preserve the source meaning; do not add information that is not in the source.",
            "- Use the target language named in the original request. If it is missing or ambiguous, ask for clarification.",
            "- Preserve tone and formatting where appropriate; explain an ambiguity briefly rather than guessing if it changes the meaning.",
        ]
    else:
        template = [
            "Original request (preserve its intent):",
            original,
            "\nResponse guidance:",
            "- Answer the original task directly and use only relevant context; do not expand the task unnecessarily.",
            "- Follow any audience, constraints, and output format already specified in the original request.",
            "- If an essential detail is missing, ask a concise clarifying question or state a reasonable assumption.",
            "- Organize the response clearly and distinguish facts from assumptions or uncertainty.",
        ]
    # Add no more than three highly relevant diagnostic reminders. The guidance remains generic by design.
    if missing:
        reminders = []
        for item in missing:
            suggestion = get_suggestion(item["key"], task_type)
            if suggestion and suggestion not in reminders:
                reminders.append(suggestion)
            if len(reminders) == 3:
                break
        if reminders:
            template.extend(["\nPotential details to clarify (only if relevant):"])
            template.extend([f"- {item}" for item in reminders])
    template.append("\nKeep the user's original objective unchanged. Do not invent missing facts or impose unnecessary requirements.")
    return "\n".join(template)


def score_band(score):
    # Illustrative display bands; they are not validated universal standards.
    if score < 40:
        return "Needs improvement"
    if score < 65:
        return "Developing"
    if score < 85:
        return "Good foundation"
    return "Strong prompt structure"


__all__ = [
    "FEATURE_PHRASES", "contains_phrase", "contains_any",
    "extract_structural_features", "RUBRICS", "SUGGESTIONS",
    "get_suggestion", "detect_criteria", "rubric_score",
    "build_improved_prompt", "score_band",
]
