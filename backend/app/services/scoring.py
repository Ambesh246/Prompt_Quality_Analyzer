"""Explainable prompt rubric. The score is a context-dependent estimate, not ground truth."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

DIMENSION_WEIGHTS: dict[str, float] = {
    "clarity": 0.14,
    "context": 0.09,
    "specificity": 0.12,
    "task_definition": 0.14,
    "constraints": 0.10,
    "output_format": 0.08,
    "audience_purpose": 0.07,
    "completeness": 0.10,
    "actionability": 0.10,
    "ambiguity_redundancy": 0.06,
}

DIMENSION_LABELS = {
    "clarity": "Clarity",
    "context": "Context",
    "specificity": "Specificity",
    "task_definition": "Task definition",
    "constraints": "Constraints",
    "output_format": "Output format",
    "audience_purpose": "Audience & purpose",
    "completeness": "Completeness",
    "actionability": "Actionability",
    "ambiguity_redundancy": "Ambiguity & redundancy",
}

TASK_VERBS = re.compile(
    r"\b(explain|summari[sz]e|write|draft|create|build|compare|analy[sz]e|calculate|"
    r"debug|fix|review|translate|design|recommend|list|define|describe|evaluate|"
    r"classify|extract|generate|teach|solve|implement|outline|rewrite|plan|"
    r"research|identify|convert|refactor|test|critique|proofread|find)\b", re.I
)
CONTEXT_CUES = re.compile(
    r"\b(context|background|because|for my|for a|audience|currently|given|based on|"
    r"assuming|the following|below|here is|here's|input|dataset|data set|project|"
    r"my goal|the goal|scenario|situation|use case|using)\b", re.I
)
SPECIFICITY_CUES = re.compile(
    r"\b(three|four|five|two|one|top \d+|at least|at most|exactly|step[- ]by[- ]step|"
    r"pros and cons|with examples|including|compare|versus|vs\.?|criteria|metric|"
    r"budget|deadline|version|python|javascript|sql|react|fastapi|table|json|csv|"
    r"word count|paragraphs?|bullets?|numbered|beginner|advanced|short|concise)\b", re.I
)
CONSTRAINT_CUES = re.compile(
    r"\b(must|should|must not|don't|do not|avoid|limit|within|maximum|minimum|under|"
    r"no more than|at least|only|without|exclude|deadline|budget|word limit|"
    r"requirements?|constraints?|use .*? only|do not include)\b", re.I
)
FORMAT_CUES = re.compile(
    r"\b(format|table|bullet points?|numbered list|json|yaml|csv|markdown|code block|"
    r"paragraphs?|outline|checklist|step[- ]by[- ]step|template|report|email|"
    r"function signature|schema|columns?)\b", re.I
)
AUDIENCE_CUES = re.compile(
    r"\b(for (?:a |an |the )?(?:beginner|student|child|expert|manager|developer|"
    r"teacher|client|executive|researcher|customer|team)|audience|my professor|"
    r"my team|for school|for college|for work|intended for|grade \d+|age \d+)\b", re.I
)
PURPOSE_CUES = re.compile(
    r"\b(so that|in order to|purpose|goal|objective|to help|for the purpose|"
    r"i need this|i want to|used for|use case|decision|presentation|assignment|"
    r"interview|exam|submission|production)\b", re.I
)
VAGUE_CUES = re.compile(
    r"\b(something|anything|somehow|stuff|things|make it good|make it better|"
    r"best possible|nice|good|better|properly|as needed|whatever|etc\.?|and so on|"
    r"fix this|do this|help me|tell me everything)\b", re.I
)
TASK_NOUN_CUES = re.compile(
    r"\b(report|essay|function|program|app|application|analysis|summary|plan|"
    r"recommendation|email|message|story|lesson|explanation|answer|solution|"
    r"dataset|model|chart|presentation|proposal|outline|review|comparison|"
    r"definition|example|steps|code|query|formula|schedule|budget)\b", re.I
)
SIMPLE_QUESTION = re.compile(
    r"^\s*(what is|what are|who is|who was|when was|where is|define|meaning of|"
    r"translate\s+.+\s+to)\b", re.I
)


@dataclass
class RubricResult:
    score: float
    dimensions: dict[str, float]
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    features: dict[str, Any]
    category: str
    assessment: str


def _clamp(value: float, low: float = 0, high: float = 100) -> float:
    return round(max(low, min(high, value)), 1)


def _count_matches(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


def _category(score: float) -> str:
    if score < 40:
        return "Needs Improvement"
    if score < 60:
        return "Fair"
    if score < 80:
        return "Good"
    return "Excellent"


def assess_prompt(prompt: str) -> RubricResult:
    """Score prompt dimensions with visible, deterministic features.

    Short prompts are not penalized merely for being short. Criteria that are often
    optional for simple factual questions get neutral-to-positive defaults there.
    """
    text = prompt.strip()
    words = re.findall(r"\b[\w'-]+\b", text)
    word_count = len(words)
    char_count = len(text)
    simple = bool(SIMPLE_QUESTION.search(text)) and word_count <= 14
    task_verbs = _count_matches(TASK_VERBS, text)
    task_nouns = _count_matches(TASK_NOUN_CUES, text)
    context_cues = _count_matches(CONTEXT_CUES, text)
    specificity_cues = _count_matches(SPECIFICITY_CUES, text)
    constraint_cues = _count_matches(CONSTRAINT_CUES, text)
    format_cues = _count_matches(FORMAT_CUES, text)
    audience_cues = _count_matches(AUDIENCE_CUES, text)
    purpose_cues = _count_matches(PURPOSE_CUES, text)
    vague_cues = _count_matches(VAGUE_CUES, text)
    pronoun_cues = len(re.findall(r"\b(it|this|that|these|those|they|them)\b", text, re.I))
    repeated_words = len(words) - len(set(w.casefold() for w in words))
    has_question_or_instruction = bool(re.search(r"\?|\b(please|can you|could you|explain|write|create|build|analy[sz]e|summari[sz]e|define|tell me|show me)\b", text, re.I))
    has_task = task_verbs > 0 or "?" in text
    has_input_data = bool(re.search(r"[:\n`{}\[\]]", text))
    complex_task = (word_count >= 18 or task_nouns > 0 or bool(re.search(r"\b(project|research|analy[sz]e|build|implement|report|proposal|plan|compare|design|code|dataset)\b", text, re.I))) and not simple

    clarity = 48 + (18 if has_question_or_instruction else 0) + (12 if task_verbs else 0) + (9 if task_nouns else 0) - 12 * min(vague_cues, 3) - (6 if word_count > 100 else 0)
    if simple:
        clarity = max(clarity, 82)

    context = 45 + min(context_cues, 2) * 16 + (12 if has_input_data else 0) + (8 if purpose_cues else 0)
    if simple:
        context = 78
    elif word_count < 8:
        context = min(context, 66)

    specificity = 43 + min(specificity_cues, 2) * 13 + (12 if re.search(r"\d", text) else 0) + (10 if task_nouns else 0) - min(vague_cues, 3) * 10
    if simple:
        specificity = max(specificity, 78)

    task_definition = 40 + (30 if task_verbs else 0) + (10 if task_nouns else 0) + (10 if has_question_or_instruction else 0)
    if simple:
        task_definition = 88
    elif "?" in text:
        task_definition += 8

    constraints = 73 if simple else 43 + min(constraint_cues, 2) * 20 + (12 if re.search(r"\b\d+\s*(words?|sentences?|lines?|minutes?|hours?|pages?|items?|examples?)\b", text, re.I) else 0)
    output_format = 75 if simple else 44 + min(format_cues, 2) * 27 + (8 if re.search(r"\b(in|as)\s+(a|an|the)\s+(?:table|list|report|email|json|paragraph|checklist)\b", text, re.I) else 0)
    audience_purpose = 78 if simple else 43 + min(audience_cues, 1) * 28 + min(purpose_cues, 1) * 20
    completeness = 45 + (20 if has_task else 0) + (12 if task_nouns else 0) + (10 if context_cues or has_input_data else 0) + (10 if constraint_cues else 0)
    if simple:
        completeness = 83
    elif word_count < 5:
        completeness -= 10
    actionability = 45 + (22 if has_task else 0) + (12 if task_nouns else 0) + (10 if format_cues else 0) + (8 if constraint_cues else 0)
    if simple:
        actionability = 84
    ambiguity_redundancy = 88 - 13 * min(vague_cues, 3) - 5 * min(pronoun_cues, 4) - 2 * min(repeated_words, 8)
    if simple:
        ambiguity_redundancy = max(ambiguity_redundancy, 84)

    dimensions = {
        "clarity": _clamp(clarity),
        "context": _clamp(context),
        "specificity": _clamp(specificity),
        "task_definition": _clamp(task_definition),
        "constraints": _clamp(constraints),
        "output_format": _clamp(output_format),
        "audience_purpose": _clamp(audience_purpose),
        "completeness": _clamp(completeness),
        "actionability": _clamp(actionability),
        "ambiguity_redundancy": _clamp(ambiguity_redundancy),
    }
    weighted_score = _clamp(sum(dimensions[key] * weight for key, weight in DIMENSION_WEIGHTS.items()))

    strengths = [
        f"{DIMENSION_LABELS[key]} is a relative strength ({value:.0f}/100)."
        for key, value in sorted(dimensions.items(), key=lambda item: item[1], reverse=True)
        if value >= 76
    ][:4]
    if not strengths:
        strengths = ["The prompt contains a starting request that can be made more actionable."]

    priority_order = ["task_definition", "specificity", "context", "constraints", "output_format", "audience_purpose", "clarity", "completeness", "actionability", "ambiguity_redundancy"]
    weaknesses = [
        f"{DIMENSION_LABELS[key]} needs attention ({dimensions[key]:.0f}/100)."
        for key in priority_order if dimensions[key] < 65
    ][:4]

    recommendations: list[str] = []
    if dimensions["task_definition"] < 70:
        recommendations.append("Start with one explicit action verb and state the exact task to complete.")
    if dimensions["specificity"] < 70:
        recommendations.append("Name the subject, scope, and the level of detail that would make the answer useful.")
    if dimensions["context"] < 70:
        recommendations.append("Add only the background, input data, or goal the model genuinely needs; use placeholders for unknowns.")
    if dimensions["constraints"] < 70 and complex_task:
        recommendations.append("State relevant limits or requirements, such as tools, length, deadline, scope, or what to avoid.")
    if dimensions["output_format"] < 70 and complex_task:
        recommendations.append("Choose a helpful output structure, such as a checklist, table, code block, or sectioned report.")
    if dimensions["audience_purpose"] < 70 and complex_task:
        recommendations.append("Specify the intended audience or how you will use the result if that changes the answer.")
    if dimensions["ambiguity_redundancy"] < 70:
        recommendations.append("Replace vague words or unexplained references (for example, 'make it better') with concrete criteria.")
    if not recommendations:
        recommendations = ["Keep the prompt focused. Add constraints or context only when they would materially change the answer."]

    assessment = {
        "Needs Improvement": "The request is understandable as a starting point, but important details may be missing.",
        "Fair": "The core request is present; a few targeted details could make the expected result more predictable.",
        "Good": "The prompt gives a useful direction. A small amount of targeted refinement may make the output more consistent.",
        "Excellent": "The prompt is well-directed for its apparent task. More words are not necessarily needed.",
    }[_category(weighted_score)]

    features = {
        "word_count": word_count,
        "character_count": char_count,
        "is_simple_question": simple,
        "complex_task_detected": complex_task,
        "task_verb_count": task_verbs,
        "context_cue_count": context_cues,
        "specificity_cue_count": specificity_cues,
        "constraint_cue_count": constraint_cues,
        "format_cue_count": format_cues,
        "vague_cue_count": vague_cues,
        "repeated_word_count": repeated_words,
    }
    score_category = _category(weighted_score)
    return RubricResult(
        score=weighted_score,
        dimensions=dimensions,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recommendations,
        features=features,
        category=score_category,
        assessment=assessment,
    )



def make_score(
    rubric_score: float,
    ml_score: float | None,
    scoring_mode: str = "bpo_preference_classifier",
) -> tuple[float, str]:
    """Blend the correct ML signal with the explainable rubric."""

    bounded_rubric = _clamp(rubric_score)

    # Graceful fallback if the model cannot predict.
    if ml_score is None:
        return bounded_rubric, "explainable_rubric_only"

    bounded_ml = _clamp(ml_score)

    if scoring_mode == "colab_regression":
        combined = (
            0.60 * bounded_ml
            + 0.40 * bounded_rubric
        )

        return (
            _clamp(combined),
            "60%_colab_ridge_regression_plus_40%_rubric",
        )

    # Legacy classifier: this is a preference signal, not a
    # universal prompt-quality prediction.
    combined = (
        0.80 * bounded_rubric
        + 0.20 * bounded_ml
    )

    return (
        _clamp(combined),
        "80%_rubric_plus_20%_learned_pair_preference_proxy",
    )
