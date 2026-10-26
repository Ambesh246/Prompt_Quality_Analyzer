# Prompt Quality Rubric v1.0

The rubric returns scores from 0 to 100 for each dimension. It uses explicit text cues and task-aware defaults. A weighted rubric score is combined with a learned original-vs-optimized preference proxy at 80/20. The final score is an estimate, not objective ground truth.

| Dimension | Weight | What it estimates | Signals and caveats |
|---|---:|---|---|
| Clarity | 14% | Whether the request appears understandable | Task verbs, noun cues, vagueness cues. A brief request can still be clear. |
| Context | 9% | Whether necessary background/input/goal is present | Context terms, supplied input markers and purpose cues. For simple definitions, context is treated as optional. |
| Specificity | 12% | Whether the task's subject/scope/details are sufficiently concrete | Numeric requirements, examples, comparison/criteria terms and vague language. |
| Task definition | 14% | Whether the requested action can be identified | Task verbs, task noun cues and questions/instructions. |
| Constraints | 10% | Whether useful limits or requirements are stated | Must/avoid/length/budget/deadline cues. Simple questions receive a neutral default. |
| Output format | 8% | Whether a needed output structure is specified | Table/list/JSON/report/email/code cues. Missing format is not treated as a problem for a simple factual question. |
| Audience & purpose | 7% | Whether audience or usage context is clear when it matters | Explicit audience and goal cues. Optional for many simple tasks. |
| Completeness | 10% | Whether enough information exists to start usefully | Task, object, supplied data/context and constraints. Not a measure of prompt length alone. |
| Actionability | 10% | Whether instructions are reasonably followable | Action/task cues, expected artifact and constraints/format cues. |
| Ambiguity & redundancy | 6% | Whether vague references/repetitions may obscure the task | Vague wording, pronouns without clear antecedents and repeated words. This is an approximate surface check, not deep semantic analysis. |

## Category thresholds

- 0–39: Needs Improvement
- 40–59: Fair
- 60–79: Good
- 80–100: Excellent

## Configuration and score composition

The dimension weights are in `DIMENSION_WEIGHTS` in `backend/app/services/scoring.py`. The weighted dimension average is called `rubric_score`. The endpoint's overall score is:

`overall = 0.80 × rubric_score + 0.20 × learned_preference_score`

The model signal is the classifier probability that the text resembles the optimized side of a training pair, mapped to 0–100. It is **not** calibrated as a success probability. If no model artifact is available, `/health` reports degraded and `/analyze` returns 503 rather than silently substituting a fake score.

## Actionable explanation rules

Low dimension values produce prioritized recommendations. The recommendations are derived from the actual detected features and dimension values; they are not evidence that a particular LLM will follow them. Strengths are the top dimensions when values are at least 76. Short and simple prompts get context/format/constraint defaults to reduce needless length bias.

## Limitations

The rubric is deliberately interpretable, but its regex cues are imperfect, language-dependent and not semantic truth. The optional learned signal inherits the dataset's biases. A higher score is not a guarantee of higher task performance; validate rewritten prompt outputs empirically for important applications.
