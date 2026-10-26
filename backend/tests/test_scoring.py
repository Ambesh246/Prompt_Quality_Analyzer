from app.services.scoring import DIMENSION_WEIGHTS, assess_prompt, make_score


def test_all_ten_dimensions_and_bounds():
    result = assess_prompt("Please explain photosynthesis in simple terms, with one brief example.")
    assert len(result.dimensions) == 10
    assert set(result.dimensions) == set(DIMENSION_WEIGHTS)
    assert all(0 <= value <= 100 for value in result.dimensions.values())
    assert 0 <= result.score <= 100


def test_simple_question_is_not_penalized_for_lack_of_format():
    simple = assess_prompt("What is photosynthesis?")
    complex_without_format = assess_prompt("Analyze a dataset for trends and anomalies and recommend improvements")
    assert simple.dimensions["output_format"] >= 70
    assert simple.dimensions["constraints"] >= 70
    assert simple.score > 55
    assert complex_without_format.dimensions["output_format"] < 75


def test_vague_prompt_has_actionable_recommendations():
    vague = assess_prompt("help me with this, make it better and do something good")
    assert vague.features["vague_cue_count"] > 0
    assert vague.weaknesses
    assert vague.recommendations


def test_short_prompt_is_not_automatically_scored_low_due_to_length():
    short = assess_prompt("Define entropy.")
    assert short.features["word_count"] == 2
    assert short.score >= 60


def test_combination_reports_method_and_clamps_score():
    score, method = make_score(80, 120)
    assert score == 84.0
    assert "learned_pair_preference_proxy" in method
    assert make_score(72, None) == (72, "explainable_rubric_only")


def test_simple_rewrite_does_not_produce_broken_punctuation():
    from app.services.rewrite import _build_rule_rewrite
    definition = assess_prompt("Define entropy.")
    rewritten, _ = _build_rule_rewrite("Define entropy.", definition)
    assert "entropy. in" not in rewritten
    translation = assess_prompt("Translate bonjour to English")
    rewrite_translation, _ = _build_rule_rewrite("Translate bonjour to English", translation)
    assert rewrite_translation == "Translate bonjour to English"


def test_classification_metrics_calculate_expected_values():
    from app.ml.evaluation import classification_metrics
    metrics = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.8, 0.4, 0.2])
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.6667
    assert metrics["roc_auc"] == 1.0


def test_classification_metrics_reject_invalid_shapes():
    from app.ml.evaluation import classification_metrics
    import pytest
    with pytest.raises(ValueError):
        classification_metrics([1], [1, 0], [0.8])


def test_loaded_model_prediction_is_bounded_and_reproducible():
    from app.ml.model import PreferenceModel
    model = PreferenceModel()
    assert model.available
    one = model.preference_score("Explain recursion to a beginner with one example.")
    two = model.preference_score("Explain recursion to a beginner with one example.")
    assert one == two
    assert one is not None and 0 <= one <= 100

def test_colab_regression_blend_uses_60_40_weights():
    score, method = make_score(
        rubric_score=50,
        ml_score=80,
        scoring_mode="colab_regression",
    )

    assert score == 68.0
    assert method == (
        "60%_colab_ridge_regression_plus_40%_rubric"
    )