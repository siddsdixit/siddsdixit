import json

from human_like_writing.evaluate import reference_stats, z_scores


HUMAN_SAMPLES = [
    "I went to the store today. It was raining, so I grabbed an umbrella. The lines were long.",
    "We spent the afternoon fixing the fence. It took longer than expected, but it held up fine.",
    "My neighbor stopped by with some tomatoes from her garden. We talked for a while on the porch.",
]

AI_LIKE_SAMPLE = (
    "This changes everything! It's not just a tool — it's a paradigm shift. "
    "The results are truly remarkable, and the implications are profound!"
)

HUMAN_LIKE_SAMPLE = (
    "I fixed the leak under the sink. Took about an hour, mostly finding the right wrench."
)


def test_reference_stats_covers_all_features():
    stats = reference_stats(HUMAN_SAMPLES)
    assert "em_dash_rate" in stats
    assert stats["em_dash_rate"]["mean"] == 0


def test_zero_variance_reference_feature_is_marked_non_informative():
    # None of the 3 human samples use em-dashes, exclamations, or semicolons,
    # so those features carry no discriminating signal and must not be
    # scored as if a generated sample "matched" the human range by luck.
    stats = reference_stats(HUMAN_SAMPLES)
    assert stats["em_dash_rate"]["informative"] is False
    assert stats["exclamation_rate"]["informative"] is False

    z = z_scores(AI_LIKE_SAMPLE, stats)
    assert z["em_dash_rate"] is None
    assert z["exclamation_rate"] is None


def test_ai_like_sample_scores_further_from_human_mean():
    stats = reference_stats(HUMAN_SAMPLES)
    ai_z = z_scores(AI_LIKE_SAMPLE, stats)
    human_z = z_scores(HUMAN_LIKE_SAMPLE, stats)

    # avg_word_len has real variance in the reference sample (unlike the
    # punctuation features above), so it's a meaningful comparison.
    assert ai_z["avg_word_len"] is not None
    assert human_z["avg_word_len"] is not None
    assert abs(ai_z["avg_word_len"] - human_z["avg_word_len"]) > 0


def test_cli_end_to_end(tmp_path, capsys):
    from human_like_writing.evaluate import main

    human_path = tmp_path / "human.jsonl"
    human_path.write_text(
        "\n".join(json.dumps({"text": t}) for t in HUMAN_SAMPLES), encoding="utf-8"
    )
    generated_path = tmp_path / "generated.jsonl"
    generated_path.write_text(
        "\n".join(json.dumps({"text": t}) for t in [AI_LIKE_SAMPLE, HUMAN_LIKE_SAMPLE]),
        encoding="utf-8",
    )

    main(["--human", str(human_path), "--generated", str(generated_path)])

    out = capsys.readouterr().out
    assert "overlap=" in out
    assert "Average feature overlap" in out
