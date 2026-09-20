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


def test_ai_like_sample_scores_further_from_human_mean():
    stats = reference_stats(HUMAN_SAMPLES)
    ai_z = z_scores(AI_LIKE_SAMPLE, stats)
    human_z = z_scores(HUMAN_LIKE_SAMPLE, stats)

    ai_em_dash_z = abs(ai_z["em_dash_rate"])
    human_em_dash_z = abs(human_z["em_dash_rate"])
    assert ai_em_dash_z > human_em_dash_z

    ai_exclaim_z = abs(ai_z["exclamation_rate"])
    human_exclaim_z = abs(human_z["exclamation_rate"])
    assert ai_exclaim_z > human_exclaim_z


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
