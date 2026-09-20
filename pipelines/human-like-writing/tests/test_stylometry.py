from human_like_writing.stylometry import FEATURE_NAMES, extract_features


def test_feature_names_are_stable():
    assert "avg_sentence_len" in FEATURE_NAMES
    assert "em_dash_rate" in FEATURE_NAMES
    assert len(FEATURE_NAMES) == 10


def test_all_features_present_for_any_text():
    features = extract_features("Short sentence. Another one here.")
    assert set(features) == set(FEATURE_NAMES)


def test_em_dash_and_exclamation_rate_detected():
    styled = "This changes everything! It's the future — and it's here."
    flat = "This changes things. It is a development. It exists now."
    assert extract_features(styled)["em_dash_rate"] > 0
    assert extract_features(styled)["exclamation_rate"] > 0
    assert extract_features(flat)["em_dash_rate"] == 0
    assert extract_features(flat)["exclamation_rate"] == 0


def test_sentence_length_variability():
    # Uniform short sentences should have lower stdev than a mix of short/long.
    uniform = "Do this. Do that. Do more. Do less."
    varied = "Do this. Consider carefully whether doing that is actually worth the time and effort involved."
    assert extract_features(uniform)["sentence_len_stdev"] <= extract_features(varied)["sentence_len_stdev"]


def test_empty_text_does_not_crash():
    features = extract_features("")
    assert features["avg_sentence_len"] >= 0
