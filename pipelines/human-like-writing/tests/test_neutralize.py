from human_like_writing.neutralize import neutralize_mock


def test_expands_contractions():
    assert "do not" in neutralize_mock("I don't think so.").lower()
    assert "it is" in neutralize_mock("It's fine.").lower()


def test_strips_emphasis_punctuation():
    out = neutralize_mock("This is amazing! Really?!")
    assert "!" not in out
    assert "?" not in out


def test_replaces_em_dash_with_comma():
    out = neutralize_mock("The plan—ambitious as it is—works.")
    assert "—" not in out
    assert "," in out


def test_idempotent_on_already_neutral_text():
    text = "The report lists three findings. Each finding has a citation."
    assert neutralize_mock(text) == text
