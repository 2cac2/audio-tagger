"""Tests for the transliteration-tolerant text layer (``audio_tagger.translit``).

All pure-function, stdlib-only — no fakes needed.
"""

from audio_tagger.translit import (
    extract_film_from_title,
    normalize_roman_hindi,
    sim,
)


# --------------------------------------------------------------------------- #
# Romanization variant folding.
# --------------------------------------------------------------------------- #
def test_humein_variants_collapse_to_one_form():
    forms = ["humein", "hume", "humey", "Humein", "HUME"]
    normed = {normalize_roman_hindi(f) for f in forms}
    assert normed == {"hame"}, normed


def test_kabhi_kabhie_normalize_equal_and_similar():
    assert normalize_roman_hindi("Kabhie") == normalize_roman_hindi("Kabhi") == "kabhi"
    # A whole-title comparison stays high despite the spelling drift.
    assert sim("Kabhi Kabhie", "Kabhi Kabhi") >= 0.9


def test_char_folds_bring_spellings_together():
    # ee->i, oo->u, ph->f, w->v, z->j
    assert sim("zindagi", "jindagi") >= 0.9
    assert sim("khoobsurat", "khubsurat") >= 0.85
    assert sim("pyaar", "pyar") >= 0.85


# --------------------------------------------------------------------------- #
# Film extraction from streaming-service title suffixes.
# --------------------------------------------------------------------------- #
def test_extract_film_from_from_suffix():
    clean, film = extract_film_from_title('Kesariya (From "Brahmastra")')
    assert film == "Brahmastra"
    assert clean == "Kesariya"


def test_extract_film_from_bare_parenthetical():
    clean, film = extract_film_from_title("Channa Mereya (Ae Dil Hai Mushkil)")
    assert film == "Ae Dil Hai Mushkil"
    assert clean == "Channa Mereya"


def test_version_marker_is_not_a_film():
    clean, film = extract_film_from_title("Kesariya (Remix)")
    assert film is None
    assert clean == "Kesariya (Remix)"


def test_no_parenthetical_returns_title_unchanged():
    clean, film = extract_film_from_title("Brown Munde")
    assert film is None
    assert clean == "Brown Munde"


# --------------------------------------------------------------------------- #
# Similarity edge cases.
# --------------------------------------------------------------------------- #
def test_sim_empty_is_zero():
    assert sim("", "anything") == 0.0
    assert sim("anything", None) == 0.0
    assert sim(None, None) == 0.0


def test_sim_identical_is_one():
    assert sim("Kesariya", "Kesariya") == 1.0
