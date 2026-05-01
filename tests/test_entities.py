"""Tests for the minimal entity extractor used by cross-session links."""

from mnemoria.entities import entity_overlap, extract_entities


def test_extracts_person_names():
    text = "Caroline: Thanks, Melanie! I painted it because it was calming."
    assert extract_entities(text) == ["caroline", "melanie"]


def test_extracts_acronyms():
    text = "When did Caroline go to the LGBTQ support group?"
    # LGBTQ acronym + Caroline title-case
    out = extract_entities(text)
    assert "lgbtq" in out
    assert "caroline" in out
    # "When" must be filtered as a sentence starter.
    assert "when" not in out


def test_strips_sentence_starters():
    text = "Thanks, the new feature works great. The team is happy."
    # All capitalized words here are starters or in starter list.
    assert extract_entities(text) == []


def test_dedupes_case_insensitively():
    text = "Caroline mentioned Caroline's new role and Caroline agrees."
    out = extract_entities(text)
    # "caroline" only once, even though it appears thrice (one with apostrophe)
    assert out.count("caroline") == 1


def test_multi_word_phrase():
    text = "We met at New York City last month."
    out = extract_entities(text)
    assert "new york city" in out


def test_empty_input():
    assert extract_entities("") == []
    assert extract_entities("   ") == []


def test_entity_overlap_counts_substring_matches():
    fact = "Caroline mentioned the LGBTQ support group meeting on Tuesday."
    assert entity_overlap(["caroline"], fact) == 1
    assert entity_overlap(["caroline", "lgbtq"], fact) == 2
    assert entity_overlap(["melanie"], fact) == 0
    assert entity_overlap([], fact) == 0
    assert entity_overlap(["caroline"], "") == 0
