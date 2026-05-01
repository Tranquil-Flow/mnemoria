"""Tests for the conversational-filler classifier in retrieval.py.

Used by score_candidates to demote tone-matched supportive turns that
otherwise crowd out fact-bearing turns on long-conversation corpora.
"""

from mnemoria.retrieval import _is_conversational_filler


# ── Short pure filler ─────────────────────────────────────────────────────


def test_short_thanks():
    assert _is_conversational_filler("Thanks!")
    assert _is_conversational_filler("Thank you so much!")
    assert _is_conversational_filler("Thanks Mel!")


def test_short_reactions():
    assert _is_conversational_filler("Wow, that's awesome!")
    assert _is_conversational_filler("That's so cool!")
    assert _is_conversational_filler("Awesome!")
    assert _is_conversational_filler("Yeah, exactly.")


def test_short_supportive():
    assert _is_conversational_filler("That must have been so tough.")
    assert _is_conversational_filler("Glad you're feeling better!")
    assert _is_conversational_filler("I'm so happy for you!")


# ── Longer supportive turns (the LoCoMo case) ────────────────────────────


def test_long_supportive_glad_appreciate():
    text = (
        "Caroline: Glad you agree, Caroline. Appreciate the support of those "
        "close to me. Their encouragement made me who I am."
    )
    assert _is_conversational_filler(text)


def test_long_supportive_proud_brave():
    text = (
        "Melanie: That must have been tough for you, Caroline. Respect for "
        "finding acceptance and helping others with what you've been through. "
        "You're so strong and inspiring."
    )
    # Multiple supportive markers (tough, must have been, respect, inspiring,
    # strong) and no substantive tokens — flagged as filler.
    assert _is_conversational_filler(text)


# ── Not filler when content present ──────────────────────────────────────


def test_not_filler_with_year():
    # "2022" is a year → substantive content
    text = "Thanks Mel! I painted that lake sunrise in 2022."
    assert not _is_conversational_filler(text)


def test_not_filler_with_duration():
    text = "Caroline: I've known these friends for 4 years, since I moved."
    assert not _is_conversational_filler(text)


def test_not_filler_with_acronym():
    text = "I went to a LGBTQ support group yesterday and it was so powerful."
    assert not _is_conversational_filler(text)


def test_not_filler_with_specific_facts():
    text = "Caroline: I'm thinking of moving to Sydney next month."
    assert not _is_conversational_filler(text)


def test_not_filler_long_content():
    # Long body with no obvious filler signals.
    text = (
        "We took the kids to the beach last weekend and they had a great time "
        "building sand castles. The water was warm enough to swim in for the "
        "first time this year."
    )
    # Has duration ("first time this year") and content — should not be filler
    assert not _is_conversational_filler(text)


# ── Edge cases ───────────────────────────────────────────────────────────


def test_empty_input():
    assert not _is_conversational_filler("")
    assert not _is_conversational_filler("   ")


def test_speaker_prefix_stripped():
    # The classifier must look past "Speaker: " to the body.
    assert _is_conversational_filler("Caroline: Thanks!")
    assert _is_conversational_filler("Melanie: That's so cool!")
    assert not _is_conversational_filler(
        "Caroline: I went to the LGBTQ support group on 7 May 2023."
    )


def test_very_long_not_filler():
    # >250 chars is presumed informative — never filler.
    text = "Thanks " * 60  # ~420 chars of pure "Thanks Thanks Thanks ..."
    assert not _is_conversational_filler(text)
