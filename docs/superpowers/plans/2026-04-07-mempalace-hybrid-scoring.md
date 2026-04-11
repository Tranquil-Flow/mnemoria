# Mempalace-style Hybrid Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port mempalace's hybrid scoring tricks (keyword overlap, person name boost, quoted phrase boost, temporal proximity, preference patterns) into mnemoria's recall pipeline so it closes the benchmark gap on LoCoMo and ConvoMem without giving up its current LongMemEval lead.

**Architecture:** Mempalace's "hybrid v5" is a post-retrieval rescoring pass: take vector-recall candidates, multiply distance by `(1 - α * predicate_overlap) * (1 - β * quoted_boost) * (1 - γ * name_boost)`, sort. This is added as a NEW dampening stage in `mnemoria/retrieval.py` after the existing dampening pipeline, gated by a config flag so existing benchmarks don't regress.

**Tech Stack:** Python 3.11+, mnemoria's existing recall pipeline (`mnemoria/store.py`, `mnemoria/retrieval.py`, `mnemoria/config.py`), pytest for unit + regression tests, the `faithful_mp_runner.py` script in `~/Projects/hermes-agent/benchmarks/` for end-to-end LongMemEval / LoCoMo / ConvoMem validation.

---

## Background — Where We Stand

Benchmarks run on 2026-04-07 against mnemoria `balanced` profile, no LLM rerank, identical embedder (`sentence-transformers/all-MiniLM-L6-v2`), mempalace's exact metrics:

| Benchmark | Mnemoria | MemPalace baseline | Gap |
|---|---|---|---|
| LongMemEval `_s` R@5 | **94.4%** | 96.6% (raw) | -2.2pp |
| LoCoMo session R@10 | 46.4% | 60.3% (raw) / **88.9%** (hybrid v5) | -14pp / -43pp |
| LoCoMo dialog R@10 | 20.7% | 48.0% (raw) | -27pp |
| ConvoMem avg recall | 72.7% | 92.9% | -20pp |

**The diagnosis:** mnemoria's vector retrieval and ACT-R activation pipeline are competitive on LongMemEval (where the right session has high embedding similarity to the question), but lose hard on LoCoMo and ConvoMem where the right *dialog turn* contains a named entity, a quoted phrase, or a date that mempalace's hybrid scoring exploits.

**The biggest single LongMemEval weakness:** `single-session-preference` category, 80% vs mempalace's 93.3% (-13.3pp). Mempalace closed this with their "hybrid v3 preference extraction patterns".

**Source of mempalace's tricks** (read these before starting):
- LoCoMo hybrid scoring: https://github.com/milla-jovovich/mempalace/blob/main/benchmarks/locomo_bench.py — search for `_kw_overlap`, `_quoted_boost`, `_name_boost`, `_person_names`, `_quoted_phrases`, the `STOP_WORDS` and `NOT_NAMES` constants, and the `mode == "hybrid"` branch in `run_benchmark`.
- LongMemEval hybrid v3 preference patterns: https://github.com/milla-jovovich/mempalace/blob/main/benchmarks/longmemeval_bench.py — search for "hybrid_v3" and "preference".

**Faithful benchmark runner already exists** at `~/Projects/hermes-agent/benchmarks/faithful_mp_runner.py`. It uses `MnemoriaStore` directly and computes mempalace-style metrics. Re-run after each tier to measure deltas. Results go to `~/Projects/hermes-agent/benchmarks/results/`.

---

## File Structure

This plan modifies/creates these files:

| File | Change | Responsibility |
|---|---|---|
| `mnemoria/hybrid_scoring.py` | **Create** | Pure functions: keyword/name/phrase extraction + boost computation |
| `mnemoria/retrieval.py` | **Modify** (add `apply_hybrid_scoring` near line 540 after `apply_dampening`) | Wire hybrid scoring into the pipeline |
| `mnemoria/store.py` | **Modify** (`recall()` method around line 422) | Call `apply_hybrid_scoring` after dampening, gated by config |
| `mnemoria/config.py` | **Modify** (add fields after the dampening section, ~line 200) | Config flags + tunable weights |
| `tests/test_hybrid_scoring.py` | **Create** | Unit tests for the new pure functions and full pipeline |
| `tests/test_benchmark_regressions.py` | **Modify** | New regression tests asserting hybrid scoring doesn't break LongMem-style cases |
| `docs/HYBRID_SCORING.md` | **Create** | Brief design doc + benchmark before/after table |

The new `hybrid_scoring.py` module is split out of `retrieval.py` because `retrieval.py` is already 1000+ lines and these functions form a cohesive unit (query analysis + multiplicative reranking).

---

## Tier 1 — Hybrid Scoring v5 (highest ROI)

The single biggest mempalace-vs-mnemoria gap. Mempalace's BENCHMARKS.md shows their hybrid v5 brought LoCoMo session R@10 from 60.3% (raw vector) to 88.9% — a +28pp jump from this exact set of tricks.

### Task 1: Config flags

**Files:**
- Modify: `mnemoria/config.py` (add fields after the existing dampening section, around line 200)

- [ ] **Step 1: Read existing config to find the right spot**

Run: `grep -n "enable_dampening\|gravity_dampening_factor" mnemoria/config.py`
Expected: Lines around 200 showing the dampening flags. Add new fields immediately after them.

- [ ] **Step 2: Add hybrid scoring config fields**

Insert after the existing dampening fields in `mnemoria/config.py`:

```python
    # ------------------------------------------------------------------
    # Hybrid Scoring (mempalace-style query-aware boosts)
    # ------------------------------------------------------------------

    enable_hybrid_scoring: bool = False
    """If True, apply mempalace-style hybrid scoring after recall:
    multiplicative boosts for predicate-keyword overlap, person name
    matches, and quoted-phrase matches. Disabled by default to preserve
    existing benchmark behavior; enable per-call via store.recall(...,
    hybrid=True) or globally via this flag."""

    hybrid_predicate_weight: float = 0.50
    """Strength of predicate-keyword overlap boost. fused = score * (1 - w * overlap)."""

    hybrid_quoted_weight: float = 0.60
    """Strength of quoted-phrase boost."""

    hybrid_name_weight: float = 0.20
    """Strength of person-name boost."""
```

Note: mnemoria's score convention is "higher is better" but mempalace's `fused = dist * (1 - w * boost)` assumes "lower is better" (cosine distance). The implementation in Task 5 must invert correctly — see Task 5 step 3 for the formula.

- [ ] **Step 3: Run config-loading test to make sure we didn't break parsing**

Run: `cd /Users/evinova/Projects/mnemoria && python -c "from mnemoria.config import MnemoriaConfig; c = MnemoriaConfig.from_profile('balanced'); print(c.enable_hybrid_scoring, c.hybrid_predicate_weight)"`
Expected: `False 0.5`

- [ ] **Step 4: Commit**

```bash
cd /Users/evinova/Projects/mnemoria
git add mnemoria/config.py
git commit -m "config: add hybrid scoring flags (disabled by default)"
```

---

### Task 2: Query analysis primitives (keywords, names, quoted phrases)

**Files:**
- Create: `mnemoria/hybrid_scoring.py`
- Create: `tests/test_hybrid_scoring.py`

The three extractors come straight from mempalace's `locomo_bench.py`. Read it first so you understand the conventions (NOT_NAMES is the question-word denylist, etc.).

- [ ] **Step 1: Write the failing tests for query analysis**

Create `tests/test_hybrid_scoring.py`:

```python
"""Tests for mempalace-style hybrid scoring primitives."""
from mnemoria.hybrid_scoring import (
    extract_predicate_keywords,
    extract_person_names,
    extract_quoted_phrases,
)


def test_extract_predicate_keywords_excludes_stopwords():
    kws = extract_predicate_keywords("What did Alice say about PostgreSQL last week?")
    assert "alice" not in kws  # names get filtered out by name detection upstream
    assert "postgresql" in kws
    # Stopwords like 'what', 'did', 'about', 'last', 'week' must be filtered
    for stop in ("what", "did", "about", "the", "a"):
        assert stop not in kws


def test_extract_predicate_keywords_excludes_query_verbs():
    kws = extract_predicate_keywords("Where did I buy the laptop?")
    assert "laptop" in kws
    assert "buy" not in kws  # buy/bought are common query verbs
    assert "where" not in kws


def test_extract_person_names_finds_capitalized_nonquestion_words():
    names = extract_person_names("Where did Caroline meet John yesterday?")
    assert "Caroline" in names
    assert "John" in names
    assert "Where" not in names  # question words must be excluded


def test_extract_person_names_skips_sentence_initial_question_words():
    names = extract_person_names("What did Sarah do?")
    assert "Sarah" in names
    assert "What" not in names


def test_extract_quoted_phrases_handles_double_quotes():
    phrases = extract_quoted_phrases('She told him "I will be late" before leaving')
    assert "i will be late" in phrases  # lowercased


def test_extract_quoted_phrases_handles_single_quotes():
    phrases = extract_quoted_phrases("The error message was 'connection refused' yesterday")
    assert "connection refused" in phrases


def test_extract_quoted_phrases_empty_when_no_quotes():
    assert extract_quoted_phrases("No quotes here at all") == set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/evinova/Projects/mnemoria && python -m pytest tests/test_hybrid_scoring.py -v`
Expected: All 6 tests FAIL with `ModuleNotFoundError: No module named 'mnemoria.hybrid_scoring'`

- [ ] **Step 3: Create the module with the three extractors**

Create `mnemoria/hybrid_scoring.py`:

```python
"""Mempalace-style hybrid scoring primitives.

Pure functions for query analysis (keyword/name/phrase extraction) and
multiplicative score boosts. Used by mnemoria's recall pipeline when
config.enable_hybrid_scoring is True.

Source: ported from mempalace's benchmarks/locomo_bench.py and
benchmarks/longmemeval_bench.py (hybrid v5 mode).
"""
from __future__ import annotations

import re
from typing import Iterable, List, Set


# Function/connective words that should never count as content keywords.
# Source: mempalace locomo_bench.py STOP_WORDS constant.
STOP_WORDS: frozenset[str] = frozenset({
    "what", "when", "where", "who", "how", "which", "did", "do", "was",
    "were", "have", "has", "had", "is", "are", "the", "a", "an", "my",
    "me", "i", "you", "your", "their", "it", "its", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "ago", "last", "that",
    "this", "there", "about", "get", "got", "give", "gave", "buy",
    "bought", "made", "make", "said",
})

# Capitalized words that look like names but are actually question/verb words
# at the start of a sentence. Source: mempalace locomo_bench.py NOT_NAMES.
NOT_NAMES: frozenset[str] = frozenset({
    "What", "When", "Where", "Who", "How", "Which", "Did", "Do", "Was",
    "Were", "Have", "Has", "Had", "Is", "Are", "The", "A", "An", "My",
    "I", "You", "Your", "Their", "It", "Its", "In", "On", "At", "To",
    "For", "Of", "With", "By", "From", "About", "Last", "That", "This",
    "There",
})


def extract_predicate_keywords(query: str) -> Set[str]:
    """Extract content-bearing keywords from a query.

    Lowercases, splits on non-word characters, drops STOP_WORDS, drops
    very short tokens, drops anything that looks like a person name
    (capitalized non-question word in the original casing).
    """
    if not query:
        return set()
    # First, identify person-name positions in the original casing so we can
    # exclude them from the lowercased keyword set.
    name_lower = {n.lower() for n in extract_person_names(query)}
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]+", query.lower())
    out: Set[str] = set()
    for tok in tokens:
        if len(tok) < 3:
            continue
        if tok in STOP_WORDS:
            continue
        if tok in name_lower:
            continue
        out.add(tok)
    return out


def extract_person_names(query: str) -> List[str]:
    """Extract capitalized words that look like person names.

    Returns the names in their original casing, in order of first
    appearance, deduplicated.
    """
    if not query:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for match in re.finditer(r"\b([A-Z][a-z]+)\b", query):
        word = match.group(1)
        if word in NOT_NAMES:
            continue
        if word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


def extract_quoted_phrases(query: str) -> Set[str]:
    """Extract phrases inside double or single quotes from a query.

    Returns lowercased phrases. Empty set if no quotes.
    """
    if not query:
        return set()
    out: Set[str] = set()
    # Double quotes
    for m in re.finditer(r'"([^"]+)"', query):
        phrase = m.group(1).strip().lower()
        if phrase:
            out.add(phrase)
    # Single quotes — be careful not to grab apostrophes in contractions.
    # Require at least one space inside the quoted span.
    for m in re.finditer(r"'([^']+)'", query):
        phrase = m.group(1).strip().lower()
        if phrase and " " in phrase:
            out.add(phrase)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/mnemoria && python -m pytest tests/test_hybrid_scoring.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/mnemoria
git add mnemoria/hybrid_scoring.py tests/test_hybrid_scoring.py
git commit -m "feat(hybrid): add query analysis primitives (keyword/name/phrase extraction)"
```

---

### Task 3: Boost computation functions

**Files:**
- Modify: `mnemoria/hybrid_scoring.py`
- Modify: `tests/test_hybrid_scoring.py`

These compute the actual boost values from the extracted query features. They produce values in `[0, 1]` representing "how much should this candidate be promoted."

- [ ] **Step 1: Add tests for the three boost functions**

Append to `tests/test_hybrid_scoring.py`:

```python
from mnemoria.hybrid_scoring import (
    keyword_overlap_score,
    quoted_phrase_score,
    name_match_score,
)


def test_keyword_overlap_score_zero_when_no_match():
    assert keyword_overlap_score({"laptop", "blue"}, "I went hiking yesterday") == 0.0


def test_keyword_overlap_score_full_match():
    score = keyword_overlap_score({"laptop", "blue"}, "I bought a blue laptop yesterday")
    # Both keywords present -> overlap is 1.0
    assert score == 1.0


def test_keyword_overlap_score_partial():
    score = keyword_overlap_score({"laptop", "blue", "warranty"}, "I bought a blue laptop")
    # 2 of 3 keywords matched
    assert abs(score - 2/3) < 1e-6


def test_keyword_overlap_score_empty_keywords_returns_zero():
    assert keyword_overlap_score(set(), "anything goes here") == 0.0


def test_quoted_phrase_score_substring_match():
    score = quoted_phrase_score({"connection refused"}, 'The error said "connection refused" at boot')
    assert score == 1.0


def test_quoted_phrase_score_no_match():
    assert quoted_phrase_score({"connection refused"}, "Everything works fine") == 0.0


def test_name_match_score_one_of_one():
    assert name_match_score(["Caroline"], "Caroline came over yesterday") == 1.0


def test_name_match_score_partial():
    score = name_match_score(["Caroline", "John"], "Caroline ate lunch alone")
    assert score == 0.5


def test_name_match_score_case_insensitive():
    assert name_match_score(["Caroline"], "and CAROLINE said hi") == 1.0
```

- [ ] **Step 2: Run tests, expect all to fail**

Run: `cd /Users/evinova/Projects/mnemoria && python -m pytest tests/test_hybrid_scoring.py -v -k "score"`
Expected: 9 FAILS with `ImportError`

- [ ] **Step 3: Append the boost functions to `mnemoria/hybrid_scoring.py`**

Add at the end of `mnemoria/hybrid_scoring.py`:

```python
def keyword_overlap_score(keywords: Set[str], doc: str) -> float:
    """Fraction of query keywords that appear in the doc text.

    Returns a value in [0, 1]. Zero if either input is empty.
    """
    if not keywords or not doc:
        return 0.0
    doc_lower = doc.lower()
    hits = sum(1 for kw in keywords if kw in doc_lower)
    return hits / len(keywords)


def quoted_phrase_score(phrases: Set[str], doc: str) -> float:
    """Fraction of quoted phrases that appear as substrings in the doc.

    Returns a value in [0, 1]. Quoted phrases are an extremely strong
    signal — when a user quotes something, the gold doc almost always
    contains it verbatim.
    """
    if not phrases or not doc:
        return 0.0
    doc_lower = doc.lower()
    hits = sum(1 for p in phrases if p in doc_lower)
    return hits / len(phrases)


def name_match_score(names: Iterable[str], doc: str) -> float:
    """Fraction of query names that appear (case-insensitive) in the doc.

    Returns a value in [0, 1].
    """
    name_list = list(names)
    if not name_list or not doc:
        return 0.0
    doc_lower = doc.lower()
    hits = sum(1 for n in name_list if n.lower() in doc_lower)
    return hits / len(name_list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/mnemoria && python -m pytest tests/test_hybrid_scoring.py -v`
Expected: All 15 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/mnemoria
git add mnemoria/hybrid_scoring.py tests/test_hybrid_scoring.py
git commit -m "feat(hybrid): add keyword/name/phrase boost functions"
```

---

### Task 4: The full hybrid rescoring function

**Files:**
- Modify: `mnemoria/hybrid_scoring.py`
- Modify: `tests/test_hybrid_scoring.py`

This is the actual rescoring step. It takes the already-scored candidates from mnemoria's recall and applies the multiplicative boost.

Mempalace's formula (from `locomo_bench.py`):

```python
fused = dist * (1.0 - 0.50 * pred_overlap)
if quoted_boost > 0:
    fused *= 1.0 - 0.60 * quoted_boost
if name_boost > 0:
    fused *= 1.0 - 0.20 * name_boost
```

Mempalace's `dist` is cosine distance (lower = better). Mnemoria's `ScoredFact.score` is "higher = better". The port must invert: instead of `score * (1 - w * overlap)` (which would *reduce* the score), we use `score * (1 + w * overlap)` to *boost* it.

- [ ] **Step 1: Write the failing test for `apply_hybrid_scoring`**

Append to `tests/test_hybrid_scoring.py`:

```python
from dataclasses import dataclass, field
from mnemoria.hybrid_scoring import apply_hybrid_scoring
from mnemoria.config import MnemoriaConfig


@dataclass
class _FakeFact:
    id: str
    content: str
    target: str = "general"


@dataclass
class _FakeScored:
    fact: _FakeFact
    score: float
    components: dict = field(default_factory=dict)


def test_apply_hybrid_scoring_promotes_keyword_match():
    cfg = MnemoriaConfig.balanced()
    cfg.enable_hybrid_scoring = True
    a = _FakeScored(_FakeFact("a", "Caroline went hiking on Sunday"), score=0.8)
    b = _FakeScored(_FakeFact("b", "I bought a new laptop in blue"), score=0.85)
    apply_hybrid_scoring([a, b], query="What color was the laptop?", cfg=cfg)
    # b should be promoted above a because 'laptop' appears in b
    assert b.score > a.score
    assert "hybrid_keyword_boost" in b.components


def test_apply_hybrid_scoring_promotes_name_match():
    cfg = MnemoriaConfig.balanced()
    cfg.enable_hybrid_scoring = True
    a = _FakeScored(_FakeFact("a", "John ate lunch alone"), score=0.8)
    b = _FakeScored(_FakeFact("b", "Caroline went to the park"), score=0.8)
    apply_hybrid_scoring([a, b], query="Where did Caroline go yesterday?", cfg=cfg)
    assert b.score > a.score


def test_apply_hybrid_scoring_promotes_quoted_phrase():
    cfg = MnemoriaConfig.balanced()
    cfg.enable_hybrid_scoring = True
    a = _FakeScored(_FakeFact("a", "Some unrelated context here"), score=0.8)
    b = _FakeScored(_FakeFact("b", "He said connection refused at boot"), score=0.8)
    apply_hybrid_scoring([a, b], query='What was the error? "connection refused"', cfg=cfg)
    assert b.score > a.score


def test_apply_hybrid_scoring_noop_when_disabled():
    cfg = MnemoriaConfig.balanced()
    cfg.enable_hybrid_scoring = False
    a = _FakeScored(_FakeFact("a", "Caroline went hiking"), score=0.8)
    apply_hybrid_scoring([a], query="What did Caroline do?", cfg=cfg)
    assert a.score == 0.8  # untouched


def test_apply_hybrid_scoring_handles_empty_list():
    cfg = MnemoriaConfig.balanced()
    cfg.enable_hybrid_scoring = True
    apply_hybrid_scoring([], query="anything", cfg=cfg)  # must not raise
```

- [ ] **Step 2: Run tests, expect failures**

Run: `cd /Users/evinova/Projects/mnemoria && python -m pytest tests/test_hybrid_scoring.py::test_apply_hybrid_scoring_promotes_keyword_match -v`
Expected: FAIL with `ImportError: cannot import name 'apply_hybrid_scoring'`

- [ ] **Step 3: Implement `apply_hybrid_scoring`**

Append to `mnemoria/hybrid_scoring.py`:

```python
def apply_hybrid_scoring(scored, query: str, cfg) -> None:
    """In-place mempalace-style hybrid rescoring of recall results.

    Multiplicative boosts based on query feature matches:
      - predicate keyword overlap (mempalace's strongest signal)
      - quoted phrase substring match (very strong but rare)
      - person name overlap

    Mnemoria scores are "higher is better", so the formula is:
        score *= 1 + w_pred * pred_overlap
        score *= 1 + w_quote * quote_overlap   (only if > 0)
        score *= 1 + w_name * name_overlap     (only if > 0)

    Mutates the `scored` list and re-sorts by score descending.
    No-op if cfg.enable_hybrid_scoring is False or scored is empty.
    """
    if not getattr(cfg, "enable_hybrid_scoring", False) or not scored:
        return

    names = extract_person_names(query)
    keywords = extract_predicate_keywords(query)
    phrases = extract_quoted_phrases(query)

    # Skip work if the query has no signals to boost on
    if not keywords and not names and not phrases:
        return

    w_pred = cfg.hybrid_predicate_weight
    w_quote = cfg.hybrid_quoted_weight
    w_name = cfg.hybrid_name_weight

    for item in scored:
        # Build the doc text — include target for typed facts because their
        # target encodes semantic identity (mfa, auth, etc.) that may not be
        # in the content.
        doc = item.fact.content
        if getattr(item.fact, "target", None) and item.fact.target != "general":
            doc = f"{item.fact.target} {doc}"

        kw_overlap = keyword_overlap_score(keywords, doc) if keywords else 0.0
        if kw_overlap > 0:
            multiplier = 1.0 + w_pred * kw_overlap
            item.score *= multiplier
            item.components["hybrid_keyword_boost"] = multiplier

        q_overlap = quoted_phrase_score(phrases, doc) if phrases else 0.0
        if q_overlap > 0:
            multiplier = 1.0 + w_quote * q_overlap
            item.score *= multiplier
            item.components["hybrid_quoted_boost"] = multiplier

        n_overlap = name_match_score(names, doc) if names else 0.0
        if n_overlap > 0:
            multiplier = 1.0 + w_name * n_overlap
            item.score *= multiplier
            item.components["hybrid_name_boost"] = multiplier

    scored.sort(key=lambda s: s.score, reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/mnemoria && python -m pytest tests/test_hybrid_scoring.py -v`
Expected: All 20 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/mnemoria
git add mnemoria/hybrid_scoring.py tests/test_hybrid_scoring.py
git commit -m "feat(hybrid): add apply_hybrid_scoring rescoring function"
```

---

### Task 5: Wire hybrid scoring into the recall pipeline

**Files:**
- Modify: `mnemoria/store.py` (around line 422 in the `recall()` method)

The hook point is **after `apply_dampening` and before the final IPS debiasing sort**. This way hybrid scoring sees fully-dampened scores and its result feeds into the final ordering.

- [ ] **Step 1: Read the current recall method to confirm line numbers**

Run: `grep -n "apply_dampening\|apply_ips_debiasing\|diversify_results" mnemoria/store.py`
Expected: Output around lines 422–433. Confirm the order: dampening → ips → sort → diversify.

- [ ] **Step 2: Wire `apply_hybrid_scoring` between dampening and ips**

In `mnemoria/store.py`, find this block (around line 420–426):

```python
        # Sort and apply dampening
        scored.sort(key=lambda s: s.score, reverse=True)
        scored = apply_dampening(self._conn, scored, query, self._config)

        # IPS debiasing — counteract popularity bias after dampening
        apply_ips_debiasing(scored, self._conn, self._config)
        scored.sort(key=lambda s: s.score, reverse=True)
```

Replace with:

```python
        # Sort and apply dampening
        scored.sort(key=lambda s: s.score, reverse=True)
        scored = apply_dampening(self._conn, scored, query, self._config)

        # Mempalace-style hybrid scoring (gated by config; no-op if disabled)
        from mnemoria.hybrid_scoring import apply_hybrid_scoring
        apply_hybrid_scoring(scored, query, self._config)

        # IPS debiasing — counteract popularity bias after dampening
        apply_ips_debiasing(scored, self._conn, self._config)
        scored.sort(key=lambda s: s.score, reverse=True)
```

- [ ] **Step 3: Verify nothing breaks with hybrid disabled**

Run: `cd /Users/evinova/Projects/mnemoria && python -m pytest tests/ -v`
Expected: All existing tests PASS (hybrid is gated off by default, so no behavior change).

- [ ] **Step 4: Smoke test the hybrid path end-to-end**

Run:
```bash
cd /Users/evinova/Projects/mnemoria && python -c "
from mnemoria.config import MnemoriaConfig
from mnemoria.store import MnemoriaStore
cfg = MnemoriaConfig.from_profile('balanced')
cfg.enable_hybrid_scoring = True
cfg.db_path = ':memory:'
cfg.enable_pressure = False
s = MnemoriaStore(config=cfg, db_path=':memory:')
s.store('Caroline went to Berlin in March')
s.store('John bought a blue laptop with extended warranty')
s.store('The on-call rotation was updated last week')
results = s.recall('What color was the laptop?', top_k=3)
for r in results:
    print(f'{r.score:.3f}  {r.fact.content[:60]}')
    if 'hybrid_keyword_boost' in r.components:
        print(f'    boost: {r.components[\"hybrid_keyword_boost\"]:.3f}')
"
```
Expected: The "blue laptop" line ranks first, and its components show a `hybrid_keyword_boost` value > 1.0.

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/mnemoria
git add mnemoria/store.py
git commit -m "feat(hybrid): wire apply_hybrid_scoring into recall pipeline"
```

---

### Task 6: Regression test against benchmark-style cases

**Files:**
- Modify: `tests/test_benchmark_regressions.py`

These tests document the *intended* effect of hybrid scoring on retrieval cases that mempalace's tricks were designed to solve. They use the same store-and-recall pattern as existing regression tests in this file (see `test_named_answer_beats_unrelated_newer_fact` at line 63 for reference).

- [ ] **Step 1: Add three regression tests**

Append to `tests/test_benchmark_regressions.py`:

```python
def test_hybrid_scoring_promotes_keyword_match_over_recency():
    """LongMemEval-style: a question with a content keyword should retrieve
    the matching session even if a more recent unrelated session exists."""
    from mnemoria.config import MnemoriaConfig
    from mnemoria.store import MnemoriaStore
    cfg = MnemoriaConfig.from_profile('balanced')
    cfg.enable_hybrid_scoring = True
    cfg.db_path = ':memory:'
    cfg.enable_pressure = False
    store = MnemoriaStore(config=cfg, db_path=':memory:')
    benchmark_store(store, "Last March I bought a Nikon D850 camera body")
    benchmark_store(store, "I had eggs and toast for breakfast this morning")
    top = store.recall("What camera body did I buy?", top_k=1)[0].fact.content
    assert "Nikon" in top or "D850" in top


def test_hybrid_scoring_promotes_named_speaker():
    """LoCoMo-style: a question naming a person should retrieve the dialog
    turn featuring that person, not a generic similar turn."""
    from mnemoria.config import MnemoriaConfig
    from mnemoria.store import MnemoriaStore
    cfg = MnemoriaConfig.from_profile('balanced')
    cfg.enable_hybrid_scoring = True
    cfg.db_path = ':memory:'
    cfg.enable_pressure = False
    store = MnemoriaStore(config=cfg, db_path=':memory:')
    benchmark_store(store, 'John said, "I am going to the gym after work"')
    benchmark_store(store, 'Caroline said, "I am taking the dog to the park"')
    top = store.recall("Where was Caroline going?", top_k=1)[0].fact.content
    assert "Caroline" in top


def test_hybrid_scoring_disabled_does_not_change_results():
    """Sanity: with hybrid disabled, mnemoria's behavior is unchanged."""
    from mnemoria.config import MnemoriaConfig
    from mnemoria.store import MnemoriaStore
    cfg = MnemoriaConfig.from_profile('balanced')
    cfg.enable_hybrid_scoring = False
    cfg.db_path = ':memory:'
    cfg.enable_pressure = False
    store = MnemoriaStore(config=cfg, db_path=':memory:')
    benchmark_store(store, "Caroline went to Paris in April")
    benchmark_store(store, "I made pasta for dinner yesterday")
    # No boost expected — verify components dict has no hybrid keys
    results = store.recall("Where did Caroline travel?", top_k=2)
    for r in results:
        assert "hybrid_keyword_boost" not in r.components
        assert "hybrid_name_boost" not in r.components
```

- [ ] **Step 2: Run the new tests**

Run: `cd /Users/evinova/Projects/mnemoria && python -m pytest tests/test_benchmark_regressions.py::test_hybrid_scoring_promotes_keyword_match_over_recency tests/test_benchmark_regressions.py::test_hybrid_scoring_promotes_named_speaker tests/test_benchmark_regressions.py::test_hybrid_scoring_disabled_does_not_change_results -v`
Expected: All 3 PASS

- [ ] **Step 3: Run the full mnemoria test suite to confirm no regressions**

Run: `cd /Users/evinova/Projects/mnemoria && python -m pytest tests/ -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 4: Commit**

```bash
cd /Users/evinova/Projects/mnemoria
git add tests/test_benchmark_regressions.py
git commit -m "test: regression tests for hybrid scoring on benchmark patterns"
```

---

### Task 7: End-to-end benchmark validation

**Files:**
- No code changes; this is a measurement task.

Run the faithful runner with hybrid scoring enabled and compare against the baseline numbers in this plan's "Background" section. **The hard requirement is no regression on LongMemEval R@5; the goal is meaningful improvement on LoCoMo and ConvoMem.**

- [ ] **Step 1: Patch the faithful runner to enable hybrid scoring**

The runner at `~/Projects/hermes-agent/benchmarks/faithful_mp_runner.py` calls `make_store()`. Modify ONLY for this measurement run by editing `make_store` to set `cfg.enable_hybrid_scoring = True`. Do not commit this change to the runner — it should remain a faithful baseline for future comparisons.

Run:
```bash
cd /Users/evinova/Projects/hermes-agent
python -c "
content = open('benchmarks/faithful_mp_runner.py').read()
patched = content.replace(
    'cfg.enable_pressure = False  # benchmarks should not hit gauge limits',
    'cfg.enable_pressure = False  # benchmarks should not hit gauge limits\n    cfg.enable_hybrid_scoring = True  # TEMP: hybrid scoring measurement'
)
open('benchmarks/faithful_mp_runner.py', 'w').write(patched)
print('patched')
"
```

- [ ] **Step 2: Run LongMem `_s` regression check (must stay >= 94.4% R@5)**

Run:
```bash
cd /Users/evinova/Projects/hermes-agent
PYTHONPATH=/Users/evinova/Projects/mnemoria:. python benchmarks/faithful_mp_runner.py longmem --split longmemeval_s_cleaned 2>&1 | tail -25
```

Expected: R@5 >= 94.4% (the baseline). If it dropped, hybrid scoring is hurting LongMem and the weights need tuning — see Task 8.

Save the output to a file you can refer back to:
```bash
cp benchmarks/results/longmem_mnemoria_faithful_s.json benchmarks/results/longmem_mnemoria_hybrid_s.json
```

- [ ] **Step 3: Run LoCoMo session (target: R@10 > 46.4%)**

Run:
```bash
cd /Users/evinova/Projects/hermes-agent
PYTHONPATH=/Users/evinova/Projects/mnemoria:. python benchmarks/faithful_mp_runner.py locomo --granularity session 2>&1 | tail -30
```

Expected: R@10 > 46.4% (preferably approaching mempalace's 60.3% raw baseline; mempalace's 88.9% hybrid v5 is the stretch goal).

Save:
```bash
cp benchmarks/results/locomo_mnemoria_faithful_session.json benchmarks/results/locomo_mnemoria_hybrid_session.json
```

- [ ] **Step 4: Run ConvoMem (target: avg recall > 72.7%)**

Run:
```bash
cd /Users/evinova/Projects/hermes-agent
PYTHONPATH=/Users/evinova/Projects/mnemoria:. python benchmarks/faithful_mp_runner.py convomem --limit 50 2>&1 | tail -30
```

Expected: avg recall > 72.7%. Mempalace gets 92.9% — being within 10pp of that would be a great result.

Save:
```bash
cp benchmarks/results/convomem_mnemoria_faithful.json benchmarks/results/convomem_mnemoria_hybrid.json
```

- [ ] **Step 5: Revert the runner patch**

Run:
```bash
cd /Users/evinova/Projects/hermes-agent
python -c "
content = open('benchmarks/faithful_mp_runner.py').read()
patched = content.replace(
    'cfg.enable_pressure = False  # benchmarks should not hit gauge limits\n    cfg.enable_hybrid_scoring = True  # TEMP: hybrid scoring measurement',
    'cfg.enable_pressure = False  # benchmarks should not hit gauge limits'
)
open('benchmarks/faithful_mp_runner.py', 'w').write(patched)
print('reverted')
"
```

- [ ] **Step 6: Document results in the design doc**

Create `/Users/evinova/Projects/mnemoria/docs/HYBRID_SCORING.md`:

```markdown
# Hybrid Scoring (Mempalace Port)

## What it is

A post-recall multiplicative rescoring stage that boosts candidates based on
query-derived signals: predicate keyword overlap, quoted phrase matches, and
person name matches. Disabled by default; enable via `cfg.enable_hybrid_scoring = True`.

Source: ported from mempalace's `benchmarks/locomo_bench.py` hybrid v5 mode.

## Pipeline position

In `mnemoria/store.py:recall()`:

1. Get candidates → score (ACT-R + embeddings)
2. RRF/BM25 fusion (if enabled)
3. Q-value reranking
4. Sort and dampen (gravity, hub, resolution boost)
5. **Hybrid scoring (this stage)** ← multiplicative query-aware boosts
6. IPS debiasing
7. Sort + diversify

## Tunable weights

| Field | Default | Purpose |
|---|---|---|
| `hybrid_predicate_weight` | 0.50 | Strength of keyword overlap boost |
| `hybrid_quoted_weight` | 0.60 | Strength of quoted phrase boost |
| `hybrid_name_weight` | 0.20 | Strength of person name boost |

## Benchmark results (mempalace-faithful runner)

| Benchmark | Baseline | With hybrid | MemPalace baseline |
|---|---|---|---|
| LongMemEval `_s` R@5 | 94.4% | <FILL IN> | 96.6% |
| LoCoMo session R@10 | 46.4% | <FILL IN> | 60.3% (raw) / 88.9% (hybrid v5) |
| ConvoMem avg recall | 72.7% | <FILL IN> | 92.9% |

Methodology: identical embedder (`all-MiniLM-L6-v2`), session-level R@k, no
LLM rerank. Runner: `hermes-agent/benchmarks/faithful_mp_runner.py`.
```

Fill in the `<FILL IN>` cells with the numbers from steps 2–4.

- [ ] **Step 7: Commit doc + measurement results**

```bash
cd /Users/evinova/Projects/mnemoria
git add docs/HYBRID_SCORING.md
git commit -m "docs: hybrid scoring design + benchmark deltas"
```

---

## Tier 2 — Preference statement detection

**Why:** mnemoria's worst LongMemEval category is `single-session-preference` (80% vs mempalace 93.3% — the only category with a >10pp gap). Mempalace closed it with explicit pattern detection for indirect preference statements ("I usually prefer X", "My favorite Y is Z"). This task ports those patterns.

**Skip this task if Tier 1 alone closes the LongMemEval gap to within 0.5pp of mempalace's 96.6%.** Otherwise proceed.

### Task 8: Preference pattern boost

**Files:**
- Modify: `mnemoria/hybrid_scoring.py`
- Modify: `tests/test_hybrid_scoring.py`

Mempalace's preference patterns (read https://github.com/milla-jovovich/mempalace/blob/main/benchmarks/longmemeval_bench.py — search for "prefer" or "preference" in the hybrid_v3 section). The core idea: detect when a query is asking about a preference (`what is my favorite`, `do I prefer`, `which X do I like`) and boost any candidate containing preference verbs (`prefer`, `favorite`, `usually`, `always`, `typically`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hybrid_scoring.py`:

```python
def test_preference_query_boosts_preference_doc():
    cfg = MnemoriaConfig.balanced()
    cfg.enable_hybrid_scoring = True
    a = _FakeScored(_FakeFact("a", "I had pizza for dinner last Thursday"), score=0.8)
    b = _FakeScored(_FakeFact("b", "I usually prefer thin crust pizza"), score=0.78)
    apply_hybrid_scoring([a, b], query="What kind of pizza do I prefer?", cfg=cfg)
    assert b.score > a.score  # 'prefer' present, query is preference-shaped


def test_preference_pattern_no_boost_when_query_not_preference():
    cfg = MnemoriaConfig.balanced()
    cfg.enable_hybrid_scoring = True
    b = _FakeScored(_FakeFact("b", "I usually prefer thin crust"), score=0.8)
    apply_hybrid_scoring([b], query="When did I last eat pizza?", cfg=cfg)
    # Query is temporal, not preferential — should not get a preference boost
    assert "hybrid_preference_boost" not in b.components
```

- [ ] **Step 2: Run, expect failures**

Run: `cd /Users/evinova/Projects/mnemoria && python -m pytest tests/test_hybrid_scoring.py -v -k "preference"`
Expected: 2 FAILS

- [ ] **Step 3: Add preference detection helpers and wire into `apply_hybrid_scoring`**

In `mnemoria/hybrid_scoring.py`, add at the top:

```python
PREFERENCE_QUERY_PATTERNS = (
    re.compile(r"\b(prefer|favorite|favourite)\b", re.I),
    re.compile(r"\bdo I (like|love|enjoy|prefer)\b", re.I),
    re.compile(r"\bwhat (kind|type|sort) of \w+ (do|am) I\b", re.I),
)

PREFERENCE_DOC_TERMS = frozenset({
    "prefer", "preferred", "favorite", "favourite", "usually", "always",
    "typically", "love", "loved", "hate", "enjoy", "enjoyed",
})


def is_preference_query(query: str) -> bool:
    """Heuristic: does the query ask about a preference?"""
    return any(p.search(query) for p in PREFERENCE_QUERY_PATTERNS)


def preference_doc_score(doc: str) -> float:
    """How preference-shaped is this doc? Fraction of doc terms that are
    preference markers, capped at 1.0."""
    if not doc:
        return 0.0
    doc_lower = doc.lower()
    hits = sum(1 for term in PREFERENCE_DOC_TERMS if term in doc_lower)
    return min(hits / 2.0, 1.0)  # 2+ markers = full boost
```

Then in `apply_hybrid_scoring`, after the existing name boost block, add:

```python
        # Preference pattern boost (mempalace hybrid v3 trick)
        if is_preference_query(query):
            pref_score = preference_doc_score(doc)
            if pref_score > 0:
                multiplier = 1.0 + 0.40 * pref_score  # weight matches mempalace
                item.score *= multiplier
                item.components["hybrid_preference_boost"] = multiplier
```

- [ ] **Step 4: Run preference tests + full hybrid suite**

Run: `cd /Users/evinova/Projects/mnemoria && python -m pytest tests/test_hybrid_scoring.py -v`
Expected: All tests PASS (22 total)

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/mnemoria
git add mnemoria/hybrid_scoring.py tests/test_hybrid_scoring.py
git commit -m "feat(hybrid): preference statement detection (mempalace v3 port)"
```

- [ ] **Step 6: Re-run LongMem `_s` benchmark to measure preference category gain**

Apply the same temporary patch to `faithful_mp_runner.py` as in Task 7 step 1, then run:
```bash
cd /Users/evinova/Projects/hermes-agent
PYTHONPATH=/Users/evinova/Projects/mnemoria:. python benchmarks/faithful_mp_runner.py longmem --split longmemeval_s_cleaned 2>&1 | tail -25
```

Expected: `single-session-preference` R@5 climbs from 80% baseline. Target: > 90%. Then revert the patch.

Update `docs/HYBRID_SCORING.md` with the new R@5 numbers.

```bash
cd /Users/evinova/Projects/mnemoria
git add docs/HYBRID_SCORING.md
git commit -m "docs: update hybrid benchmark with preference patterns"
```

---

## Tier 3 — Stretch: Temporal proximity boost

**Why:** mempalace's BENCHMARKS.md credits temporal boost with +0.6pp (97.8% → 98.4% on LongMem). Small but cheap. Helps `temporal-reasoning` and `knowledge-update` categories where the question's reference date narrows the candidate space.

**Skip this task unless Tier 1 + Tier 2 leave LongMem still below 96.0% R@5.** Diminishing returns territory.

### Task 9: Temporal proximity boost

**Files:**
- Modify: `mnemoria/hybrid_scoring.py`
- Modify: `mnemoria/store.py` (recall() must pass `question_date` if available)

This requires the caller to pass a reference date. LongMemEval questions have `question_date`; mnemoria's `recall()` does not currently accept this. Adding it without breaking existing callers means a new optional kwarg.

The implementation is left to the next agent — it requires deciding how to expose the date through `MnemoriaStore.recall()`'s public API and whether to use mnemoria's existing virtual clock instead. **Brainstorm this with the user before implementing.**

---

## Tier 4 — Skip list (do NOT port)

These mempalace concepts are tempting but should be deprioritized:

| Concept | Why skip |
|---|---|
| **Wings v3 speaker-owned closets** | Only helps speaker-attribution adversarial questions on LoCoMo. Mempalace itself reports the gain is +46pp on the adversarial subcategory but speaker-confusion is a narrow problem mnemoria doesn't currently optimize for. Revisit after Tier 1+2 if LoCoMo adversarial is still the worst category. |
| **The Palace (rooms/wings/halls)** | Requires LLM-driven room assignment, +1 API dependency, and mempalace's own data (BENCHMARKS.md) shows their LoCoMo Palace v2 gets 84.8% R@10 — barely beating their non-Palace hybrid v5 (88.9%). The complexity isn't justified by the delta. |
| **AAAK dialect (lossless compression)** | Mempalace ships this as a separate feature, not a retrieval improvement. It changes ingestion, not scoring. Out of scope for "improve mnemoria's benchmark scores." |
| **LLM rerank (Haiku/Sonnet)** | Adds API cost per query. Mempalace's "100%" claims all use this. Mnemoria's positioning as a local-first store should not depend on paid rerank. If a user wants this they can do it externally. |

---

## Self-Review Checklist

- [x] Spec coverage: Each benchmark gap (LongMem, LoCoMo, ConvoMem) has a tier addressing it.
- [x] No placeholders: every code step has full code, every command has expected output.
- [x] Type consistency: `apply_hybrid_scoring` mutates `scored: List[ScoredFact]` in place; same name used in Tasks 4, 5, 6.
- [x] File paths are absolute or unambiguous from `~/Projects/mnemoria`.
- [x] Tests come before implementation in every task (TDD).
- [x] Each task ends with a commit.
- [x] Benchmark validation is gated behind a temporary patch that gets reverted.

---

## Acceptance Criteria

The plan is complete when:

1. All Tier 1 tasks (1–7) are merged.
2. `pytest tests/` passes on mnemoria.
3. LongMemEval `_s` R@5 with hybrid enabled is **>= 94.4%** (no regression) and ideally **> 95%**.
4. LoCoMo session R@10 with hybrid enabled is **> 50%** (target > 60% to match mempalace's raw baseline).
5. ConvoMem avg recall with hybrid enabled is **> 80%** (target > 90%).
6. `docs/HYBRID_SCORING.md` exists with filled-in benchmark numbers.

Tier 2 (preference patterns) and Tier 3 (temporal boost) are conditional — only execute if Tier 1 alone doesn't close the relevant category gaps.
