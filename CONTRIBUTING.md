# Contributing to Mnemoria

Thank you for wanting to improve Mnemoria.

## Development setup

```bash
git clone <repo-url>
cd mnemoria
python -m pip install -e '.[dev]'
python -m pytest tests/ -q
```

## Project shape

Core package lives in:
- `mnemoria/`

Tests live in:
- `tests/`

Important modules:
- `store.py` — main public API and lifecycle wiring
- `retrieval.py` — ranking, fusion, dampening, reranking
- `links.py` — semantic, keyword, and temporal links
- `config.py` — tunables and profiles

## Contribution guidelines

- Prefer focused changes with tests.
- Do not overstate benchmark claims; report exact commands and observed results.
- Optimize for real memory usefulness, not benchmark-only heuristics.
- If changing retrieval behavior, run both targeted regressions and the full benchmark if feasible.

## Before opening a PR

- Run tests:
  - `python -m pytest tests/ -q`
- If you changed retrieval behavior, note benchmark impact and limitations honestly.
- Update docs when public API or installation changes.

## Reporting bugs

Please include:
- your install method
- Python version
- whether you are using TF-IDF fallback or real embeddings
- a minimal reproduction
- exact error output or incorrect retrieval example
