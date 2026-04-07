# Changelog

All notable changes to Mnemoria will be documented in this file.

The format is based on Keep a Changelog.

## [0.1.0] - 2026-04-07

Initial public release.

### Added
- SQLite-backed cognitive memory engine with `MnemoriaStore` and `MnemoriaConfig`
- ACT-R-style activation scoring with typed metabolic decay
- MEMORY_SPEC typed facts (`C/D/V/?/✓/~`) with supersession support
- FTS5 keyword retrieval plus semantic embedding fallback/provider abstraction
- Hebbian semantic links, keyword links, and temporal adjacency links
- Scope lifecycle management and gauge pressure controls
- Q-value reranking, IPS debiasing, and session reward tracking
- Intent-aware retrieval hooks and exploration support
- Optional embedding backends via extras:
  - `mnemoria[embeddings]`
  - `mnemoria[ollama]`
  - `mnemoria[openai]`
  - `mnemoria[all]`

### Verified
- Local smoke and regression test suite passing (`7 passed`)
- Full hermes-agent benchmark run in TF-IDF/container environment reaching ~0.934 overall

### Notes
- In constrained containers, Mnemoria may fall back to TF-IDF instead of real embeddings.
- Semantic retrieval quality is expected to improve on a local machine with `mnemoria[embeddings]` installed.
