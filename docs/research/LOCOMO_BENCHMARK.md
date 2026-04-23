# LOCOMO Benchmark Research

Research date: 2026-04-23
Mnemoria version: v0.2.1 (0.927 internal benchmark)

## What is LOCOMO?

LOCOMO (Long-term COnversational MEmory) is the primary academic benchmark for evaluating long-term conversational memory in LLM agents. It was introduced by Maharana et al. in "Evaluating Very Long-Term Conversational Memory of LLM Agents" (arXiv:2402.17753). The benchmark was the basis for the Mem0 ECAI 2025 paper, which remains the most cited head-to-head comparison of AI memory approaches.

**Paper:** https://arxiv.org/abs/2402.17753
**Repository:** https://github.com/snap-research/locomo
**Project page:** https://snap-research.github.io/locomo/

## Dataset

- 10 multi-session conversations generated between LLM-architected virtual agents
- Seeded with multi-sentence personas and causal, temporally organized event graphs (up to 25 events over 6-12 months)
- Each conversation spans up to 32-35 sessions
- Average ~600 turns, ~16,000-26,000 tokens per conversation
- Images embedded in conversations (URLs + captions; images not publicly released)
- Total: 1,986 annotated question-answer pairs
- Data file: `locomo10.json` in the repository

### Dataset Structure (per conversation)

| Field | Description |
|-------|-------------|
| `sample_id` | Conversation identifier |
| `conversation` | Sessions with timestamps, speaker names, turns |
| `observation` | Session-level observations (generated, for RAG) |
| `session_summary` | Summaries for RAG databases |
| `event_summary` | Annotated significant events per speaker per session |
| `qa` | Question-answer pairs with categories and evidence dialog IDs |

## Question Categories

Five distinct reasoning types, designed to probe different aspects of memory:

| Category | Description | Difficulty |
|----------|-------------|------------|
| **Single-hop** | Direct factual recall from a single session | Baseline |
| **Multi-hop** | Chain two or more events/facts across sessions | Hard |
| **Temporal** | Ordinal/precedence questions ("When did X do Y?") | Very hard |
| **Open-domain** | Questions requiring commonsense/world knowledge combined with conversation facts | Moderate |
| **Adversarial** | Misleading/unanswerable questions designed to probe consistency | Very hard |

## Evaluation Metrics

The benchmark uses multiple complementary metrics:

| Metric | Description |
|--------|-------------|
| **F1 Score** | Harmonic mean of precision and recall over response tokens |
| **BLEU Score** | Token-level similarity between model response and ground truth |
| **LLM-as-Judge (J Score)** | Binary correctness (0 or 1) via LLM evaluating factual accuracy |
| **ROUGE** | Recall-oriented summary evaluation |
| **MMRelevance** | Multi-modal relevance (for image-integrated questions) |
| **Token consumption** | Total tokens required for the answer |
| **Latency** | Wall-clock time for search + response |

The LLM-as-Judge score has become the de facto primary metric in recent comparisons (Mem0, Zep, MemMachine all report this).

## Competitor Scores on LOCOMO

### Memory System Leaderboard (as of 2026-04)

| System | J Score (LLM Judge) | F1 | Notes |
|--------|--------------------:|---:|-------|
| **MemMachine v0.2** | **91.69%** | -- | gpt-4.1-mini; highest published score |
| **Zep (Graphiti)** | **75.14%** | -- | Contested; originally claimed 84%, corrected methodology |
| **Mem0** | **66.9-67.1%** | -- | ECAI 2025 paper; 1.4s latency, 91% fewer tokens vs full-context |
| **OpenAI Memory** | ~52.9% | -- | Built-in ChatGPT memory feature |
| **Full-context (no memory)** | varies | ~37-42 | GPT-3.5/4-Turbo baseline |
| **Human performance** | -- | ~88 | Upper bound; especially strong on temporal |

### Per-Category Performance (approximate, from published data)

| Category | Human F1 | Best LLM F1 | Gap |
|----------|----------|-------------|-----|
| Single-hop | ~85 | ~42 | Large |
| Multi-hop | ~88 | ~35 | Very large |
| Temporal | ~93 | ~20-30 | Enormous |
| Open-domain | ~82 | ~40 | Large |
| Adversarial | ~90 | ~2-10 | Extreme |

### Cognis System Scores (F1, for reference)

| Category | Cognis | vs Mem0 | vs Zep |
|----------|--------|---------|--------|
| Single-hop | 48.66 | +25.7% | -- |
| Multi-hop | 31.51 | +10.0% | -- |
| Open-domain | 54.77 | -- | +10.5% |
| Temporal | 62.68 | +21.6% (vs Mem0g) | -- |

### Important Caveat

LOCOMO benchmark scores are contested. Zep and Mem0 have publicly disputed each other's methodologies (see https://github.com/getzep/zep-papers/issues/5). Treat any single vendor-reported figure with appropriate skepticism. The most reliable comparisons come from independent evaluations using the same methodology.

## Repository Structure and Evaluation Scripts

```
snap-research/locomo/
  data/
    locomo10.json           # The 10-conversation dataset
  scripts/
    env.sh                  # Configuration
    evaluate_gpts.sh        # OpenAI model evaluation
    evaluate_claude.sh      # Anthropic model evaluation
    evaluate_gemini.sh      # Gemini model evaluation
    evaluate_hf_llm.sh      # HuggingFace model evaluation
    evaluate_rag_gpts.sh    # RAG-augmented evaluation
    generate_observations.sh
    generate_session_summaries.sh
  task_eval/                # Evaluation code
  prompt_examples/          # Prompt templates
  generative_agents/        # Agent implementations
  requirements.txt
```

There is also a modernized fork: https://github.com/playeriv65/EasyLocomo -- a streamlined refactor that supports evaluation of any LLM via OpenAI-compatible APIs.

## What's Needed to Run Mnemoria Against LOCOMO

### Integration Approach

LOCOMO evaluates memory-augmented response generation, not memory retrieval in isolation. To benchmark Mnemoria:

1. **Conversation ingestion**: Feed each conversation turn-by-turn into Mnemoria, calling `store()` for fact extraction (continuous extraction pipeline handles this naturally)

2. **Query handling**: For each QA pair, use `recall()` to retrieve relevant context, then feed retrieved memories + question to an LLM for answer generation

3. **Evaluation**: Compare generated answers against ground truth using LOCOMO's metrics (F1, BLEU, J Score)

### Adapter Requirements

```
mnemoria_locomo_adapter.py:
  - load_conversations(locomo10.json) -> iterate sessions
  - for each session, each turn:
      store.store(turn_text)   # or use observe_event() pipeline
  - for each QA pair:
      memories = store.recall(question, top_k=10)
      context = format_memories(memories)
      answer = llm_generate(question, context)
  - collect_scores(answers, ground_truth)
```

### Key Considerations

- **LLM dependency**: LOCOMO scores depend heavily on which LLM generates the final answer. Mnemoria's recall quality is only part of the score. Compare with the same LLM across systems.
- **Category alignment**: Mnemoria's internal benchmark categories (contradictions, supersession, temporal_decay, etc.) test retrieval quality. LOCOMO tests end-to-end answer quality. Both are valuable but measure different things.
- **Mnemoria strengths that may shine**: temporal_decay (0.933 internal), contradictions (0.950), cross_reference (0.956) -- these align with LOCOMO's multi-hop and temporal categories.
- **Mnemoria weaknesses to address first**: semantic_recall (0.800, capped by embedding model) and deduplication (0.750) could hurt LOCOMO scores.

### Estimated Effort

- Adapter script: 1-2 days
- Running full evaluation: ~2-4 hours (10 conversations, API calls)
- Analysis and scoring: 1 day
- Total: 3-4 days to first LOCOMO score

## LOCOMO-Plus (2026 Extension)

A newer extension called LoCoMo-Plus (arXiv:2602.10715) adds Level-2 Cognitive Memory evaluation beyond factual recall:

| Dimension | Description |
|-----------|-------------|
| Causal | Earlier causes affecting later events |
| State | Physical/emotional states influencing behavior |
| Goal | Long-term intentions shaping current choices |
| Value | Beliefs/values guiding reactions |

Key differences from LOCOMO v1:
- Removes explicit task-type disclosure (no "prompt bias")
- Replaces string-matching metrics with constraint-consistency evaluation
- Uses intentional semantic disconnect between cues and queries (no BM25/embedding shortcuts)
- Exposes failures undetected by LOCOMO v1

This is relevant for Mnemoria's future roadmap -- the cognitive architecture (ACT-R, Hebbian links, PPR exploration) is well-positioned for Level-2 cognitive tasks.
