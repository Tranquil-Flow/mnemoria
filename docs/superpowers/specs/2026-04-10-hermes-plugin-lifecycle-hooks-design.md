# Mnemoria Hermes Plugin: Full Lifecycle Integration

**Date:** 2026-04-10
**Status:** Approved design, ready for implementation planning

## Goal

Transform the Mnemoria hermes-agent plugin from a passive store/recall endpoint into a full lifecycle participant that hooks into compression, delegation, built-in memory mirroring, and session management. This makes Mnemoria the first hermes-agent memory plugin that actively participates in the entire agent lifecycle.

## Architecture Overview

```
Session start          -> initialize()          context-aware setup, profile scoping
                       -> system_prompt_block() inject identity facts + usage hint (cached)
                       -> get_config_schema()   interactive setup wizard support
                       -> save_config()         persist config to mnemoria.json

Each turn              -> prefetch()            return recalled facts (from cache if available)
                       -> queue_prefetch()      background pre-warm for next turn

During tool execution  -> on_memory_write()     mirror built-in MEMORY.md/USER.md writes

Context compression    -> on_pre_compress()     extract facts before messages are discarded

Session end            -> on_session_end()      full conversation extraction + consolidation
                       -> on_delegation()       store delegation outcomes + extract from results
```

**sync_turn remains a no-op.** It only receives clean user text and final assistant text (no tool calls/results). The valuable extraction happens in on_pre_compress and on_session_end where we get full message history.

## File Structure

```
plugins/memory/mnemoria/
  __init__.py       register(ctx) entry point (already fixed)
  plugin.yaml       minimal metadata (already fixed)
  provider.py       MnemoriaMemoryProvider with all hook implementations
  extract.py        NEW — pattern matching extraction logic
  README.md         setup and usage docs (already created)
```

**Why split provider.py and extract.py:** Extraction logic is reused across on_pre_compress, on_session_end, and on_delegation. Having it in its own module makes it independently testable without mocking the full MemoryProvider. Keeps provider.py focused on hermes-agent integration at ~400-500 lines.

## Extraction Strategy

Rule-based pattern matching for high-signal tool outputs. No summary fallback. No LLM extraction in v1.

**Rationale:** Mnemoria's value proposition is structured, typed facts with different decay rates. Dumping vague summaries into it makes it a worse version of the built-in compressor. Better to store 5 correct typed facts than 1 vague summary. The architecture is designed so LLM-based extraction (v0.2.0 agent_inference mode) can be added later without changing the hook wiring.

### Pattern Matchers (initial set)

| Pattern | Source | Fact Type | Example |
|---------|--------|-----------|---------|
| Error/exception/traceback in tool results | tool role messages | `?[error]: {tool}: {summary}` | `?[error]: pytest: 3 tests failed in test_auth.py` |
| URLs in error context (within 3 lines of error) | tool role messages | `V[url]: {url}` | `V[url]: https://api.example.com:3005` |
| File paths in error context (within 3 lines of error) | tool role messages | `V[file]: {path}` | `V[file]: /src/auth/jwt.py` |
| URLs/paths in user messages | user role messages | `V[url]` or `V[file]` | `V[url]: https://docs.example.com` |
| User directive statements | user role messages | `C[user.pref]: {statement}` | `C[user.pref]: always use TypeScript for new code` |

**Noise control:**
- URLs and file paths are only extracted from tool results when they appear near error/exception/traceback patterns (within 3 lines). This prevents flooding the store from routine grep/cat output.
- URLs and file paths in user messages are always extracted (the user explicitly shared them).
- User preference patterns are restricted to imperative directives: "always use", "never use", "prefer to use", "don't use", "don't ever", "always make sure". General uses of "don't" or "never" in conversational context are not matched. Confidence is set to 0.6 (lower than tool-observed facts at 0.9) for future filtering.

**Dedup safety:** Mnemoria's exact dedup (source_hash) prevents storing the same extracted content twice even if hooks overlap. The message index tracking is the primary dedup mechanism; source_hash is the safety net.

### Message Tracking

Track `self._last_extracted_msg_index` to avoid re-processing messages across multiple compression events and session end. Each extraction pass only processes new messages since the last index. on_session_end only extracts from messages not already processed by on_pre_compress.

### extract.py Interface

```python
@dataclass
class ExtractedFact:
    content: str       # MEMORY_SPEC notation, e.g. "V[url]: https://..."
    source: str        # "tool_result", "user_statement"
    confidence: float  # 0.0-1.0, for future filtering

def extract_from_messages(
    messages: list[dict],
    start_index: int = 0,
) -> tuple[list[ExtractedFact], int]:
    """Extract facts from conversation messages.

    Returns (extracted_facts, new_last_index).
    Only processes messages from start_index onward.
    """

def extract_from_text(
    text: str,
    source: str = "tool_result",
) -> list[ExtractedFact]:
    """Extract facts from a plain text string.

    Used by on_delegation to extract from the subagent's result text,
    which is not in message-dict format.
    """
```

`extract_from_messages` iterates message dicts and dispatches to `extract_from_text` for each message's content. `extract_from_text` is the shared core that runs pattern matchers on raw text. `on_delegation` calls `extract_from_text` directly on the result string.

## Feature Specifications

### 1. initialize(session_id, **kwargs)

**Read from kwargs:**
- `agent_context`: If "cron" or "flush", set `self._read_only = True`. All write hooks become no-ops. Reads (prefetch, system_prompt_block) still work.
- `agent_identity`: Store as `self._profile`. Scope DB path to `~/.hermes/mnemoria-{profile}.db` if set. Each hermes profile gets its own fact store.
- `user_id`: Store as `self._user_id`. Tag extracted facts with user_id for per-user scoping in gateway mode.
- `platform`: Store as `self._platform` for metadata on extracted facts.

**DB path resolution order:**
1. `HERMES_MNEMORIA_DB` env var (explicit override)
2. `~/.hermes/mnemoria-{profile}.db` (if agent_identity is set)
3. `~/.hermes/mnemoria.db` (default)

### 2. system_prompt_block()

Query store for Constraint and Decision facts with high activation scores. Format as:

```
[MNEMORIA MEMORY]
Mnemoria is active. Use mnemoria_write to store facts (supports
MEMORY_SPEC notation: C[target]: constraints, D[target]: decisions,
V[target]: values). Use mnemoria_recall for semantic search,
mnemoria_explore for multi-hop discovery.

[MNEMORIA IDENTITY]
C[auth]: JWT tokens must expire after 7 days
D[style]: Prefer concise responses
```

**Cap at 10 identity facts**, sorted by activation score (highest first). Only include facts with importance >= 0.7. This block is cached for the session.

### 3. get_config_schema() / save_config()

**Schema:**
```python
[
    {"key": "db_path", "description": "SQLite database path",
     "default": "~/.hermes/mnemoria.db"},
    {"key": "profile", "description": "Memory profile",
     "default": "balanced", "choices": ["balanced"]},
]
```

**save_config:** Write to `$HERMES_HOME/mnemoria.json`. Merge with existing config if file exists. Match Honcho's pattern exactly.

### 4. prefetch(query)

Check `self._prefetch_result` under lock first. If a cached result exists from the previous turn's queue_prefetch, format and return it. Clear the cache after use. Fall back to synchronous `store.recall(query, top_k=8)` if no cache hit.

### 5. queue_prefetch(query)

Spawn a daemon thread named "mnemoria-prefetch" that runs `store.recall(query, top_k=8)` and stores results in `self._prefetch_result` under `self._prefetch_lock`. If a previous prefetch thread is still running, join it with a 2-second timeout before starting a new one.

Skip if `self._read_only` or query is empty.

### 6. on_memory_write(action, target, content)

**Guard:** Skip if `self._read_only`, action not in ("add", "replace"), or content is empty.

**Fact mapping:**
- `target="user"` -> `V[user.{slug}]: {content}` where `{slug}` is derived from the first few meaningful words of content (e.g., "User prefers dark mode" -> `V[user.dark-mode]: User prefers dark mode`). This ensures different user facts don't supersede each other.
- `target="memory"` -> `V[memory.{slug}]: {content}` (same slugging approach).

The slug derivation keeps supersession working correctly: if the user updates a preference ("now I prefer light mode"), the new `V[user.light-mode]` doesn't supersede the unrelated `V[user.timezone]`. But if the same topic is updated, the slug matches and supersession retracts the old fact.

**Slug generation:** Strip common words (the, a, is, are, etc.), take first 2-3 content words, lowercase, join with hyphens. E.g., "Python 3.12 project" -> `python-3.12`. Keep it simple — this is a best-effort discriminator, not a semantic classifier.

**Run in background thread** (daemon, named "mnemoria-memory-write") to avoid blocking the agent. Match Supermemory's pattern.

### 7. on_delegation(task, result, child_session_id)

**Guard:** Skip if `self._read_only`.

**Two-step process:**
1. Store summary fact: `D[delegation]: {task_truncated} -> {result_truncated}` (truncate both to ~200 chars)
2. Run pattern extraction on the `result` text (URLs, errors, file paths, values). Tag extracted facts with `scope="delegation:{child_session_id}"` for traceability.

### 8. on_pre_compress(messages)

**Guard:** Skip if `self._read_only`.

**Process:**
1. Call `extract_from_messages(messages, start_index=self._last_extracted_msg_index)`
2. Store each extracted fact via `store.store(fact.content)`
3. Update `self._last_extracted_msg_index` to the returned new index

**Return:** Empty string (return value is discarded upstream — see hermes-agent bug #1 in handover doc).

### 9. on_session_end(messages)

**Process:**
1. Run pattern extraction on messages not yet processed (from `self._last_extracted_msg_index` onward)
2. Store extracted facts
3. Run `store.consolidate()` — promote/demote/prune facts, decay Hebbian links
4. Log extraction and consolidation stats at INFO level

### 10. shutdown()

**Current behavior is fine.** Close the per-thread store connection. Wait for any background threads (prefetch, memory-write) with a 5-second timeout.

## Testing Strategy

**Unit tests for extract.py:**
- Test each pattern matcher independently with sample messages
- Test message index tracking (only new messages processed)
- Test edge cases: empty messages, messages with no tool results, unicode content

**Integration tests for provider.py:**
- Test on_memory_write stores correct fact type for user vs memory target
- Test on_delegation stores summary + extracts from result
- Test on_pre_compress + on_session_end don't double-extract
- Test initialize with agent_context="cron" makes all writes no-op
- Test system_prompt_block returns identity facts
- Test queue_prefetch -> prefetch cache flow
- Test get_config_schema returns valid schema

**Smoke tests:**
- Plugin loads and registers via register(ctx)
- All 8 tool schemas present
- Graceful degradation when mnemoria package not installed

## Upstream Dependencies

These hermes-agent bugs affect Mnemoria but are NOT blocking. The plugin works around them:

1. **on_pre_compress return value discarded** — We return empty string and don't depend on it. Potential upstream PR to fix.
2. **on_turn_start never called** — We don't use it. No workaround needed.
3. **remaining_tokens never computed** — We don't use it. No workaround needed.

## Out of Scope

- LLM-based fact extraction (future v0.2.0 agent_inference mode)
- Summary fallback for unmatched content
- CLI commands (hermes mnemoria status, etc.)
- on_turn_start integration (dead code upstream)
- Token-aware prefetch (remaining_tokens not available)
- Influencing compression summaries (return value discarded upstream)
