# Handover: Mnemoria Plugin Lifecycle Implementation

**Date:** 2026-04-10
**Author:** Claude Code session
**Status:** Implementation complete, needs push + PR update + final review + v2 planning

---

## What Was Done This Session

Three major workstreams completed:

### 1. Plugin Convention Fixes (pre-existing PR #5990)
- Added `register(ctx)` entry point to `__init__.py`
- Removed non-standard `plugin.yaml` fields (`provider`, `memory_provider`, `config` block)
- Renamed all tools from `mcp_umemory_*` to `mnemoria_*` (matching hermes-agent convention)
- Added `README.md` with setup/config/tools documentation
- Updated tests to match new tool names

### 2. Full Lifecycle Hook Implementation (12 commits on `draft/mnemoria-plugin`)
Transformed the plugin from a passive store/recall endpoint into a full lifecycle participant:

| Hook | What it does |
|------|-------------|
| `initialize` | Context-aware setup: read-only for cron/flush, per-profile DB scoping, user_id tracking |
| `system_prompt_block` | Injects usage hint + identity facts (C/D types with importance >= 0.7) |
| `get_config_schema` / `save_config` | `hermes memory setup` wizard support, writes to mnemoria.json |
| `prefetch` | Cache-aware: checks queue_prefetch result first, falls back to sync recall |
| `queue_prefetch` | Background daemon thread pre-warms recall for next turn |
| `on_memory_write` | Mirrors MEMORY.md/USER.md writes as typed facts with content_slug targets |
| `on_delegation` | Stores D[delegation] summary + extracts facts from subagent results |
| `on_pre_compress` | Extracts typed facts from messages before context compression |
| `on_session_end` | Final extraction pass + consolidation (promote/demote/prune) |
| `shutdown` | Waits for background threads (prefetch, write) before closing |

New file `extract.py` provides shared extraction logic (pattern matching for errors, URLs near errors, file paths near errors, user directives).

### 3. Competitive Audit + Acknowledgements Cleanup
- Audited all 8 existing memory plugins for hook usage (see "Competitive Landscape" below)
- Updated `ACKNOWLEDGEMENTS.md`: added Ori-Mnemos (correct URL: github.com/aayoawoyemi/Ori-Mnemos), removed "reviewed but not incorporated" section
- Found 4 hermes-agent bugs (documented in `HANDOVER_hermes-agent-bugs.md`)

---

## Files Changed

### Hermes-agent worktree (`/Users/evinova/Projects/hermes-agent-mnemoria-pr-draft/`)
Branch: `draft/mnemoria-plugin`

| File | Status | Description |
|------|--------|-------------|
| `plugins/memory/mnemoria/__init__.py` | Modified | Added `register(ctx)` entry point |
| `plugins/memory/mnemoria/plugin.yaml` | Modified | Cleaned to match conventions |
| `plugins/memory/mnemoria/provider.py` | Modified | All lifecycle hooks implemented |
| `plugins/memory/mnemoria/extract.py` | **New** | Rule-based fact extraction module |
| `plugins/memory/mnemoria/README.md` | **New** | Full documentation |
| `tests/plugins/memory/test_mnemoria_plugin.py` | Modified | 22 tests (was 4) |
| `tests/plugins/memory/test_mnemoria_extract.py` | **New** | 23 tests |

### Mnemoria repo (`/Users/evinova/Projects/mnemoria/`)
Branch: `main` (uncommitted changes)

| File | Status | Description |
|------|--------|-------------|
| `ACKNOWLEDGEMENTS.md` | Modified | Added Ori-Mnemos, removed non-inspiration section |
| `docs/HANDOVER_hermes-agent-bugs.md` | **New** | 4 upstream bugs documented |
| `docs/HANDOVER_plugin-lifecycle-implementation.md` | **New** | This file |
| `docs/superpowers/specs/2026-04-10-hermes-plugin-lifecycle-hooks-design.md` | **New** | Design spec |
| `docs/superpowers/plans/2026-04-10-hermes-plugin-lifecycle-hooks.md` | **New** | Implementation plan |

### Test Status

45/45 passing:
```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
python -m pytest tests/plugins/memory/test_mnemoria_plugin.py tests/plugins/memory/test_mnemoria_extract.py -v
```

---

## Immediate TODO: Push + PR Update + Commits

### 1. Push plugin branch and update PR #5990

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git push origin draft/mnemoria-plugin --force
```

Update PR description at https://github.com/NousResearch/hermes-agent/pull/5990 — it currently says "Draft — opening for early feedback". Replace with the lifecycle integration description.

### 2. Commit mnemoria repo changes

```bash
cd /Users/evinova/Projects/mnemoria
git add ACKNOWLEDGEMENTS.md
git commit -m "docs: add Ori-Mnemos to acknowledgements, clean up credit list"

git add docs/
git commit -m "docs: add plugin lifecycle spec, plan, and handover docs"

git push
```

### 3. Discord announcement

A finalized draft Discord post was prepared. Key elements:
- Honest differentiation (first to use on_delegation, one of two using on_pre_compress, typed decay, RL, multi-hop)
- Explicitly notes what IS standard practice to avoid overclaiming
- Comparative benchmark table with heavy disclaimers
- Ori-Mnemos as primary inspiration (correct URL)
- Links: benchmark PR #5728, plugin PR #5990, mnemoria repo

The post is inline in the conversation — search for "Introducing Mnemoria -- cognitive memory for Hermes agents" (the final revision, not the earlier draft).

### 4. Upstream hermes-agent bug PRs (optional, separate)

Documented in `/Users/evinova/Projects/mnemoria/docs/HANDOVER_hermes-agent-bugs.md`:

| Bug | Severity | File:Line | Summary |
|-----|----------|-----------|---------|
| on_pre_compress return value discarded | Medium | run_agent.py:6095 | Return value thrown away, compress() has no param for it |
| on_turn_start never called | Medium | run_agent.py (missing) | ABC defined, plugins implement it, but never wired up |
| on_delegation silent exception | Low | delegate_tool.py:683 | Bare `except: pass` at call site |
| remaining_tokens never computed | Low | memory_provider.py docstring | Documented but never calculated or passed |

---

## Competitive Landscape: All 8 Memory Plugins Audited

This audit was done by reading the actual source code of every plugin.

### Hook usage by plugin

| Hook | honcho | mem0 | supermemory | retaindb | openviking | byterover | hindsight | holographic | **mnemoria** |
|------|--------|------|-------------|----------|------------|-----------|-----------|-------------|-------------|
| initialize | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | **Yes** |
| system_prompt_block | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | **Yes** |
| prefetch | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | **Yes** |
| queue_prefetch | Yes | Yes | No | Yes | Yes | No | Yes | No | **Yes** |
| sync_turn | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No |
| on_turn_start | Yes* | No | Yes* | No | No | No | No | No | No* |
| on_session_end | Yes | No | Yes | Yes | Yes | No | No | Yes | **Yes** |
| on_pre_compress | No | No | No | No | No | **Yes** | No | No | **Yes** |
| on_memory_write | Yes | No | Yes | Yes | Yes | Yes | No | Yes | **Yes** |
| on_delegation | No | No | No | No | No | No | No | No | **Yes (first)** |
| get_config_schema | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | **Yes** |
| save_config | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | **Yes** |
| shutdown | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | **Yes** |

*on_turn_start is dead code upstream — never called by run_agent.py. Honcho and Supermemory implement it but their _turn_count is always 0.

### Notable patterns worth learning from

**RetainDB — Durable SQLite write queue:**
- Writes to SQLite queue BEFORE sending to API
- Pending rows replay on startup if hermes crashed
- `_WriteQueue` class with background worker thread
- Location: `plugins/memory/retaindb/__init__.py:327-406`

**Honcho — Cadence controls:**
- Configurable `dialectic_cadence` and `injection_frequency`
- Skips prefetch/injection on turns where it's not needed
- Cost-aware: doesn't waste API calls on trivial turns
- Location: `plugins/memory/honcho/__init__.py:138-142, 483-509`

**Mem0 — Circuit breaker:**
- After N consecutive API failures, pauses for cooldown
- Prevents cascading failures from hammering a down API
- Simple counter + timestamp pattern
- Location: `plugins/memory/mem0/__init__.py:134-200`

**RetainDB — Parallel prefetch:**
- Fires 3 parallel background queries (context, dialectic, agent model)
- Merges results before returning
- Location: `plugins/memory/retaindb/__init__.py:571-608`

**Supermemory/OpenViking — Full conversation ingest on session end:**
- Stores the entire conversation, not just extracted facts
- Enables future re-analysis and richer retrieval

---

## V2 Plugin: Ideas to Integrate From Other Providers

These are ideas for the next version of the Mnemoria plugin, incorporating lessons from the competitive audit.

### High Priority

#### 1. Crash-safe extraction via um_pending
**Inspired by:** RetainDB's durable SQLite write queue

Currently extracted facts from on_pre_compress and on_session_end go directly to `store.store()` in the main facts table. If hermes crashes mid-extraction, partial work is lost.

**Idea:** Route extracted facts through Mnemoria's existing `um_pending` table (from v0.2.0) instead. Facts go to pending first, get promoted later. This is crash-safe AND gives users control to review/retract auto-extracted facts before they're permanent.

**Where it fits:** This belongs in the core mnemoria package (`store.py` or `promoter.py`), not just the plugin. The plugin calls `store.store_pending()` instead of `store.store()` for auto-extracted facts.

#### 2. sync_turn for immediate user preference capture
**Inspired by:** 6/8 plugins using sync_turn

We made sync_turn a no-op because it doesn't get tool results. But it DOES get the user's message immediately after the turn — faster than waiting for on_pre_compress or on_session_end.

**Idea:** Use sync_turn for lightweight user-statement extraction only (directives like "always use TypeScript", URLs, file paths from user messages). Leave tool-result extraction for on_pre_compress/on_session_end. This gives faster feedback for user preferences.

**Where it fits:** Plugin only (`provider.py`). Calls `extract_from_text(user_content, source="user_statement")` and stores results.

#### 3. Extraction cadence / short-message filtering
**Inspired by:** Honcho's cadence controls

Not every turn is worth processing. "ok", "thanks", "yes" have zero extraction value. Running pattern matching on these wastes compute and could produce false positives.

**Idea:** Skip extraction in sync_turn/queue_prefetch when user message is under ~15 chars and contains no URLs/paths/directives. Also consider configurable cadence for prefetch (every turn vs. every N turns).

**Where it fits:** Plugin (`provider.py`) — simple length/content check before calling extract functions.

### Medium Priority

#### 4. Circuit breaker for store operations
**Inspired by:** Mem0's circuit breaker pattern

If the SQLite file is on a network drive or the store encounters repeated errors, the plugin should back off rather than logging the same failure every turn.

**Idea:** Track consecutive store failures. After N failures, set a cooldown period where write operations are skipped. Reads (prefetch) continue. Reset on first success.

**Where it fits:** Could go in the plugin (`provider.py`) or in the core mnemoria package. Plugin is simpler.

#### 5. Richer on_delegation: store tool trace if available upstream
**Inspired by:** delegate_tool.py builds a tool_trace but doesn't pass it to on_delegation

The tool trace (which tools the subagent used, which failed, result sizes) is built in delegate_tool.py:399-435 but not forwarded to the memory hook. If upstream adds it as a kwarg, we should consume it.

**Idea:** Check `kwargs.get("tool_trace")` in on_delegation. If present, extract richer facts: "subagent used pytest (failed)", "subagent used git (success)". This is forward-compatible — works today (ignores missing kwarg) and automatically activates if upstream adds it.

**Where it fits:** Plugin (`provider.py`), defensive kwargs check.

#### 6. Parallel prefetch with separate recall strategies
**Inspired by:** RetainDB's 3-thread parallel prefetch

RetainDB fires context + dialectic + agent-model queries in parallel. Mnemoria could fire semantic recall + FTS5 keyword search separately in parallel threads and merge results.

**Idea:** However, Mnemoria already fuses these internally in a single `recall()` call — the RRF fusion stage combines embedding similarity and BM25. So the parallelism is inside the pipeline, not outside. This is a lower priority unless profiling shows the single `recall()` is too slow.

**Where it fits:** Core mnemoria package if needed.

### Low Priority / Future

#### 7. Session summary fact on session end
**Inspired by:** Supermemory and OpenViking's full conversation ingest

Instead of (or in addition to) extracting individual facts, store a single high-level summary fact like `D[session.2026-04-10]: Refactored auth module, decided JWT 7d, deployed to staging`.

**Idea:** This would require LLM-based summarization (not rule-based) so it's deferred until v0.2.0's agent_inference mode. But the hook wiring is already in place — on_session_end would just need to call an LLM summarizer.

**Where it fits:** Core mnemoria package (agent_inference extraction mode).

#### 8. Multi-store/multi-profile management
**Inspired by:** Supermemory's multi-container support

Current implementation creates one DB per hermes profile. But users might want shared facts across profiles (e.g., user preferences) while keeping project-specific facts isolated.

**Idea:** Two-tier store: a shared "identity" store (C/D facts, user preferences) and a per-profile "project" store (V facts, errors, delegation outcomes). system_prompt_block reads from both.

**Where it fits:** Core mnemoria package (multi-store architecture) + plugin (routing logic).

---

## Architecture Decisions Made (Reference)

### Why rule-based extraction (not LLM)
- No API calls, zero latency, deterministic
- High-precision for tool errors, URLs, file paths
- LLM extraction deferred to v0.2.0's agent_inference mode
- Architecture is clean — just swap extract_from_text internals later

### Why URLs/paths only near errors
- Prevents noise from routine grep/cat/read tool output
- A single grep result can contain 50+ file paths
- User-shared URLs/paths are always extracted (different source type)

### Why content_slug for on_memory_write targets
- Prevents supersession clobbering (all user writes going to same target)
- `V[user.jwt-authentication]` won't supersede `V[user.timezone]`
- Simple: strip stop words, take first 2-3 meaningful words, lowercase

### Why sync_turn is currently a no-op (may change in v2)
- Only gets clean user text + final assistant text (no tool calls/results)
- on_pre_compress and on_session_end get full message history including tool results
- V2 idea: use it for lightweight user-preference extraction only

### Why on_turn_start is not used
- Dead code upstream — defined in ABC but never called by run_agent.py
- Honcho and Supermemory implement it but their _turn_count is always 0
- Not worth building on until upstream fixes it

---

## Worktree Notes

The worktree at `/Users/evinova/Projects/hermes-agent-mnemoria-pr-draft/` had a broken git link (referenced `/workspace/Projects/` from a sandbox). Fixed during this session. 9 other broken worktrees also fixed. 3 `/tmp/` worktrees are prunable (`git worktree prune`).

---

## Document Locations

| Document | Path |
|----------|------|
| This handover | `/Users/evinova/Projects/mnemoria/docs/HANDOVER_plugin-lifecycle-implementation.md` |
| Design spec | `/Users/evinova/Projects/mnemoria/docs/superpowers/specs/2026-04-10-hermes-plugin-lifecycle-hooks-design.md` |
| Implementation plan | `/Users/evinova/Projects/mnemoria/docs/superpowers/plans/2026-04-10-hermes-plugin-lifecycle-hooks.md` |
| Bug handover | `/Users/evinova/Projects/mnemoria/docs/HANDOVER_hermes-agent-bugs.md` |
| Discord post | Inline in conversation — search for "Introducing Mnemoria -- cognitive memory for Hermes agents" (final revision) |
| Competitive audit | Inline in this handover (section: "Competitive Landscape") |
