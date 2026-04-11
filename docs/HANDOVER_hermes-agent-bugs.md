# Handover: Hermes-Agent Bugs Found During Mnemoria Integration

**Date:** 2026-04-10
**Author:** Claude Code session (ironman analysis of hermes-agent memory provider hooks)
**Repo:** https://github.com/NousResearch/hermes-agent
**Base commit:** c722983b (branch: fix/update-restart-health-check at time of analysis)

## Context

These bugs were found while verifying every assumption about hermes-agent's MemoryProvider hook system. The analysis read actual source code with line numbers, not documentation. Each bug is confirmed by reading the call site, the dispatch logic, and the ABC definition.

These affect ALL memory plugins (Honcho, Holographic, Mem0, Supermemory, RetainDB, etc.), not just Mnemoria.

---

## Bug 1: on_pre_compress return value is silently discarded

**Severity:** Medium — feature silently broken, no error raised
**Files:**
- `run_agent.py` line 6095 (call site)
- `agent/memory_manager.py` lines 290-307 (dispatch, returns combined string)
- `agent/context_compressor.py` lines 612-755 (compress() has no param for it)

**What happens:**
```python
# run_agent.py line 6095
self._memory_manager.on_pre_compress(messages)  # return value IGNORED
```

The `memory_manager.on_pre_compress()` docstring says it returns "combined text from providers to include in the compression summary prompt." The method correctly collects provider contributions and joins them with `"\n\n"`. But `run_agent.py` calls it as a statement, not capturing the return value. And `compress()` doesn't have a parameter to accept memory provider context anyway.

**Impact:** Every memory plugin that returns text from `on_pre_compress()` expecting it to influence the compression summary is silently broken. The hook only works as a notification (observe messages before they're discarded), not as an injection point.

**Verification steps:**
1. Read `run_agent.py` line 6095 — confirm return value is not captured
2. Read `context_compressor.compress()` signature — confirm no `memory_context` parameter
3. Read `_generate_summary()` template — confirm no memory provider section

**Suggested fix:**
```python
# run_agent.py line 6095 — capture the return value
pre_compress_context = ""
if self._memory_manager:
    try:
        pre_compress_context = self._memory_manager.on_pre_compress(messages)
    except Exception:
        pass

# Pass to compress()
compressed = self.context_compressor.compress(
    messages, current_tokens=approx_tokens,
    memory_context=pre_compress_context,
)
```

Then update `compress()` and `_generate_summary()` to accept and inject `memory_context` into the summarization template. A natural place would be a "Memory Provider Insights" section in the summary template, after the turns-to-summarize section.

---

## Bug 2: on_turn_start is defined but never called

**Severity:** Medium — dead code causing silent behavior bugs in plugins
**Files:**
- `agent/memory_provider.py` line 144 (ABC definition)
- `agent/memory_manager.py` lines 265-277 (dispatch logic, fully implemented)
- `run_agent.py` — **missing call site entirely**

**What happens:**
The MemoryProvider ABC defines `on_turn_start(turn_number, message, **kwargs)`. The memory_manager has a complete dispatch method that iterates providers and calls it. But `run_agent.py` never calls `memory_manager.on_turn_start()` anywhere.

**Impact on existing plugins:**
- **Honcho** (`plugins/memory/honcho/__init__.py` lines 516-518): Uses `on_turn_start` to set `self._turn_count`. Since it's never called, `_turn_count` is always 0. This breaks Honcho's cadence logic for `dialectic_cadence` and `injection_frequency` — the cadence checks in `queue_prefetch()` (lines 485-490) compare against `self._turn_count` which is always 0.
- **Supermemory** (`plugins/memory/supermemory/__init__.py` lines 527-528): Same pattern — sets `self._turn_count = max(turn_number, 0)`, but it's never called.

**Verification steps:**
1. `grep -n "on_turn_start" run_agent.py` — returns no results
2. Read Honcho's `queue_prefetch()` — confirm it uses `self._turn_count` for cadence decisions
3. Confirm `self._turn_count` is never set elsewhere in Honcho's code (only in `on_turn_start`)

**Suggested fix:**
Add the call in `run_agent.py` around line 7170, after `self._user_turn_count` is incremented:
```python
# After line 7170: self._user_turn_count += 1
if self._memory_manager:
    try:
        self._memory_manager.on_turn_start(
            self._user_turn_count,
            user_message,
            platform=platform,
            model=self._current_model,
            tool_count=len(self.valid_tool_names),
        )
    except Exception:
        pass
```

**Additional missing kwarg:** The ABC docstring says kwargs may include `remaining_tokens`, but this value is never computed anywhere in the codebase. To fulfill the contract:
```python
remaining_tokens = self.context_compressor.context_length - (self.context_compressor.last_prompt_tokens or 0)
```
Pass this as an additional kwarg.

---

## Bug 3: on_delegation exception handling is too silent

**Severity:** Low — functional but impossible to debug
**Files:**
- `tools/delegate_tool.py` lines 673-684 (call site)

**What happens:**
```python
# delegate_tool.py lines 673-684
for entry in results:
    try:
        _task_goal = task_list[entry["task_index"]]["goal"] if entry["task_index"] < len(task_list) else ""
        parent_agent._memory_manager.on_delegation(
            task=_task_goal,
            result=entry.get("summary", "") or "",
            child_session_id=getattr(children[entry["task_index"]][2], "session_id", "") if entry["task_index"] < len(children) else "",
        )
    except Exception:
        pass  # <-- completely silent
```

Compare with memory_manager.py's own dispatch (lines 334-337) which logs at DEBUG level:
```python
except Exception as e:
    logger.debug(
        "Memory provider '%s' on_delegation failed: %s",
        provider.name, e,
    )
```

The call site swallows all exceptions with bare `except: pass`. If `parent_agent._memory_manager` doesn't exist, or if the attribute access fails, or if the method raises — no logging, no way to know it happened.

**Impact:** If on_delegation silently fails, memory plugins never know delegations happened. Debugging requires adding print statements.

**Suggested fix:**
```python
except Exception as e:
    logger.debug("on_delegation notification failed for task %d: %s", entry.get("task_index", -1), e)
```

---

## Bug 4 (Missing Feature): remaining_tokens never computed

**Severity:** Low — documented but unimplemented
**Files:**
- `agent/memory_provider.py` line 149 (docstring promises it)
- `agent/memory_manager.py` line 268 (docstring promises it)

**What happens:**
The on_turn_start docstring says `kwargs may include: remaining_tokens, model, platform, tool_count`. But since on_turn_start is never called (Bug 2), and even if it were, no code computes `remaining_tokens`.

**Impact:** Plugins can't implement token-aware prefetch (inject less context when the context window is tight). This would be genuinely useful for memory plugins that inject variable-length recall context.

**Suggested fix:** When wiring up on_turn_start (Bug 2 fix), also compute and pass remaining_tokens:
```python
_remaining = self.context_compressor.context_length - (self.context_compressor.last_prompt_tokens or 0)
```

---

## Additional Note: on_delegation doesn't pass tool trace

**Not a bug** — this is a design choice. But worth noting for future enhancement.

`delegate_tool.py` lines 399-435 build a detailed `tool_trace` list for each subagent (tool names, arg sizes, result sizes, error status). This trace is stored in the returned JSON dict but is NOT passed to the `on_delegation` hook. Memory plugins only receive `(task, result, child_session_id)`.

Passing the tool trace as a kwarg would let memory plugins extract richer information from delegation outcomes (which tools were used, which failed, how much data was produced).

**Suggested enhancement:**
```python
parent_agent._memory_manager.on_delegation(
    task=_task_goal,
    result=entry.get("summary", "") or "",
    child_session_id=...,
    tool_trace=entry.get("tool_trace", []),  # NEW
    status=entry.get("status", ""),           # NEW
    exit_reason=entry.get("exit_reason", ""), # NEW
)
```

---

## Summary

| # | Issue | Severity | Type | Affects |
|---|-------|----------|------|---------|
| 1 | on_pre_compress return value discarded | Medium | Bug | All plugins with on_pre_compress |
| 2 | on_turn_start never called | Medium | Bug | Honcho cadence, Supermemory turn tracking |
| 3 | on_delegation silent exception handling | Low | Bug | All plugins with on_delegation |
| 4 | remaining_tokens never computed | Low | Missing feature | All plugins wanting token-aware behavior |

## How to Verify

All bugs can be confirmed by reading the source code at the listed file paths and line numbers. No runtime testing needed — these are verifiable by code inspection. The line numbers are accurate as of commit c722983b.

## Recommended PR Strategy

These are independent fixes and can be submitted as separate PRs:
- **PR 1:** Fix on_pre_compress (Bug 1) — most impactful, benefits all plugins
- **PR 2:** Wire up on_turn_start (Bug 2) — fixes Honcho cadence bug
- **PR 3:** Add logging to on_delegation (Bug 3) — trivial fix
- **PR 4:** Compute remaining_tokens (Bug 4) — can be combined with PR 2
