# Issue 1: on_pre_compress return value silently discarded

## gh issue create command

```bash
gh issue create \
  --repo NousResearch/hermes-agent \
  --title "[Bug]: MemoryProvider.on_pre_compress() return value silently discarded — provider insights never reach the compressor" \
  --label "bug" \
  --body "$(cat <<'ISSUE_EOF'
### Bug Description

`run_agent.py` line 6081 calls `self._memory_manager.on_pre_compress(messages)` as a bare statement, discarding the return value. The `memory_manager.on_pre_compress()` method correctly collects text from all providers and returns it as a combined string (joined with `"\n\n"`), but this text is never passed to the compressor.

Additionally, `context_compressor.compress()` and `_generate_summary()` have no parameter to accept memory provider context, so even capturing the return value has nowhere to go without updating those signatures.

This affects **all** MemoryProvider plugins (Honcho, Holographic, Mem0, Supermemory, RetainDB, and any third-party plugins).

### Steps to Reproduce

1. Implement a MemoryProvider that returns non-empty text from `on_pre_compress()`
2. Trigger context compression (long conversation or `/compress`)
3. Observe via DEBUG logging that the provider's `on_pre_compress()` returns text
4. Observe that the returned text never appears in the compression summary

### Expected Behavior

Text returned by `MemoryProvider.on_pre_compress()` should be injected into the compression summary prompt so the compressor preserves provider-extracted insights (as documented in the ABC docstring and `memory_manager.on_pre_compress()` return-value docstring).

### Actual Behavior

The return value is silently discarded. The hook only works as a notification (observe messages before they're discarded), not as the injection point it was designed to be.

### Affected Component

Agent Core (conversation loop, context compression, memory)

### Messaging Platform (if gateway-related)

N/A (CLI only)

### Operating System

macOS 15.6.1

### Python Version

3.11.14

### Hermes Version

v0.8.0

### Relevant Logs / Traceback

No error is raised — the bug is silent. The return value is simply dropped.

```python
# run_agent.py line 6081 — return value not captured
self._memory_manager.on_pre_compress(messages)
```

### Root Cause Analysis (optional)

Three-part gap:

1. **Call site** (`run_agent.py` line 6081): `self._memory_manager.on_pre_compress(messages)` is called as a statement — the return value is not captured.
2. **compress() signature** (`agent/context_compressor.py` line 612): `compress(self, messages, current_tokens=None)` has no parameter to accept memory context.
3. **_generate_summary()** (`agent/context_compressor.py` line 287): No mechanism to inject provider insights into the summarization template.

The dispatch layer (`memory_manager.py` lines 290-307) is correct — it collects provider contributions and returns a combined string. The bug is entirely at the caller and consumer level.

### Proposed Fix (optional)

1. Capture the return value in `run_agent.py`
2. Add `memory_context: str = ""` parameter to `compress()` and `_generate_summary()`
3. Inject a "MEMORY PROVIDER INSIGHTS" section into the summarization prompt when non-empty

PR ready: fix/on-pre-compress-return-discarded
ISSUE_EOF
)"
```

## PR body

```
## What does this PR do?

Captures the return value of `MemoryProvider.on_pre_compress()` and passes it through to the context compressor's summarization prompt. Previously, the return value was silently discarded — every memory plugin returning compression context was broken without any error.

The fix threads `memory_context` through three layers:
1. `run_agent.py` — captures the return value
2. `compress()` — accepts and forwards the new parameter
3. `_generate_summary()` — injects a "MEMORY PROVIDER INSIGHTS" section into the summarization template when non-empty

## Related Issue

Fixes #ISSUE_NUMBER

## Type of Change

- [x] 🐛 Bug fix (non-breaking change that fixes an issue)

## Changes Made

- `run_agent.py` line ~6078: Capture `on_pre_compress()` return value into `pre_compress_context`; pass to `compress()` as `memory_context=`
- `agent/context_compressor.py` `compress()`: Add `memory_context: str = ""` parameter; forward to `_generate_summary()`
- `agent/context_compressor.py` `_generate_summary()`: Add `memory_context: str = ""` parameter; conditionally inject "MEMORY PROVIDER INSIGHTS" section into both the iterative-update and first-compaction prompts

## How to Test

1. Enable a memory provider that returns text from `on_pre_compress()` (e.g., Honcho, or a test stub)
2. Have a conversation long enough to trigger compression (or use `/compress`)
3. Verify (via DEBUG logs) that the provider's returned text appears in the summarization prompt
4. Verify the compression summary references provider insights when relevant

For code review only (no runtime test harness):
1. Read `run_agent.py` line ~6078 — confirm return value is now captured
2. Read `compress()` signature — confirm `memory_context` parameter exists
3. Read `_generate_summary()` — confirm the "MEMORY PROVIDER INSIGHTS" section is injected conditionally

## Checklist

### Code

- [x] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`fix(agent): ...`)
- [x] I searched for [existing PRs](https://github.com/NousResearch/hermes-agent/pulls) to make sure this isn't a duplicate
- [x] My PR contains **only** changes related to this fix/feature (no unrelated commits)
- [ ] I've run `pytest tests/ -q` and all tests pass
- [ ] I've added tests for my changes (required for bug fixes, strongly encouraged for features)
- [x] I've tested on my platform: macOS 15.6.1 (arm64)

### Documentation & Housekeeping

- [x] I've updated relevant documentation (README, `docs/`, docstrings) — or N/A
- [x] I've updated `cli-config.yaml.example` if I added/changed config keys — or N/A
- [x] I've updated `CONTRIBUTING.md` or `AGENTS.md` if I changed architecture or workflows — or N/A
- [x] I've considered cross-platform impact (Windows, macOS) per the [compatibility guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md#cross-platform-compatibility) — or N/A
- [x] I've updated tool descriptions/schemas if I changed tool behavior — or N/A
```
