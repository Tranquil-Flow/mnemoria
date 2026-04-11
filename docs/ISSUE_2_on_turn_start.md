# Issue 2: on_turn_start never called

## gh issue create command

```bash
gh issue create \
  --repo NousResearch/hermes-agent \
  --title "[Bug]: MemoryProvider.on_turn_start() hook is never called — breaks Honcho cadence logic and plugin turn tracking" \
  --label "bug" \
  --body "$(cat <<'ISSUE_EOF'
### Bug Description

The `MemoryProvider` ABC defines `on_turn_start(turn_number, message, **kwargs)` (`agent/memory_provider.py` line 144) and `memory_manager.py` has a complete dispatch method that iterates providers and calls it (lines 265-277). However, `run_agent.py` never calls `memory_manager.on_turn_start()` anywhere.

This means `_turn_count` stays at 0 in every memory plugin that relies on it.

### Steps to Reproduce

1. Enable Honcho with `dialectic_cadence` or `injection_frequency` set to any value > 1
2. Have a multi-turn conversation (5+ turns)
3. Observe via DEBUG logging that Honcho's `on_turn_start()` is never called
4. Observe that `_turn_count` remains 0 throughout the session

Quick verification:
```bash
grep -n "on_turn_start" run_agent.py
# Returns no results — the call is missing entirely
```

### Expected Behavior

`memory_manager.on_turn_start()` should be called once per user turn, after the turn counter is incremented, so that memory plugins can track turn progression and gate behavior (e.g., Honcho's cadence logic, Supermemory's turn tracking).

### Actual Behavior

The hook is never invoked. All plugins that implement `on_turn_start()` have their turn-tracking fields stuck at their initial value (0).

**Affected plugins:**
- **Honcho** (`plugins/memory/honcho/__init__.py` lines 516-518): Sets `self._turn_count = turn_number`. Used in `queue_prefetch()` (lines 485-490) for `dialectic_cadence` and `injection_frequency` checks — both broken.
- **Supermemory** (`plugins/memory/supermemory/__init__.py` lines 527-528): Sets `self._turn_count = max(turn_number, 0)` — never called.

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

### Root Cause Analysis (optional)

The hook was defined in the ABC and dispatch was implemented in `memory_manager.py`, but the call was never wired into the main agent loop in `run_agent.py`. The natural insertion point is after `self._user_turn_count += 1` (line 7156), where the agent already tracks turn progression.

The infrastructure is complete — the only missing piece is the single call from `run_agent.py`.

### Proposed Fix (optional)

Add the call in `run_agent.py` after line 7156 (`self._user_turn_count += 1`):

```python
if self._memory_manager:
    try:
        self._memory_manager.on_turn_start(
            self._user_turn_count,
            user_message if isinstance(user_message, str) else str(user_message),
            platform=getattr(self, '_platform', None),
            model=self._current_model if hasattr(self, '_current_model') else self.model,
            tool_count=len(self.valid_tool_names) if hasattr(self, 'valid_tool_names') else 0,
        )
    except Exception:
        pass
```

PR ready: fix/on-turn-start-never-called
ISSUE_EOF
)"
```

## PR body

```
## What does this PR do?

Wires up the `MemoryProvider.on_turn_start()` hook that was defined in the ABC and fully implemented in the memory manager dispatcher but never called from `run_agent.py`. This fixes turn tracking for all memory plugins, most notably Honcho's `dialectic_cadence` and `injection_frequency` logic which depend on accurate turn counts.

## Related Issue

Fixes #ISSUE_NUMBER

## Type of Change

- [x] 🐛 Bug fix (non-breaking change that fixes an issue)

## Changes Made

- `run_agent.py` after line ~7156: Add `self._memory_manager.on_turn_start()` call after `_user_turn_count` is incremented, passing `platform`, `model`, and `tool_count` as kwargs per the ABC docstring contract

## How to Test

1. Enable Honcho with `dialectic_cadence: 3` (or any value > 1)
2. Start a multi-turn conversation
3. Verify via DEBUG logs that `on_turn_start` is called each turn with incrementing turn numbers
4. Verify that Honcho's cadence gating now activates at the correct turn intervals

For code review only:
1. `grep -n "on_turn_start" run_agent.py` — confirm the call now exists
2. Verify the call passes `turn_number`, `message`, and kwargs consistent with the ABC docstring
3. Verify the call is wrapped in try/except consistent with other memory hook call sites

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
