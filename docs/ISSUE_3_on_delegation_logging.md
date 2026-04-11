# Issue 3: on_delegation silent exception handling

## gh issue create command

```bash
gh issue create \
  --repo NousResearch/hermes-agent \
  --title "[Bug]: on_delegation notification in delegate_tool.py silently swallows all exceptions" \
  --label "bug" \
  --body "$(cat <<'ISSUE_EOF'
### Bug Description

`tools/delegate_tool.py` (lines 683-684) uses a bare `except: pass` when notifying memory providers of delegation outcomes. This makes it impossible to diagnose failures in the delegation memory hook without adding print statements.

### Steps to Reproduce

1. Enable a memory provider with an `on_delegation()` implementation that raises an exception (e.g., network error, attribute error)
2. Run a delegation task
3. Observe: no log output, no error, no indication that `on_delegation` failed

### Expected Behavior

Exceptions from `on_delegation` should be logged at DEBUG level, consistent with how `memory_manager.py` handles exceptions in its own dispatch methods (lines 334-337).

### Actual Behavior

All exceptions are silently swallowed:

```python
# delegate_tool.py lines 683-684
except Exception:
    pass  # completely silent
```

Compare with memory_manager.py's consistent pattern:

```python
# memory_manager.py lines 334-337
except Exception as e:
    logger.debug(
        "Memory provider '%s' on_delegation failed: %s",
        provider.name, e,
    )
```

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

The call site in `delegate_tool.py` wraps the `on_delegation` notification in a try/except that catches all exceptions and silently discards them. The `logger` is already imported and used elsewhere in the file (line 21), so this is likely an oversight rather than a deliberate choice.

### Proposed Fix (optional)

Replace:
```python
except Exception:
    pass
```
With:
```python
except Exception as e:
    logger.debug("on_delegation notification failed for task %d: %s", entry.get("task_index", -1), e)
```

PR ready: fix/on-delegation-silent-exception
ISSUE_EOF
)"
```

## PR body

```
## What does this PR do?

Replaces a bare `except: pass` in `delegate_tool.py` with debug-level logging when the `on_delegation` memory hook notification fails. This makes delegation failures diagnosable without adding print statements, consistent with the logging pattern used in `memory_manager.py`.

## Related Issue

Fixes #ISSUE_NUMBER

## Type of Change

- [x] 🐛 Bug fix (non-breaking change that fixes an issue)

## Changes Made

- `tools/delegate_tool.py` lines 683-684: Replace `except Exception: pass` with `except Exception as e: logger.debug(...)`, logging the task index and exception message

## How to Test

1. Enable a memory provider with an `on_delegation()` that raises (or temporarily inject a `raise RuntimeError("test")`)
2. Run a delegation task
3. Run with `--log-level DEBUG` and verify the exception is now logged
4. Verify normal (non-failing) delegations are unaffected

For code review only:
1. Confirm the bare `except: pass` is replaced with `except Exception as e: logger.debug(...)`
2. Confirm `logger` is already defined in the file (line 21: `logger = logging.getLogger(__name__)`)
3. Confirm the logging format is consistent with `memory_manager.py` patterns

## Checklist

### Code

- [x] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`fix(tools): ...`)
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
