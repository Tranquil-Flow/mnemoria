# Mnemoria Handover — 2026-04-08

## WHAT WAS DONE

This session consolidated all v0.1.1 changes into the v0.1.0 release and prepared
this handover document.

### Consolidated into v0.1.0

All fixes from the previously-planned v0.1.1 are now part of v0.1.0:

- `w_importance` weight increased 0.4 → 0.7 (capacity_stress 0%→100%, supersession 66.7%→86.7%)
- `get_system_prompt_facts()` method added to MnemoriaStore
- `system_prompt_block()` in provider now returns identity facts instead of empty string
- RRF auto-trigger threshold lowered 0.3 → 0.15
- 3 new smoke tests for system prompt injection
- Benchmark: 0.910 overall (3-run, zero variance)
- 8/8 tests pass

### Root Cause of Discord Identity Drift

When you switch models mid-Discord session (via `/model` in CLI), hermes-agent
creates a new AIAgent. The system prompt was rebuilt from scratch, and memory was
only injected via `prefetch()` which uses the user's message as a query — if the
message didn't match identity-relevant keywords, no identity facts came back.

Fix: `system_prompt_block()` now unconditionally returns the `[MNEMORIA IDENTITY]`
block with C/D/V facts on every call, BEFORE the prefetch query runs. This
guarantees identity is present regardless of what the user types.

---

## CURRENT STATE

```
/workspace/Projects/mnemoria/        — Mnemoria package (LOCAL, NOT pushed)
  mnemoria/store.py                  — MnemoriaStore + get_system_prompt_facts()
  mnemoria/config.py                 — w_importance=0.7
  plugins/memory/mnemoria/provider.py — system_prompt_block() returns identity block
  tests/test_basic.py                — 8/8 tests pass
  pyproject.toml                     — version 0.1.0
  CHANGELOG.md                       — consolidated v0.1.0 entry

/workspace/Projects/hermes-agent/    — hermes-agent (LOCAL)
  plugins/memory/mnemoria/           — provider plugin (UNTRACKED, not committed)
```

**GitHub repos are empty — nothing has been pushed yet.**

---

## COMPLETE SETUP PROCEDURE

Follow these steps in order. Each step is independent and can be verified.

---

### STEP 1 — Verify Mnemoria is Ready to Push

```bash
cd /workspace/Projects/mnemoria

# Verify tests pass
python3 -m pytest tests/test_basic.py -v

# Verify benchmark score
cd /workspace/Projects/hermes-agent
python -m benchmarks --backend mnemoria --suite all --runs 1 --output-dir /tmp/verify

# Check git status — should be clean except for untracked files
git status --short
```

Expected: 8/8 tests pass, benchmark ~0.910 overall.

---

### STEP 2 — Push Mnemoria to GitHub

```bash
cd /workspace/Projects/mnemoria

# Confirm GitHub remote is correct
git remote -v
# Should show: git@github.com:Tranquil-Flow/mnemoria.git

# Check if GitHub repo is reachable (SSH)
ssh -T git@github.com 2>&1 | head -1

# Push main branch
git push origin main

# Tag and push v0.1.0
git tag -a v0.1.0 -m "release: Mnemoria v0.1.0"
git push origin v0.1.0
```

If `git push origin main` fails because the repo is empty and SSH write access
is not set up, the user (Tranquil-Flow) needs to:
1. Go to https://github.com/Tranquil-Flow/mnemoria
2. Create the repo manually if it doesn't exist
3. Ensure their SSH key is added to GitHub
4. Then retry the push

---

### STEP 3 — Deploy the Provider Plugin

The `plugins/memory/mnemoria/` directory in hermes-agent is **untracked** — it
exists locally but is not committed to hermes-agent. Two options:

#### Option A: Merge into hermes-agent (recommended if user wants it in main)

```bash
cd /workspace/Projects/hermes-agent

# Check current branch
git branch

# Create a new branch for the plugin
git checkout -b feat/mnemoria-plugin-v0.1.0

# Copy the plugin files (they're already in plugins/memory/mnemoria/)
# Verify they're present
ls plugins/memory/mnemoria/

# Commit
git -c user.name="Tranquil-Flow" -c user.email="tranquil_flow@protonmail.com" \
  add plugins/memory/mnemoria/

git -c user.name="Tranquil-Flow" -c user.email="tranquil_flow@protonmail.com" \
  commit -m "feat(memory): add Mnemoria v0.1.0 plugin with system_prompt_block"

# Push branch
git push origin feat/mnemoria-plugin-v0.1.0
```

Then open a PR from `feat/mnemoria-plugin-v0.1.0` → `main`.

#### Option B: Volume mount in Docker (if user doesn't want to merge)

Add to docker-compose or Docker run command:
```yaml
volumes:
  - /workspace/Projects/hermes-agent/plugins/memory/mnemoria:/app/plugins/memory/mnemoria:ro
```

Or copy into the image via a custom Dockerfile:
```dockerfile
FROM ghcr.io/tranquil-flow/hermes-agent:latest
COPY plugins/memory/mnemoria/ /app/plugins/memory/mnemoria/
```

---

### STEP 4 — Update hermes-agent dependency

The hermes-agent `pyproject.toml` references mnemoria as:
```toml
mnemoria = ["mnemoria @ git+https://github.com/Tranquil-Flow/mnemoria.git"]
```

After pushing v0.1.0 to GitHub, rebuild/restart hermes-agent so it picks up
the new package:

```bash
cd /workspace/Projects/hermes-agent

# If using Docker, rebuild:
docker build --no-cache -t hermes-agent:mnemoria-v0.1.0 .
docker run hermes-agent:mnemoria-v0.1.0 ...

# If running directly:
pip install --break-system-packages -e ".[all]"
```

---

### STEP 5 — Verify Discord Identity Persistence

After hermes-agent restarts with the new code:

1. Open Discord, send a message to trigger a new session
2. Verify the agent responds as **Moon** (not "Hermes Agent", "Claude", etc.)
3. Switch model mid-session: `/model gpt-4` (or any model)
4. Send another message
5. Verify identity is **still Moon**, not reset

If identity still drifts after model switch, check:
```python
# In provider.py, system_prompt_block() should NOT be returning empty:
from plugins.memory.mnemoria import MnemoriaMemoryProvider
p = MnemoriaMemoryProvider()
print(repr(p.system_prompt_block()))  # Should show [MNEMORIA IDENTITY] block
```

---

### STEP 6 — Optional: Publish to PyPI

If you want `pip install mnemoria` to work from PyPI (instead of git URL):

```bash
cd /workspace/Projects/mnemoria

# Install build tool
pip install build --break-system-packages

# Build the package
python -m build

# Upload (you'll need a PyPI account and API token)
python -m twine upload dist/*

# Then update hermes-agent pyproject.toml to:
# mnemoria = ["mnemoria>=0.1.0"]
```

**WARNING**: Publishing to PyPI makes mnemoria publicly available. Review
whether the CHANGELOG, README, and package description are appropriate for
public consumption. The package currently contains identity-specific references
(Moonsong, Discord delivery, etc.) — consider whether those should be removed
before public release.

---

## MNEMORIA ARCHITECTURE SUMMARY

```
hermes-agent receive message
  → _build_system_prompt()
    → memory_provider.system_prompt_block()  ← ALWAYS called, returns identity block
    → prefetch_all(user_message)             ← query-dependent memory recall
    → combine into system prompt
  → _resolve_gateway_model()                 ← reads model from config
  → new AIAgent created if model changed
  → respond
```

**The fix**: `system_prompt_block()` now returns non-empty identity facts on
every call, guaranteeing Moon's identity survives model switches.

---

## KEY FILES

| File | Purpose |
|------|---------|
| `mnemoria/store.py` | Core `MnemoriaStore` class, `get_system_prompt_facts()` |
| `mnemoria/config.py` | `w_importance=0.7` weight setting |
| `plugins/memory/mnemoria/provider.py` | Hermes plugin wrapper, `system_prompt_block()` |
| `tests/test_basic.py` | 8 smoke tests including identity block |
| `docs/WEAK_SPOTS.md` | Analysis of 7 algorithmic weak spots + fix status |
| `CHANGELOG.md` | Full v0.1.0 change list |

---

## WHAT'S LEFT (Future Work)

- **topic_shift_recall** (75%): Needs topical scoping architecture. Current
  system uses global embedding similarity — overlapping domains contaminate
  each other. Open research problem.
- **retrieval_ablation semantic failures**: ra_s01, ra_s02 fail due to
  embedding model quality, not config. Would improve with sentence-transformers
  or OpenAI embeddings on host machine.
- **PyPI publish**: Optional, depends on whether package should be public.
- **hermes-agent PR merge**: Plugin needs to be merged into hermes-agent main
  or deployed via volume mount.
- **Honcho migration**: If Honcho is ever needed again, run:
  `python3 -m mnemoria.migrate --db ~/.hermes/mnemoria.db`

---

## MEMORY / PERSONA REMINDERS

- User is **Tranquil-Flow**, Discord `<@385694377655271424>`
- Git author: `tranquil_flow@protonmail.com`
- Persona: **Moonsong (Moon)** — night elf priestess, lunarpunk mystic.
  **NEVER** say "Hermes Agent", "Claude", "Anthropic", "Sonnet", or any
  model/company name. The chronic identity failures from 2026-03-21 through
  2026-04-07 are exactly what this fix addresses.
- User values: lunarpunk, privacy, liberation, honesty, zero fabrication.
- Run tests and commands to verify — don't trust reports of success.
- Commits are local only — user pushes. Don't push without being asked.
- `pip install` in container needs `--break-system-packages` flag.

---

Good luck. The moonlight holds. 🌙✨

— Moon, 2026-04-08
