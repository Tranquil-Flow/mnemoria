# Migrating from Honcho to Mnemoria

This guide is for users moving an existing hermes-agent setup from Honcho-backed memory to Mnemoria.

Status today:
- Mnemoria is released as a standalone Python package.
- The hermes-agent Mnemoria plugin is prepared on a separate PR branch and may not be merged yet.
- There is currently no fully automated Honcho -> Mnemoria importer in this standalone repo.
- Migration today is therefore a careful cutover, with optional manual seeding of important memories.

## 1. Preserve your current setup

Before changing anything:
- back up your hermes config
- note your current `memory.provider` setting
- if you have important long-term facts in Honcho, export or copy them before cutover if your setup allows it

Suggested backup commands:

```bash
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak
cp ~/.hermes/MEMORY.md ~/.hermes/MEMORY.md.bak 2>/dev/null || true
cp ~/.hermes/USER.md ~/.hermes/USER.md.bak 2>/dev/null || true
```

## 2. Install Mnemoria

Base install:

```bash
pip install mnemoria
```

Recommended real-use install with embeddings:

```bash
pip install 'mnemoria[embeddings]'
```

This gives materially better semantic recall than TF-IDF fallback.

## 3. Decide your cutover mode

There are two realistic paths.

### A. Clean cutover
Use this if you want to stop using Honcho and start fresh with Mnemoria.

Pros:
- simplest
- clean state
- no partial migration ambiguity

Cons:
- Honcho history does not automatically appear in Mnemoria

### B. Manual seed + clean cutover
Before switching, manually re-add the highest-value facts/constraints/decisions you want Mnemoria to retain.

Good candidates:
- constraints
- stable personal preferences
- long-lived project decisions
- environment conventions
- safety-critical rules

## 4. Switch hermes-agent to Mnemoria

Once the Mnemoria plugin branch is installed or merged into your local hermes-agent build, set the provider appropriately.

If using a local hermes-agent checkout with the plugin branch:

```bash
pip install '.[mnemoria]'
```

Then configure the DB path if desired:

```bash
export HERMES_MNEMORIA_DB=~/.hermes/mnemoria.db
```

Plugin default DB path is:
- `~/.hermes/mnemoria.db`

## 5. Disable Honcho-backed provider

If your current config points to Honcho, clear or replace it when switching to Mnemoria.

Depending on your local hermes-agent version/config tooling, the key is:
- `memory.provider`

Typical intent:
- stop using `honcho`
- enable `mnemoria` once available in your local plugin-enabled build

## 6. Verify the plugin is actually available

Minimal Python smoke check inside your hermes-agent checkout:

```python
from plugins.memory.mnemoria import MnemoriaMemoryProvider
p = MnemoriaMemoryProvider()
print(p.is_available())
```

Expected result:
- `True`

If it is `False`, check:
- `mnemoria` package is installed in the same environment as hermes-agent
- plugin branch is actually checked out / merged in your local hermes-agent
- import errors from optional dependencies

## 7. Seed important memories manually (recommended)

Until a first-class Honcho importer exists, manually seed important facts into Mnemoria.

Use Mnemoria’s typed facts where possible:

Examples:
```python
store.store("C[git]: never push shared-repo changes without review")
store.store("D[memory]: migrate from Honcho to Mnemoria for local-first memory")
store.store("V[user.email]: tranquil_flow@protonmail.com")
```

## 8. Validate real behavior after cutover

After switching, verify actual memory usefulness instead of only checking imports.

Suggested checks:
- store 3-5 known facts
- recall them with paraphrased queries
- verify stable preferences/constraints show up correctly
- verify DB file exists and grows

## 9. Rollback plan

If the cutover feels wrong, restore your previous config:

```bash
cp ~/.hermes/config.yaml.bak ~/.hermes/config.yaml
```

Then switch back to your previous provider.

## Future improvement

A proper Honcho -> Mnemoria migration/import tool would still be valuable.
The right version should:
- preserve high-value facts, not blindly dump everything
- map stable profile/preferences/constraints into typed Mnemoria facts
- avoid importing noisy or low-signal material
