# Procedural Memory Spec

**Status:** Draft
**Date:** 2026-04-23
**Scope:** New fact type `P`, `ProceduralObserver`, retrieval changes, system prompt integration


## 1. What procedural memory means for an AI agent

In ACT-R, procedural memory stores **production rules** -- condition-action pairs
that encode "how to do things." Declarative memory (facts, decisions) says *what*;
procedural memory says *how*.

For an AI agent backed by Mnemoria, procedural memory captures:

- **Workflows**: "To deploy, first run lint, then tests, then push to staging."
- **Preferences-in-action**: "When writing tests, always use pytest fixtures over manual setup."
- **Tool usage sequences**: "For database migrations, run `alembic upgrade head` then verify with `alembic current`."
- **Coding patterns**: "In this codebase, error handlers return a tuple of (error_code, message)."
- **Conditional behaviors**: "When the CI fails on type checks, run `mypy --strict` locally first."
- **Recovery procedures**: "If the build cache is stale, delete `.cache/` and re-run `npm install`."

These differ from D-type decisions ("we chose PostgreSQL over MySQL") in that
procedural facts encode *executable sequences*, not *resolved choices*. A decision
is a settled point; a procedure is a reusable path.


## 2. New fact type: P

### Why not extend D-type?

D-type (Decision) facts represent settled choices: "we use PostgreSQL", "the API
version is v3." They have a metabolic rate of 0.7 and are surfaced in system
prompts alongside C-type constraints.

Procedural knowledge is structurally different:

| Property | D (Decision) | P (Procedural) |
|---|---|---|
| Shape | Single assertion | Condition-action pair or ordered sequence |
| Supersession | New decision replaces old | Procedures evolve (steps added/removed) |
| Decay | Moderate (0.7) | Slow (skills persist) |
| Retrieval boost | When decision-intent queries fire | When the agent is about to act |
| System prompt role | Binding constraints | Available playbooks |

Overloading D-type would conflate "what we decided" with "how to do it," breaking
the intent classifier's ability to route queries and the system prompt's ability to
prioritize constraint/decision facts over procedural ones.

### Type definition

Add to `mnemoria/types.py`:

```python
class FactType(Enum):
    CONSTRAINT = 'C'
    DECISION   = 'D'
    VALUE      = 'V'
    UNKNOWN    = '?'
    DONE       = 'done'
    OBSOLETE   = 'obs'
    PROCEDURAL = 'P'       # NEW
```

Notation: `P[target]: content`

Examples:
- `P[deploy]: run lint, then tests, then push to staging branch`
- `P[git.rebase]: always rebase feature branches onto main before opening PR`
- `P[error.recovery]: if tests fail with ImportError, check virtualenv activation first`

### Notation pattern update

Update `NOTATION_PATTERN` and `FACT_TYPE_FROM_NOTATION` in `types.py`:

```python
FACT_TYPE_FROM_NOTATION: dict[str, FactType] = {
    'C': FactType.CONSTRAINT,
    'D': FactType.DECISION,
    'V': FactType.VALUE,
    '?': FactType.UNKNOWN,
    'done': FactType.DONE,
    'obs': FactType.OBSOLETE,
    'P': FactType.PROCEDURAL,      # NEW
}
```

Update `NOTATION_PATTERN` regex to include `P`:

```python
NOTATION_PATTERN = re.compile(r'^(C|D|V|P|\?|\u2713|~)\[([^\]]+)\]:\s*(.+)$')
```


## 3. Metabolic rate

Procedural knowledge decays slower than declarative facts. Skills, once learned,
persist longer than raw observations. ACT-R models this: production rules
strengthen with use and weaken slowly.

```python
METABOLIC_RATES: dict[FactType, float] = {
    FactType.CONSTRAINT:  0.3,
    FactType.DECISION:    0.7,
    FactType.PROCEDURAL:  0.5,    # NEW -- between constraint and decision
    FactType.VALUE:       1.0,
    FactType.UNKNOWN:     2.0,
    FactType.DONE:        2.5,
    FactType.OBSOLETE:    5.0,
}
```

Rationale for 0.5:
- Slower than D (0.7): procedures are reusable skills, not one-off choices.
- Faster than C (0.3): constraints are absolute rules; procedures can evolve.
- With `d=0.3` (default ACT-R decay) and metabolic_rate=0.5, effective decay
  exponent is 0.15. After 7 days without access, a P-fact retains ~60% of its
  base-level activation vs ~45% for a V-fact.


## 4. Extraction: ProceduralObserver

### Where it lives

`mnemoria/observers/procedural.py`

### What triggers it

The observer watches `user_message` and `agent_message` events for patterns that
indicate procedural knowledge being stated, taught, or discovered.

### Extraction patterns

#### 4.1 Explicit procedural statements (user_message)

Trigger phrases that signal the user is teaching a procedure:

```
"always do X before Y"
"the way to do X is..."
"when X happens, do Y"
"to [verb], first... then..."
"the process for X is..."
"the workflow is..."
"make sure to X before Y"
"never do X without first doing Y"
"the correct order is..."
"steps to X: ..."
"if X, then do Y"
"whenever X, you should Y"
```

Regex patterns (preliminary):

```python
PROCEDURAL_PATTERNS = [
    # "always do X before Y"
    re.compile(
        r"\balways\s+(?:do\s+)?(.+?)\s+before\s+(.+)",
        re.IGNORECASE
    ),
    # "the way to X is Y" / "the process for X is Y"
    re.compile(
        r"\bthe\s+(?:way|process|procedure|workflow|steps?)\s+(?:to|for)\s+(.+?)\s+is\s+(.+)",
        re.IGNORECASE
    ),
    # "when X happens, do Y" / "whenever X, do Y"
    re.compile(
        r"\b(?:when|whenever|if)\s+(.+?),\s*(?:do|run|execute|use|try|you\s+should)\s+(.+)",
        re.IGNORECASE
    ),
    # "to [verb], first... then..."
    re.compile(
        r"\bto\s+(\w+.*?),\s*first\s+(.+?)(?:\s*,?\s*then\s+(.+))?$",
        re.IGNORECASE
    ),
    # "make sure to X before Y" / "never X without Y"
    re.compile(
        r"\b(?:make\s+sure\s+to|never)\s+(.+?)\s+(?:before|without\s+first)\s+(.+)",
        re.IGNORECASE
    ),
    # "if X fails/errors, then Y"
    re.compile(
        r"\bif\s+(.+?)\s+(?:fails?|errors?|breaks?|crashes?),?\s*(?:then\s+)?(.+)",
        re.IGNORECASE
    ),
    # "steps to X: ..." (numbered or bulleted)
    re.compile(
        r"\bsteps?\s+(?:to|for)\s+(.+?):\s*(.+)",
        re.IGNORECASE
    ),
]
```

#### 4.2 Agent-discovered procedures (agent_message)

The agent may articulate a procedure it discovered while working. These are
lower-confidence and should go through `agent_inference` source (subject to
session TTL before promotion):

```python
AGENT_PROCEDURAL_PATTERNS = [
    # "I found that the way to X is Y"
    re.compile(
        r"\bI\s+(?:found|discovered|learned|noticed)\s+that\s+(?:the\s+)?(?:way|process|procedure)\s+(?:to|for)\s+(.+?)\s+is\s+(.+)",
        re.IGNORECASE
    ),
    # "the fix for X is to Y"
    re.compile(
        r"\bthe\s+(?:fix|solution|workaround|resolution)\s+(?:for|to)\s+(.+?)\s+is\s+(?:to\s+)?(.+)",
        re.IGNORECASE
    ),
]
```

#### 4.3 Tool-sequence inference (tool_result)

When the agent successfully executes a sequence of tools to accomplish a task
(e.g., lint -> test -> deploy), the observer can infer a procedure. This is the
most speculative extraction path and uses `agent_inference` source.

Implementation: track tool sequences within a session. When a sequence of 2+
tools completes successfully and the user confirms the result (next user message
is positive/accepting), emit a P-fact.

This is deferred to Phase 2 of procedural memory -- the initial implementation
covers only explicit statement extraction (4.1) and agent-discovered (4.2).

### Conflict with UserStatementObserver

The existing `UserStatementObserver` catches "I always..." and "I prefer..."
patterns. These overlap with procedural triggers like "always do X before Y."

Resolution: `ProceduralObserver` runs **before** `UserStatementObserver`. If a
procedural pattern matches, the event is consumed (the observer returns facts).
The `__init__.py` observer ordering already controls dispatch priority:

```python
def all_observers() -> list:
    return [
        ProceduralObserver(),        # NEW -- must come before UserStatement
        UserStatementObserver(),
        UserContentObserver(),
        ...
    ]
```

However, observers currently all run independently on every event (no
short-circuit). To prevent double-extraction, `ProceduralObserver` should set a
flag in the event dict (`event["_procedural_matched"] = True`) that
`UserStatementObserver` checks:

```python
# In UserStatementObserver.observe():
if event.get("_procedural_matched"):
    return []
```

Alternative (simpler): accept occasional overlap. The dedup logic in
`store.store()` (source_hash + semantic dedup) will catch near-duplicates. The
procedural fact will have type P and the preference fact will have type V; both
survive but serve different retrieval paths. This is acceptable for v1.


### Observer class

```python
class ProceduralObserver:
    """Extract procedural knowledge (condition-action pairs, workflows, sequences)."""

    name: str = "procedural"

    def observe(self, event: dict) -> list[PendingFact]:
        kind = event.get("kind")
        if kind not in ("user_message", "agent_message"):
            return []

        content = event.get("payload", {}).get("content", "").strip()
        if not content or len(content) < 15:
            return []

        session_id = event.get("session_id", "")

        # Determine source based on event kind
        if kind == "user_message":
            source = "user_stated"
            patterns = PROCEDURAL_PATTERNS
        else:
            source = "agent_inference"
            patterns = AGENT_PROCEDURAL_PATTERNS

        for pattern in patterns:
            m = pattern.search(content)
            if m:
                procedure_text = self._format_procedure(m, content)
                target = self._infer_target(content)
                return [PendingFact(
                    content=procedure_text,
                    type="P",
                    target=target,
                    source=source,
                    provenance={
                        "extractor": self.name,
                        "pattern": pattern.pattern[:60],
                        "sentence": content[:200],
                        "session_id": session_id,
                    },
                )]

        return []

    def _format_procedure(self, match, original: str) -> str:
        """Format the matched groups into a condition-action procedural fact."""
        groups = [g for g in match.groups() if g]
        if len(groups) >= 2:
            return f"when {groups[0].strip().rstrip('.')}, do {groups[1].strip().rstrip('.')}"
        return original.strip()[:200]

    def _infer_target(self, content: str) -> str:
        """Extract a target namespace from the procedural statement."""
        # Check for tool mentions
        tool_re = re.compile(
            r"\b(pytest|git|pip|npm|docker|kubectl|terraform|make|"
            r"alembic|cargo|yarn|pnpm|bun|deno|go|rustc|mypy|ruff|"
            r"eslint|prettier|black|isort)\b",
            re.IGNORECASE,
        )
        m = tool_re.search(content)
        if m:
            return m.group(1).lower()

        # Check for domain terms
        domain_re = re.compile(
            r"\b(deploy|test|build|migrate|release|review|debug|"
            r"refactor|rollback|backup|restore|setup|install|configure)\b",
            re.IGNORECASE,
        )
        m = domain_re.search(content)
        if m:
            return m.group(1).lower()

        return "workflow"
```


## 5. Retrieval changes

### 5.1 Resolution boost

The dampening pipeline in `retrieval.py` already boosts `BOOST_CATEGORIES` and
`BOOST_TYPES`. Add `FactType.PROCEDURAL`:

```python
BOOST_TYPES = {FactType.CONSTRAINT, FactType.DECISION, FactType.PROCEDURAL}
```

This gives P-facts a 1.25x score multiplier when they appear in results.

### 5.2 Intent-based type boost

The intent classifier in `intent.py` already classifies `PROCEDURAL` queries
("how do I...", "steps to..."). Currently it has no `type_boost` mapping.

Update `QueryIntent.type_boost`:

```python
@property
def type_boost(self) -> Optional[FactType]:
    return {
        self.VALUE: FactType.VALUE,
        self.CONSTRAINT: FactType.CONSTRAINT,
        self.DECISION: FactType.DECISION,
        self.PROCEDURAL: FactType.PROCEDURAL,    # NEW
    }.get(self.intent)
```

When a user asks "how do I deploy?", the intent classifier returns PROCEDURAL,
which boosts P-type facts by 1.15x on top of the resolution boost. Combined
effect: P-facts get up to 1.25 * 1.15 = 1.44x total boost for procedural queries.

### 5.3 Answer-shape heuristic

Add a heuristic in `score_candidates()` that boosts P-facts when the query
contains action-oriented language:

```python
# Procedural facts get a boost for "how" queries
if c.get("type") == "P" and query_lower.startswith("how "):
    answer_shape_boost += 0.25
```

### 5.4 Contextual retrieval boost (future Phase 2)

When the agent is about to invoke a tool (e.g., the pipeline detects a `tool_call`
event for `git`), automatically recall P-facts with `target=git`. This is a
pre-action retrieval step:

```python
def recall_procedures_for_tool(self, tool_name: str, top_k: int = 3) -> List[ScoredFact]:
    """Recall procedural facts relevant to an upcoming tool invocation."""
    return self.recall(
        query=f"procedure for {tool_name}",
        scope=None,
        top_k=top_k,
    )
```

This is deferred to Phase 2 -- it requires integration with the agent's tool
dispatch loop, which is outside Mnemoria's current scope.


## 6. ACT-R production rules analogy

ACT-R's procedural memory stores **production rules** of the form:

```
IF (condition matches working memory)
THEN (fire action, modify working memory)
```

Mnemoria's P-facts approximate this:

| ACT-R concept | Mnemoria equivalent |
|---|---|
| Production rule | P-type fact with condition-action content |
| Condition matching | Embedding similarity + FTS5 keyword match against the current context |
| Utility (production selection) | Activation score (base-level + spreading + importance) |
| Utility learning | Q-value reranking (reinforcement from store-after-recall) |
| Production compilation | Future: merge frequently co-recalled P-facts into composite procedures |

The key difference: ACT-R production rules fire automatically when conditions
match. Mnemoria P-facts are surfaced via retrieval and presented to the LLM, which
decides whether to follow them. This is by design -- the LLM is the executive, and
Mnemoria is the memory substrate.

### Condition-action encoding

P-facts should be stored in natural language but with a parseable structure:

```
P[deploy]: when pushing to main, first run full test suite then create release tag
P[error.recovery]: if ImportError in tests, check that virtualenv is activated
P[git]: when rebasing, always use --autostash to preserve local changes
```

The `when/if ... , do/then ...` structure maps to ACT-R's IF-THEN. The target
namespace (`deploy`, `error.recovery`, `git`) maps to the production's condition
category, enabling FTS5 to find relevant procedures quickly.


## 7. Interaction with existing types

### D-type decisions vs P-type procedures

| Signal | Classify as D | Classify as P |
|---|---|---|
| "We decided to use PostgreSQL" | Yes | No |
| "Always run migrations before deploying" | No | Yes |
| "We chose to deploy on Fridays" | Yes (decision) | No |
| "The deploy process is: lint, test, push" | No | Yes |
| "From now on, use black for formatting" | Both -- D for the choice, P for the enforcement | |

Edge case: "From now on, use black for formatting" is both a decision (choosing
black) and a procedure (apply black). The `ProceduralObserver` should extract the
P-fact; the `UserStatementObserver` may also extract a C-fact. Both are valid.
Dedup prevents exact duplicates; having both types improves retrieval coverage.

### C-type constraints vs P-type procedures

C-type: "Never commit directly to main" (a rule).
P-type: "When committing, create a feature branch first, then open a PR" (how to follow the rule).

Constraints define boundaries; procedures describe paths within those boundaries.
A constraint may imply a procedure, but the procedure encodes the steps.

### V-type values

V-type facts are raw observations with no semantic structure. A P-fact is a V-fact
that has been recognized as encoding a procedure. The `ProceduralObserver`
promotes what would otherwise be a V-fact into a P-fact with slower decay and
retrieval boosting.


## 8. System prompt integration

### Current behavior

`get_system_prompt_facts()` in `store.py` returns:
1. C-type constraints (highest priority)
2. D-type decisions
3. High-importance V-type facts
4. Identity/self-target facts

### New behavior

Add P-type procedural facts to the system prompt, positioned after D-type and
before high-importance V-type:

```python
def get_system_prompt_facts(self, session_id=None, max_facts=20):
    # 1. Constraints (C)
    # 2. Decisions (D)
    # 3. Procedures (P)  -- NEW
    # 4. High-importance values (V)
    # 5. Identity facts
```

Implementation:

```python
# After fetching C and D facts:
if len(rows) < max_facts:
    proc_rows = self._conn.execute(
        """
        SELECT * FROM um_facts
        WHERE status = 'active'
          AND type = 'P'
        ORDER BY activation DESC
        LIMIT ?
        """,
        (max_facts - len(rows),),
    ).fetchall()
    rows = list(rows) + proc_rows
```

### Formatting in system prompt

P-facts should be formatted distinctly so the LLM can distinguish them from
constraints and decisions:

```
## Active Procedures
- When pushing to main, first run full test suite then create release tag
- If ImportError in tests, check that virtualenv is activated
- When rebasing, always use --autostash to preserve local changes
```

This formatting is the responsibility of the agent framework (Hermes), not
Mnemoria itself. Mnemoria returns the facts; the agent formats them.


## 9. Supersession behavior

P-facts follow the same supersession rules as other typed facts: a new P-fact with
the same target supersedes the old one. This is correct for procedures that evolve:

```
P[deploy]: run tests then push          # v1
P[deploy]: run lint, run tests, then push  # v2 -- supersedes v1
```

The superseded fact's activation transfers to the new one at the configured ratio
(default 0.7), preserving the procedure's accumulated retrieval history.


## 10. Test plan

### Unit tests for ProceduralObserver

File: `tests/test_observer_procedural.py`

```
test_procedural_always_before
    Input: "always run lint before committing"
    Expected: P-fact with target=git or commit, content includes condition-action

test_procedural_when_then
    Input: "when the tests fail, run them locally first"
    Expected: P-fact with target=test, source=user_stated

test_procedural_steps_to
    Input: "steps to deploy: run lint, then tests, then push"
    Expected: P-fact with target=deploy

test_procedural_if_fails
    Input: "if the build fails on type checks, run mypy --strict locally"
    Expected: P-fact with target=mypy or build

test_procedural_agent_inference
    Input: agent_message "I found that the way to fix this is to clear the cache"
    Expected: P-fact with source=agent_inference

test_procedural_question_skipped
    Input: "how do I deploy?" (question)
    Expected: no P-fact emitted (questions are not procedures)

test_procedural_short_content_skipped
    Input: "do X" (< 15 chars)
    Expected: no P-fact emitted

test_procedural_no_match
    Input: "The weather is nice today"
    Expected: no P-fact emitted

test_procedural_target_inference_tool
    Input: "when using docker, always use --rm flag"
    Expected: P-fact with target=docker

test_procedural_target_inference_domain
    Input: "the deploy process requires a staging check"
    Expected: P-fact with target=deploy
```

### Integration tests

```
test_p_fact_stored_with_correct_metabolic_rate
    Store a P-fact via notation "P[deploy]: run tests first"
    Verify metabolic_rate = 0.5 in DB

test_p_fact_supersession
    Store P[deploy]: v1
    Store P[deploy]: v2
    Verify v1 is superseded, v2 is active

test_p_fact_decay_slower_than_v
    Store a P-fact and a V-fact at the same time
    Advance time by 7 days
    Recall both -- P-fact should have higher activation

test_p_fact_retrieval_boost
    Store P[deploy]: procedure and V[deploy]: random value
    Query "how do I deploy?"
    P-fact should rank higher due to intent boost + resolution boost

test_p_fact_system_prompt
    Store C, D, P, and V facts
    Call get_system_prompt_facts()
    P-fact should appear after D and before V

test_p_fact_promoted_from_pending
    Store a P-fact via store_pending() with source=user_stated
    Flush pending
    Verify the fact appears in um_facts with type=P

test_procedural_observer_in_all_observers
    Verify ProceduralObserver is in the list returned by all_observers()
```

### Benchmark regression

Add a procedural memory scenario to the benchmark suite:

```
scenario: procedural_retrieval
  setup:
    - store 20 V-facts (noise)
    - store 5 P-facts (deploy, test, build, release, debug procedures)
    - store 5 D-facts (decisions about tools)
  queries:
    - "how do I deploy?" -> expect P[deploy] in top 3
    - "what did we decide about testing?" -> expect D[test] in top 3
    - "steps to release?" -> expect P[release] in top 3
  metrics:
    - precision@3 for procedural queries
    - no regression on existing benchmarks
```


## 11. Implementation estimate

### Phase 1 (core) -- 2-3 hours

1. **types.py**: Add `PROCEDURAL = 'P'` to `FactType`, update `METABOLIC_RATES`,
   `FACT_TYPE_FROM_NOTATION`, and `NOTATION_PATTERN`. (~15 min)

2. **observers/procedural.py**: New `ProceduralObserver` class with pattern
   matching for user_message and agent_message events. (~45 min)

3. **observers/__init__.py**: Register `ProceduralObserver` in `all_observers()`
   and `__all__`. (~5 min)

4. **retrieval.py**: Add `FactType.PROCEDURAL` to `BOOST_TYPES`. (~5 min)

5. **intent.py**: Map `PROCEDURAL` intent to `FactType.PROCEDURAL` type boost.
   (~5 min)

6. **store.py**: Add P-type to `get_system_prompt_facts()` query. (~15 min)

7. **Tests**: Unit tests for `ProceduralObserver`, integration tests for
   storage/retrieval/decay. (~45 min)

### Phase 2 (contextual retrieval) -- 1-2 hours

8. **store.py**: `recall_procedures_for_tool()` method. (~30 min)

9. **observers/procedural.py**: Tool-sequence inference from `tool_result` event
   chains. (~1 hour)

10. **Benchmark scenario**: Add procedural retrieval scenario. (~30 min)

### No schema migration needed

The `um_facts.type` column is `TEXT` -- storing `'P'` requires no schema change.
The FTS5 virtual table indexes content and target, which works for P-facts. No new
tables or columns are needed.

### No breaking changes

- Existing fact types are unchanged.
- Existing observers are unchanged (no short-circuit logic in v1).
- Retrieval scoring adds P to boost sets but does not alter existing scoring.
- System prompt adds P-facts without removing any existing types.
- The notation parser accepts `P[target]: content` as a new pattern.


## 12. Comparison with Mem0 procedural memory

Mem0 v1.0 introduced procedural memory as a separate memory type with its own
storage and retrieval. Key differences from this spec:

| Aspect | Mem0 | Mnemoria (this spec) |
|---|---|---|
| Storage | Separate procedural store | Unified um_facts table with type=P |
| Decay | No decay model | ACT-R activation with metabolic_rate=0.5 |
| Retrieval | Keyword match + embedding | 4-signal fusion (ACT-R + embedding + FTS5 + Q-value) |
| Extraction | LLM-based | Rule-based patterns (no LLM call overhead) |
| Supersession | Manual | Automatic (same type+target supersedes) |
| Linking | None | Hebbian links to related facts |
| Reinforcement | None | Q-value learning from store-after-recall |

Mnemoria's approach is more deeply integrated into the cognitive architecture.
P-facts participate in the same activation-based retrieval, link formation, and
reinforcement learning as all other fact types. This means procedural knowledge
benefits from the existing consolidation, exploration (PPR), and IPS debiasing
pipelines without any special-casing.
