# Procedural Memory Research

Research date: 2026-04-23
Mnemoria version: v0.2.1 (0.927 benchmark)
Context: Gap identified vs Mem0, which explicitly markets "procedural memory" as a feature

## What Is Procedural Memory?

### Cognitive Science Definition

In cognitive psychology (Tulving, 1972; Squire, 1987), memory is divided into:

| Type | Definition | Human Example |
|------|-----------|---------------|
| **Declarative (explicit)** | Facts and events you can consciously recall | "Paris is the capital of France" |
| -- Episodic | Personal experiences, time-stamped | "I visited Paris in 2024" |
| -- Semantic | General knowledge, context-free | "Paris has the Eiffel Tower" |
| **Procedural (implicit)** | Skills, habits, motor routines you perform without conscious recall | Riding a bicycle, typing on a keyboard |

Procedural memory is "knowing how" vs declarative memory's "knowing that." It is:
- Acquired through practice and repetition
- Difficult to articulate verbally (try explaining exactly how you ride a bicycle)
- Resistant to forgetting (once learned, persists for years)
- Executed automatically without conscious effort
- Refined through error-correction feedback

### ACT-R's Production Rules

ACT-R (Anderson, 2007) -- the cognitive architecture Mnemoria is partly based on -- models procedural memory as **production rules**:

```
IF [condition pattern matches working memory]
THEN [execute action sequence]
```

Productions are:
- Condition-action pairs stored in **production memory** (separate from declarative memory)
- Selected by a **conflict resolution** mechanism when multiple productions match
- Strengthened by **utility learning**: productions that lead to goal achievement get higher utility scores
- Subject to **production compilation**: repeated sequences of productions are compiled into single, faster productions

Example ACT-R production for "how to commit code":

```lisp
(p commit-code
   =goal>
      ISA task
      state ready-to-commit
   =retrieval>
      ISA procedure
      steps ("stage" "commit" "push")
   ==>
   =goal>
      state executing-commit
   +action>
      command "git add -A && git commit -m 'message' && git push"
)
```

### Key Distinction for AI Agents

For AI agents, procedural memory is about **learned behavioral patterns**:
- "When the user asks about deployment, check the deployment runbook first"
- "When tests fail, run them individually to isolate the failure"
- "When writing Python, use type hints and follow PEP 8"
- "When debugging memory issues, check gc.get_objects() and tracemalloc"

These are not just facts (declarative) -- they are **condition-action patterns** that should trigger automatically when the right context appears.

## How Mem0 Implements Procedural Memory

### Mem0's Memory Taxonomy

Mem0 (ECAI 2025 paper, github.com/mem0ai/mem0) organizes memory into four categories:

| Category | Description | Example |
|----------|-------------|---------|
| **Episodic** | Past interactions and experiences | "User reported a bug in v2.1" |
| **Semantic** | Facts and relationships | "Project uses PostgreSQL 15" |
| **Procedural** | Preferred processes and workflows | "User prefers to review PRs before merging" |
| **Emotional/Social** | Emotional context and relationships | "User gets frustrated with slow CI" |

### Mem0's Procedural Memory Implementation

From Mem0's codebase (mem0/memory/main.py and related files), procedural memory is stored as regular memory entries but **tagged with a procedural category** and given special treatment:

1. **Extraction**: LLM-based extraction specifically looks for process descriptions, workflows, preferences about how things should be done
2. **Storage**: Stored as regular memories with `category="procedural"` metadata
3. **Retrieval boost**: Procedural memories get a relevance boost when the query context matches their trigger condition
4. **Persistence**: Procedural memories are marked as "long-term" -- they resist pruning and have slow decay

The actual implementation is simpler than the marketing suggests. Mem0 does not have true production rules or skill acquisition. It stores workflow descriptions as text facts with a procedural label.

### Mem0's Extraction Prompt (reconstructed from code)

```
Extract memories from the conversation. Categorize each as:
- episodic: specific events or interactions
- semantic: facts, preferences, attributes  
- procedural: processes, workflows, how the user likes things done
- emotional: feelings, frustrations, relationship dynamics

For procedural memories, capture:
- The trigger condition (when/what situation)
- The preferred action or process
- Any constraints or preferences about execution
```

### What Mem0 Gets Right

1. **Explicit category**: Having a "procedural" category surfaces these memories in the right context
2. **Slow decay**: Procedural memories are stable -- once you learn someone prefers `ruff` over `flake8`, that persists
3. **Trigger matching**: Querying "how should I lint the code?" retrieves procedural memories about linting preferences

### What Mem0 Gets Wrong

1. **No true production rules**: Mem0's procedural memory is just declarative facts with a label. It cannot execute procedures or learn from execution outcomes.
2. **No skill compilation**: Repeated procedures are not compiled into more efficient representations.
3. **No feedback loop**: Mem0 cannot learn that a procedure worked or failed. There is no reward signal for procedural memories.

## Could Mnemoria's D-Type Facts Become Procedural Memory?

### Current D-Type (Decision) Facts

Mnemoria's `FactType.DECISION` (D) facts represent past architectural/design decisions:

```
D[auth]: Switched from session cookies to JWT tokens
D[deploy]: Always deploy to staging before production
D[testing]: Use pytest with --no-header flag
```

These are **declarative statements about decisions**, not executable procedures. But many D-type facts implicitly encode procedures:

| D-Type Fact | Implicit Procedure |
|---|---|
| "D[deploy]: Always deploy to staging before production" | IF deploying THEN deploy-to-staging BEFORE deploy-to-prod |
| "D[testing]: Use pytest with --no-header flag" | IF running-tests THEN command = "pytest --no-header" |
| "D[code-style]: Use ruff instead of flake8" | IF linting THEN tool = ruff |

### Proposed: P-Type (Procedural) Facts

Add a new fact type to Mnemoria's type system:

```python
class FactType(Enum):
    CONSTRAINT = 'C'   # Rules that must be followed
    DECISION   = 'D'   # Past decisions that bind future behavior
    VALUE      = 'V'   # Concrete values, settings, facts
    PROCEDURE  = 'P'   # Learned behavioral patterns (NEW)
    UNKNOWN    = '?'   # Open questions
    DONE       = 'done'
    OBSOLETE   = 'obs'
```

P-type facts have structured fields:

```
P[deploy]: WHEN deploying -> deploy to staging first, verify, then deploy to production
P[debug.memory]: WHEN memory usage high -> check gc.get_objects(), run tracemalloc
P[code.lint]: WHEN writing Python -> run ruff check before committing
```

### Schema Extension

```python
# In types.py, add to METABOLIC_RATES:
METABOLIC_RATES = {
    ...
    FactType.PROCEDURE: 0.2,  # Very slow decay (procedures are stable once learned)
}

# In FACT_TYPE_FROM_NOTATION:
FACT_TYPE_FROM_NOTATION = {
    ...
    'P': FactType.PROCEDURE,
}

# In NOTATION_PATTERN, add P to the alternation:
NOTATION_PATTERN = re.compile(r'^(C|D|V|P|\?|✓|~)\[([^\]]+)\]:\s*(.+)$')
```

### Structured Procedural Fact Storage

```sql
-- Optional structured fields for P-type facts
CREATE TABLE um_procedures (
    fact_id TEXT PRIMARY KEY REFERENCES um_facts(id),
    trigger_pattern TEXT,       -- "WHEN deploying", "WHEN tests fail"
    action_template TEXT,       -- "deploy to staging first, then prod"
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_executed REAL,
    avg_reward REAL DEFAULT 0.5,
    FOREIGN KEY (fact_id) REFERENCES um_facts(id)
);
```

This enables tracking procedure execution outcomes and refining the procedure over time.

## ACT-R Production Rule Model for Mnemoria

### Production Compilation

ACT-R's production compilation merges frequently co-occurring productions into single, faster productions. For Mnemoria, this means detecting when the same sequence of facts is repeatedly retrieved together and creating a compiled procedural fact:

```python
def detect_procedural_patterns(conn, min_co_retrievals=5):
    """Find fact sequences that are repeatedly co-retrieved."""
    # Query co-occurrence counts from um_links
    patterns = conn.execute("""
        SELECT l.source_id, l.target_id, l.co_occurrence_count,
               f1.content as source_content, f2.content as target_content,
               f1.type as source_type, f2.type as target_type
        FROM um_links l
        JOIN um_facts f1 ON l.source_id = f1.id
        JOIN um_facts f2 ON l.target_id = f2.id
        WHERE l.co_occurrence_count >= ?
          AND f1.status = 'active' AND f2.status = 'active'
        ORDER BY l.co_occurrence_count DESC
    """, (min_co_retrievals,)).fetchall()
    
    return patterns
```

When a pair of facts is co-retrieved 5+ times, consider compiling them into a single procedural fact that captures the combined knowledge.

### Utility Learning

ACT-R uses utility learning to select among competing productions. Mnemoria already has Q-value learning (Phase 4) that serves a similar purpose -- facts that lead to positive outcomes (store-after-recall reward signal) get higher Q-values.

Extending this to procedures:

```python
def update_procedure_utility(conn, fact_id, reward_signal):
    """Update a procedure's utility based on execution outcome."""
    conn.execute("""
        UPDATE um_procedures
        SET avg_reward = avg_reward * 0.9 + ? * 0.1,
            success_count = success_count + CASE WHEN ? > 0.5 THEN 1 ELSE 0 END,
            failure_count = failure_count + CASE WHEN ? <= 0.5 THEN 1 ELSE 0 END,
            last_executed = ?
        WHERE fact_id = ?
    """, (reward_signal, reward_signal, reward_signal, now, fact_id))
```

### Conflict Resolution

When multiple procedures match a query, select based on:
1. **Utility score**: avg_reward * success_rate
2. **Recency**: last_executed freshness
3. **Specificity**: More specific trigger patterns take precedence

This mirrors ACT-R's conflict resolution:

```python
def select_procedure(candidates):
    """Select best procedure using ACT-R-inspired conflict resolution."""
    for proc in candidates:
        # Utility = expected reward * success_rate - cost
        success_rate = proc.success_count / max(proc.success_count + proc.failure_count, 1)
        utility = proc.avg_reward * success_rate
        
        # Specificity bonus: longer trigger patterns are more specific
        specificity = len(proc.trigger_pattern.split()) / 10.0
        
        proc.selection_score = utility + specificity * 0.2
    
    return max(candidates, key=lambda p: p.selection_score)
```

## Skill Acquisition Models from Cognitive Science

### Fitts's Three-Stage Model (1964)

Skill acquisition progresses through:

1. **Cognitive stage**: Learner consciously processes each step. Slow, error-prone. (Mnemoria equivalent: V-type and D-type facts storing declarative knowledge about a procedure)
2. **Associative stage**: Steps become linked through practice. Speed increases, errors decrease. (Mnemoria equivalent: Hebbian links form between co-retrieved procedural facts)
3. **Autonomous stage**: Execution becomes automatic. Little conscious attention needed. (Mnemoria equivalent: compiled P-type fact with high utility score)

### Power Law of Practice (Newell & Rosenbloom, 1981)

Execution time decreases as a power function of practice: T = a * N^(-b). For Mnemoria, this maps to:
- Initial recall of a procedure: slow, requires multiple fact retrievals
- After 5+ co-retrievals: facts are linked, retrieval is faster (Hebbian spreading activation)
- After 10+ co-retrievals: procedure is compiled, single P-type fact retrieved directly

### Logan's Instance Theory (1988)

With practice, retrieval transitions from algorithm-based to memory-based. Instead of computing the answer, the system retrieves a past instance. For Mnemoria:
- Early: Agent reasons from individual V/D/C facts
- Later: Agent retrieves a compiled procedure that directly answers "how to X"

### Implementation of Skill Acquisition

```python
class SkillTracker:
    """Track skill acquisition from declarative to procedural."""
    
    # Stage thresholds
    COGNITIVE_THRESHOLD = 0     # New skill, declarative facts only
    ASSOCIATIVE_THRESHOLD = 3   # Co-retrieved 3+ times, links forming
    AUTONOMOUS_THRESHOLD = 8    # Co-retrieved 8+ times, compile to P-type
    
    def check_compilation(self, conn, link_map):
        """Check if any fact clusters should be compiled into procedures."""
        # Find fact clusters with high co-occurrence
        clusters = self._find_dense_clusters(conn, link_map)
        
        for cluster in clusters:
            min_co_occurrence = min(
                link_map.get((a, b), {}).get('co_occurrence_count', 0)
                for a in cluster for b in cluster if a != b
            )
            
            if min_co_occurrence >= self.AUTONOMOUS_THRESHOLD:
                # Compile cluster into a P-type procedural fact
                self._compile_procedure(conn, cluster)
    
    def _compile_procedure(self, conn, fact_ids):
        """Merge a cluster of related facts into a single P-type fact."""
        facts = [self._get_fact(conn, fid) for fid in fact_ids]
        
        # Synthesize procedure content from constituent facts
        content = self._synthesize_procedure(facts)
        trigger = self._extract_trigger(facts)
        
        # Store as P-type
        # The compiled fact inherits activation from its constituents
        total_activation = sum(f.activation for f in facts)
        
        return content, trigger, total_activation
```

## Comparison with Mnemoria's Current Capabilities

### What Mnemoria Already Has

| Procedural Capability | Current Implementation | Gap |
|---|---|---|
| Storing process descriptions | V/D-type facts with "procedural" category | No special type or structure |
| Trigger matching | Intent classification (procedural intent) | No condition-action pairing |
| Slow decay | D-type metabolic rate 0.7x | Could be slower for procedures |
| Co-retrieval tracking | Hebbian co-occurrence counts | No compilation step |
| Feedback learning | Q-value rewards from store-after-recall | Not procedure-specific |
| Skill stages | None | No progression tracking |
| Category tagging | `encoding.py` classifies "procedural" | Label only, no behavior change |

### Mnemoria's encoding.py Already Detects Procedural Facts

```python
# From encoding.py
("procedural", [
    re.compile(r"\b(run|execute|install|command|script|steps?|how to|to do|procedure|workflow|recipe|guide)\b", re.I),
    re.compile(r"\b(pip|npm|apt|brew|docker|git|make|cargo|go build)\s", re.I),
    re.compile(r"```", re.I),  # code blocks
    re.compile(r"\b(first|then|next|finally|step \d)\b", re.I),
]),
```

This classification already tags facts as "procedural" but doesn't affect retrieval behavior beyond the resolution boost (which boosts "procedural" and "causal" categories in the dampening pipeline).

## Recommended Implementation Plan

### Phase 1: Formal P-Type Facts (v0.3.0)

1. Add `FactType.PROCEDURE = 'P'` to the type system
2. Set metabolic rate to 0.2 (slower than constraints at 0.3)
3. Add `P` to notation pattern: `P[deploy]: WHEN deploying -> stage first, then prod`
4. Recognition in `encoding.py`: upgrade "procedural" category classification to emit P-type when clear trigger-action structure detected

**No schema changes needed** -- P-type facts use existing `um_facts` table with `type = 'P'`.

### Phase 2: Procedure-Specific Retrieval (v0.3.x)

1. Intent classification: when query is procedural ("how do I...", "what's the process for..."), boost P-type facts by 1.25x (stronger than current 1.15x intent boost)
2. In `get_system_prompt_facts()`: include P-type facts alongside C and D types (procedures should always be in context)
3. P-type facts get resolution boost in dampening pipeline

### Phase 3: Procedure Compilation (v0.4.0)

1. Add `um_procedures` table with trigger_pattern, success/failure counts
2. During consolidation, scan for fact clusters with co_occurrence >= 8
3. Compile clusters into P-type facts (optionally with LLM assistance to synthesize content)
4. Mark constituent facts as `~` (obsolete) after compilation

### Phase 4: Utility Learning (v0.5.0)

1. Track procedure execution outcomes via reward signals
2. Update procedure utility scores (avg_reward * success_rate)
3. Use utility for conflict resolution when multiple procedures match
4. Implement ACT-R-style production selection

## Expected Impact

| Area | Impact |
|---|---|
| Mem0 feature parity | Direct parity on "procedural memory" marketing claim |
| Benchmark | Minimal direct benchmark impact (current tests don't specifically test procedures) |
| Agent effectiveness | Significant: agents retain learned workflows across sessions |
| User experience | Procedures surfaced in system prompt -> consistent agent behavior |
| Cognitive authenticity | Closer to ACT-R's full architecture (declarative + procedural + production compilation) |

## Future Research Directions

1. **LLM-assisted compilation**: Use the agent's LLM to synthesize procedure descriptions from co-retrieved fact clusters
2. **Procedure chaining**: Model multi-step procedures as linked P-type facts (procedure graphs)
3. **Transfer learning**: Procedures learned in one scope may apply to related scopes
4. **User-defined procedures**: Allow users to explicitly create P-type facts via notation: `P[deploy]: always run tests before deploying`
5. **Procedure versioning**: Track procedure evolution over time via temporal validity fields
