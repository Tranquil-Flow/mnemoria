# Acknowledgements

Mnemoria was developed in dialogue with prior work. These are the projects and papers we can honestly credit as real inspirations for the current design or evaluation approach.

## Direct inspirations we actually used

### Honcho
- https://honcho.dev/

Honcho influenced the packaging/integration shape more than the retrieval design itself.
Mnemoria was deliberately split into a standalone package plus a thin hermes-agent plugin wrapper, following the general external-provider pattern rather than inlining everything into hermes-agent.

### Icarus-Daedalus
- https://github.com/esaradev/icarus-daedalus

Inspired lifecycle-oriented memory ideas, especially:
- topic-change / session-hook style thinking
- decision extraction concepts
- the broader idea that memory should operate proactively, not only as passive recall

### PLUR
- https://github.com/plur-ai/plur

Influenced parts of the forward-looking memory design discussion, especially:
- meta-memory / abstraction ideas
- richer memory lifecycle thinking
- bias and retrieval-quality considerations

### fff.nvim
- https://github.com/dmtrKovalenko/fff.nvim

A lighter inspiration than the systems above, but it informed some thinking around recency/frequency tradeoffs and pragmatic memory ergonomics.

### Omni-SimpleMem / SimpleMem
- Paper: https://arxiv.org/pdf/2604.01007
- Repo: https://github.com/aiming-lab/SimpleMem

This work directly influenced benchmark thinking and retrieval analysis.
In particular, it shaped:
- retrieval-ablation style evaluation
- format-sensitivity evaluation
- timestamp-integrity concerns
- several retrieval-fusion discussions explored during development

## Not currently listed as confirmed inspirations

The following may be interesting or adjacent, but we do not currently have strong enough evidence in the Mnemoria work history to claim them here as real inspirations for the current released design:
- https://github.com/vacui-dev/synsets
- https://github.com/NousResearch/hermes-agent/pull/727
- https://github.com/agiresearch/A-mem

If later work clearly incorporates ideas from them, they should absolutely be added with a precise note about what they contributed.

## Philosophy of credit

This file is intentionally conservative.
We would rather under-credit than pretend a project shaped Mnemoria when it did not.
When a project or paper genuinely informed the design, evaluation, architecture, or workflow, we should say so plainly and specifically.
