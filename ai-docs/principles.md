# Principles

How we approach the work. These are heuristics, not laws — but a violation should be deliberate and noted.

## Research-first

- Optimize for learning velocity. Robustness, generality, and polish come last.
- A small ugly working demo beats a clean hypothetical design.
- "What would falsify this?" before "how do we make it real?"
- Negative results count. Document them in `research/` so we don't redo dead ends.

## Brain-inspired, not brain-faithful

- We borrow *algorithmic principles* from neuroscience — sparse coding, replay, predictive errors, neuromodulation, oscillatory gating. We do *not* simulate biological neurons.
- Every biological appeal in a design doc is paired with an engineering justification. If the engineering doesn't stand on its own, the biology is decoration.
- When biology and pragmatism conflict, pragmatism wins by default. Note the divergence in the relevant doc.

## Modularity around brain-system function

- Subsystems carve along functional joints (perception, working memory, episodic memory, semantic memory, sleep, control), not implementation conveniences.
- Each subsystem should be replaceable. If `episodic_memory_v1` is a Hopfield net and `v2` is a vector store, the rest of the system should not care.
- Interfaces between subsystems are the most important thing to get *roughly* right early; internals can churn freely.

## Continuous over batch

- The default mental model is a system that is *running*, not one that is *trained*. Online learning, streaming inputs, asynchronous consolidation.
- Where we use batch methods (because the algorithm requires it), they live inside sleep/consolidation phases — never on the live wake loop.

## Cheap to throw away

- Code in `experiments/` should die without ceremony. No CI gates, no API stability, no docs beyond a top-of-file comment.
- Promotion to the main system is a deliberate, documented act (an ADR).
- Resist the urge to refactor experiments toward production quality. Either promote or delete.

## Honesty about state

- Docs reflect reality. If a subsystem is a stub, say so. If a result is from a toy dataset, say so.
- Aspirational docs are a research liability — they create the illusion of progress.

## Reproducibility within reason

- Seeds are fixed in experiments. Datasets and configs are versioned alongside results.
- We don't need MLflow-grade tooling, but every reported number must be re-derivable.
- An observation log captures what the system saw and did, so behavior can be reconstructed.

## Time-honest evaluation

- The system runs in real time during wake; experiments evaluate it that way when it matters. Wall-clock latency is a first-class metric, not an afterthought.
- Sleep can be wall-clock-free (faster than real time), but its *effects* must be evaluated under wake conditions.
