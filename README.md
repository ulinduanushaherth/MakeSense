# MakeSense

**Semantic Evidence Construction with Optimal Transport**  
Reference implementation: **`eviot`**

---

## What is this?

**MakeSense** is a research framework for **constructing evidence sets**, not just ranking sentences.

Its Python implementation, **`eviot`**, uses **Optimal Transport (OT)** to select a *set* of sentences that are **jointly sufficient** to answer complex (multi-hop) queries.

This repository is intended for **research and analysis**, not production retrieval.

---

## Why MakeSense?

Traditional retrieval answers:

> “Which sentences are relevant?”

MakeSense answers:

> “Which **set of sentences together** is sufficient?”

This distinction matters for:
- multi-hop questions
- redundant evidence
- reasoning across multiple facts
- deciding **when to stop retrieving**

---

## Core idea

1. Embed the query (optionally decomposed into semantic supports)
2. Embed candidate sentences
3. Use **Optimal Transport** to measure *coverage* between:
   - query representation
   - candidate evidence set
4. Construct evidence **as a set**, not a ranking

---

## Query representation

`eviot` supports two modes.

### With query decomposition
- Query is split into semantic supports
- OT enforces coverage across supports
- Produces smaller, inference-based contexts

Best for:
- semantic sufficiency
- redundancy suppression
- theory-driven evaluation

---

### Without query decomposition
- Query is embedded as a single vector
- OT behaves closer to dense retrieval
- Favors explicit answer sentences

Best for:
- HotpotQA-style datasets
- gold sentence recall
- dataset-aligned evaluation

---

## Evidence construction modes

### Adaptive OT (default)
- Greedy selection
- OT-based stopping
- Stops when marginal gain saturates
- Produces minimal sufficient context

---

### Fixed OT
- Selects exactly `k` sentences
- Useful for ablations
- Inflates context with redundancy
- Not recommended as final method

---

### Temporal OT
- Models **progressive evidence discovery**
- Evidence appears over time
- Penalizes abrupt semantic shifts
- States are **not answer-complete individually**

Used for:
- analysis
- evidence evolution
- semantic drift

---

## Package structure

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch transformers pot spacy
python -m spacy download en_core_web_sm

```
---

## To run

```bash
python -m eviot.runners.single_query
```

## Example Configuration

```bash
CONFIG = {
    "mode": "adaptive",          # fixed | adaptive | temporal
    "use_query_decomposition": True,

    "epsilon": 0.01,
    "patience": 2,
    "k_max": 12,

    "temporal_slices": 3,
    "alpha_temporal": 0.3,
}
```

