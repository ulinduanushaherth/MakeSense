from ot_context.query.decompose import extract_phrases
from ot_context.encoders.encoder import Encoder
from ot_context.selection.adaptive import build_context_set_adaptive
from ot_context.selection.temporal import build_temporal_context

# Config
CONFIG = {
    "mode": "adaptive",   # [fixed, adaptive, temporal]

    "use_query_decomposition": False,

    # for fixed OT
    "k_fixed": 5,

    # for adaptive OT
    "epsilon": 0.01,
    "patience": 2,
    "k_max": 12,

    # for temporal OT
    "temporal_slices": 3,
    "alpha_temporal": 0.3,
}


# Input

QUERY = "Does caffeine improve long-term memory retention in adults?"

CANDIDATE_TEXTS = [
    "Caffeine is a central nervous system stimulant.",
    "Memory consolidation depends on sleep quality.",
    "Several studies examine caffeine and cognition.",
    "High caffeine intake disrupts sleep.",
    "Some experiments show attention gains but not long-term recall."
]


def run_context_construction(query, candidate_texts):
    encoder = Encoder()

    if CONFIG["use_query_decomposition"]:
        _, q_embs = extract_phrases(query)
    else:
        q_embs = encoder.encode(query)

    candidates = [
        {
            "text": t,
            "emb": encoder.encode(t)[0]
        }
        for t in candidate_texts
    ]

    mode = CONFIG["mode"]

    if mode == "fixed":
        selected, cost_curve = build_context_set_adaptive(
            query_embs=q_embs,
            candidates=candidates,
            epsilon=0.0,
            patience=10**9,
            k_max=CONFIG["k_fixed"]
        )

        return {
            "mode": "fixed",
            "context": selected,
            "cost_curve": cost_curve
        }

    if mode == "adaptive":
        selected, cost_curve = build_context_set_adaptive(
            query_embs=q_embs,
            candidates=candidates,
            epsilon=CONFIG["epsilon"],
            patience=CONFIG["patience"],
            k_max=CONFIG["k_max"]
        )

        return {
            "mode": "adaptive",
            "context": selected,
            "cost_curve": cost_curve
        }

    if mode == "temporal":
        temporal_states = build_temporal_context(
            query_embs=q_embs,
            candidates=candidates,
            num_slices=CONFIG["temporal_slices"],
            alpha=CONFIG["alpha_temporal"],
            epsilon=CONFIG["epsilon"],
            patience=CONFIG["patience"],
            k_max=CONFIG["k_max"]
        )

        return {
            "mode": "temporal",
            "states": temporal_states
        }

    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    output = run_context_construction(QUERY, CANDIDATE_TEXTS)

    print("\nMODE:", output["mode"])
    print("Query decomposition:", CONFIG["use_query_decomposition"])

    if output["mode"] == "temporal":
        for t, state in enumerate(output["states"], 1):
            print(f"\nState {t}")
            print(f"  Coverage OT cost : {state['coverage_cost']:.4f}")
            print(f"  Temporal OT cost : {state['temporal_cost']:.4f}")
            print(f"  Joint objective  : {state['objective']:.4f}")

            for i, s in enumerate(state["context"], 1):
                print(f"    [{i}] {s['text']}")

    else:
        print("\nSelected Context:")

        for i, s in enumerate(output["context"], 1):
            print(f"  [{i}] {s['text']}")

        print("\nOT cost progression:")
        for i, c in enumerate(output["cost_curve"], 1):
            print(f"  k={i}: {c:.4f}")

        print(f"\nFinal OT cost: {output['cost_curve'][-1]:.4f}")