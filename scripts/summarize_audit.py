"""Summarize repeated-baseline diagnostics; repeats are not independent trials."""
import argparse
import csv
import itertools
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    episodes = report["episodes"]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [{k: e.get(k) for k in ("seed", "policy_seed", "repeat", "success", "steps",
             "completion_steps", "final_openness", "wall_time_s")} for e in episodes]
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pairs = []
    for a, b in itertools.combinations(episodes, 2):
        if (a["seed"], a["policy_seed"]) != (b["seed"], b["policy_seed"]):
            continue
        pairs.append({"seed": a["seed"], "policy_seed": a["policy_seed"],
            "matched_initial_state": a["initial_state_sha256"] == b["initial_state_sha256"],
            "matched_initial_model_xml": a["initial_model_sha256"] == b["initial_model_sha256"],
            "matched_initial_numeric_model": (
                a["initial_numeric_model"] == b["initial_numeric_model"]
                if "initial_numeric_model" in a and "initial_numeric_model" in b else None),
            "outcome_changed": a["success"] != b["success"],
            "final_openness_difference": abs(a["final_openness"] - b["final_openness"])})
    (args.output / "repeat_variability.json").write_text(json.dumps({
        "note": "Descriptive repeat diagnostics, not independent-trial confidence intervals or causal effects.",
        "pairs": pairs}, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for (seed, policy), values in itertools.groupby(sorted(episodes, key=lambda e: (e["seed"], e["policy_seed"], e["repeat"])),
                                                   key=lambda e: (e["seed"], e["policy_seed"])):
        values = list(values)
        ax.plot([e["repeat"] for e in values], [e["final_openness"] for e in values], "o-", label=f"sim {seed}, policy {policy}")
    ax.set(xlabel="Identical-seed repeat", ylabel="Final normalized mean door openness",
           title="Frozen baseline repeat variability (lower is better)", ylim=(0, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "repeat_variability.png", dpi=160)
    plt.close(fig)
    summary = {"episodes": len(episodes), "successes": sum(e["success"] for e in episodes),
               "failures": sum(not e["success"] for e in episodes),
               "steps_per_second_including_resets": sum(e["steps"] for e in episodes) / sum(e["wall_time_s"] for e in episodes),
               "same_seed_pairs": pairs}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
