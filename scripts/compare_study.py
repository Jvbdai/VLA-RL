"""Strictly matched descriptive outcome pairs; these are not causal rescues."""
import argparse
import csv
import json
from pathlib import Path


def paired_outcomes(base, residual):
    for field in ("task", "horizon", "checkpoint", "scale", "observation_mode", "perturbation", "instruction_override"):
        if base.get(field) != residual.get(field):
            raise ValueError(f"Incompatible evaluation contract: {field}")
    if base["method"] != "base_vla" or residual["method"] != "residual_td3":
        raise ValueError("Expected base_vla and residual_td3 reports")
    def key(e):
        return e["seed"], e["policy_seed"], e["repeat"]
    baseline = {key(e): e for e in base["episodes"]}
    if len(baseline) != len(base["episodes"]):
        raise ValueError("Duplicate baseline episode keys")
    if len({key(e) for e in residual['episodes']}) != len(residual['episodes']):
        raise ValueError("Duplicate residual episode keys")
    pairs = []
    for r in residual["episodes"]:
        if key(r) not in baseline:
            raise ValueError(f"Missing baseline episode {key(r)}")
        b = baseline[key(r)]
        if not b.get("initial_numeric_model") or not r.get("initial_numeric_model"):
            raise ValueError("Numeric model identity required; legacy XML hashes are insufficient")
        matched = (b["initial_numeric_model"] == r["initial_numeric_model"]
                   and b["initial_state_sha256"] == r["initial_state_sha256"]
                   and b["instruction"] == r["instruction"]
                   and b["perturbation_parameters"] == r["perturbation_parameters"])
        if not matched:
            outcome = "unmatched"
        else:
            outcome = {(False, False): "both_fail", (True, True): "both_succeed",
                       (False, True): "rescue_outcome", (True, False): "harm_outcome"}[(b["success"], r["success"])]
        pairs.append({"seed": r["seed"], "policy_seed": r["policy_seed"], "repeat": r["repeat"],
                      "matched_initial_conditions": matched,
                      "mismatched_numeric_fields": sorted(k for k in set(b["initial_numeric_model"].get("fields", {})) | set(r["initial_numeric_model"].get("fields", {}))
                          if b["initial_numeric_model"].get("fields", {}).get(k) != r["initial_numeric_model"].get("fields", {}).get(k)),
                      "exact_initial_cameras": b.get("initial_camera_sha256") == r.get("initial_camera_sha256")
                      if b.get("initial_camera_sha256") and r.get("initial_camera_sha256") else None,
                      "outcome": outcome, "base_success": b["success"], "residual_success": r["success"],
                      "base_final_error": b.get("final_task_error"), "residual_final_error": r.get("final_task_error"),
                      "base_completion_steps": b["completion_steps"], "residual_completion_steps": r["completion_steps"]})
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, residual = [json.loads(p.read_text(encoding="utf-8")) for p in (args.baseline, args.residual)]
    pairs = paired_outcomes(base, residual)
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {label: sum(p["outcome"] == label for p in pairs)
              for label in ("both_fail", "both_succeed", "rescue_outcome", "harm_outcome", "unmatched")}
    report = {"interpretation": "Descriptive matched outcomes, not causal effects; compare repeated baseline variability.",
              "base_report": str(args.baseline), "residual_report": str(args.residual), "counts": counts, "pairs": pairs}
    (args.output/"paired_outcomes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output/"paired_outcomes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    print(json.dumps(counts, indent=2))

if __name__ == "__main__":
    main()
