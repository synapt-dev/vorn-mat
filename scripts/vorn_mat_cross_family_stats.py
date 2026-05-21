from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data/vorn-mat-cross-family-stats-2026-05-15.json"
ALPHA = 0.05


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    ref: str
    path: str
    pr_number: int | None
    title: str


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def git_text(ref: str, path: str) -> str:
    return git(["show", f"{ref}:{path}"])


def git_rev(ref: str) -> str:
    return git(["rev-parse", ref]).strip()


def last_touch_commit(ref: str, path: str) -> str:
    return git(["log", "-n1", "--format=%H", ref, "--", path]).strip()


def load_artifact(spec: ArtifactSpec) -> dict[str, Any]:
    payload = json.loads(git_text(spec.ref, spec.path))
    source_commit = git_rev(spec.ref) if spec.ref.startswith("pr") else last_touch_commit(spec.ref, spec.path)
    payload["_source"] = {
        "artifact_id": spec.artifact_id,
        "ref": spec.ref,
        "commit": source_commit,
        "path": spec.path,
        "pr_number": spec.pr_number,
        "title": spec.title,
    }
    return payload


def iter_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in payload.values():
        if isinstance(value, list) and value and isinstance(value[0], dict) and "hit_rate" in value[0]:
            rows.extend(value)
    return rows


def row_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["label"]: row for row in iter_rows(payload) if "label" in row}


def pairwise_map(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    tests = payload.get("pairwise_tests", [])
    return {(test["lhs"], test["rhs"]): test for test in tests}


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, (original_idx, p_value) in enumerate(indexed):
        candidate = min(1.0, (total - rank) * p_value)
        running = max(running, candidate)
        adjusted[original_idx] = running
    return adjusted


def min_discordant_pairs_for_alpha(alpha: float = ALPHA) -> int:
    discordant = 0
    while True:
        discordant += 1
        if 2 * (0.5**discordant) < alpha:
            return discordant


def comparison_record(
    *,
    artifact: dict[str, Any],
    family_id: str,
    lhs: str,
    rhs: str,
    family_adjusted_p: float,
    note: str | None = None,
) -> dict[str, Any]:
    rows = row_map(artifact)
    tests = pairwise_map(artifact)
    pair = tests[(lhs, rhs)]
    lhs_row = rows[lhs]
    rhs_row = rows[rhs]
    delta = lhs_row["hit_rate"] - rhs_row["hit_rate"]
    if pair["discordant_pairs"] == 0:
        strength = "non_discriminative"
    elif family_adjusted_p < ALPHA:
        strength = "publication_strong"
    else:
        strength = "directional_only"
    return {
        "family_id": family_id,
        "artifact_id": artifact["_source"]["artifact_id"],
        "source": artifact["_source"],
        "comparison": {"lhs": lhs, "rhs": rhs},
        "lhs_hit_rate": lhs_row["hit_rate"],
        "rhs_hit_rate": rhs_row["hit_rate"],
        "lhs_wilson_ci_95": lhs_row.get("wilson_ci_95"),
        "rhs_wilson_ci_95": rhs_row.get("wilson_ci_95"),
        "delta_hit_rate": delta,
        "direction": "lhs_better" if delta > 0 else "rhs_better" if delta < 0 else "tied",
        "paired_test": {
            "name": pair["test"],
            "p_value": pair["p_value"],
            "holm_adjusted_p_within_family": family_adjusted_p,
            "discordant_pairs": pair["discordant_pairs"],
            "table": pair["table"],
        },
        "classification": strength,
        "note": note,
    }


ARTIFACT_SPECS = {
    "mistral_4k": ArtifactSpec(
        artifact_id="mistral_4k_sentence_budget_sweep",
        ref="origin/main",
        path="results/sentence-level-eviction-4k-budget-sweep-2026-05-13.json",
        pr_number=None,
        title="research: sentence-level vorn eviction budget sweep",
    ),
    "mistral_8k": ArtifactSpec(
        artifact_id="mistral_8k_sentence_budget_sweep",
        ref="origin/main",
        path="results/sentence-level-eviction-8k-budget-sweep-2026-05-13.json",
        pr_number=None,
        title="research: extend vorn sentence matrix to 8k",
    ),
    "qwen_4k": ArtifactSpec(
        artifact_id="qwen_4k_budget_sweep",
        ref="origin/main",
        path="results/qwen-4k-budget-sweep-2026-05-14.json",
        pr_number=None,
        title="research: characterize qwen 4k sentence/token budget sweep",
    ),
    "qwen_word": ArtifactSpec(
        artifact_id="qwen_4k_word_budget_sweep",
        ref="origin/main",
        path="results/qwen-4k-word-budget-sweep-2026-05-14.json",
        pr_number=None,
        title="research: characterize qwen word-shaped eviction",
    ),
    "adaptive_posthoc": ArtifactSpec(
        artifact_id="adaptive_posthoc_map",
        ref="origin/main",
        path="results/adaptive-granularity-posthoc-2026-05-14.json",
        pr_number=None,
        title="research: map adaptive-granularity posthoc oracle",
    ),
    "gemma_gate": ArtifactSpec(
        artifact_id="gemma_4k_gate",
        ref="origin/main",
        path="results/gemma-4-4k-gate-2026-05-15.json",
        pr_number=None,
        title="feat: add gemma 4 cross-model gate",
    ),
    "gemma_curve": ArtifactSpec(
        artifact_id="gemma_4k_degradation_curve",
        ref="origin/main",
        path="results/gemma-4-4k-degradation-curve-2026-05-15.json",
        pr_number=None,
        title="feat: add Gemma 4 degradation curve",
    ),
    "qwen3_gate": ArtifactSpec(
        artifact_id="qwen3_4k_gate",
        ref="origin/main",
        path="results/qwen3-4k-gate-2026-05-15.json",
        pr_number=None,
        title="feat: package qwen 3 cross-model gate",
    ),
    "llama_gate": ArtifactSpec(
        artifact_id="llama31_4k_gate",
        ref="origin/main",
        path="results/llama31-4k-gate-2026-05-15.json",
        pr_number=None,
        title="feat: add llama 3.1 cross-model gate",
    ),
    "gemma_xbase": ArtifactSpec(
        artifact_id="gemma_4k_cross_baseline_extension",
        ref="origin/main",
        path="results/gemma-4-4k-cross-baseline-extension-2026-05-15.json",
        pr_number=None,
        title="feat: add gemma 4 cross-baseline extension",
    ),
    "llama_xbase": ArtifactSpec(
        artifact_id="llama31_4k_cross_baseline_extension",
        ref="origin/main",
        path="results/llama31-4k-cross-baseline-extension-2026-05-15.json",
        pr_number=None,
        title="feat: add llama 3.1 cross-baseline extension",
    ),
    "cross_family_no_guards": ArtifactSpec(
        artifact_id="cross_family_no_guards_mirror",
        ref="origin/main",
        path="results/cross-family-no-guards-mirror-2026-05-15.json",
        pr_number=None,
        title="feat: add no-guards mirror and llama threshold artifacts",
    ),
    "llama_threshold_cut": ArtifactSpec(
        artifact_id="llama31_4k_threshold_cut",
        ref="origin/main",
        path="results/llama31-4k-threshold-cut-2026-05-15.json",
        pr_number=None,
        title="feat: add no-guards mirror and llama threshold artifacts",
    ),
    "ministral_fast_read": ArtifactSpec(
        artifact_id="ministral_8b_4k_fast_read",
        ref="origin/main",
        path="results/ministral-8b-4k-fast-read-2026-05-15.json",
        pr_number=None,
        title="feat: add Ministral 8B fast-read artifact",
    ),
}


def build_family_records(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add_family(
        family_id: str,
        artifact_key: str,
        comparisons: list[tuple[str, str, str | None]],
    ) -> None:
        artifact = artifacts[artifact_key]
        tests = pairwise_map(artifact)
        raw_ps = [tests[(lhs, rhs)]["p_value"] for lhs, rhs, _ in comparisons]
        adjusted = holm_adjust(raw_ps)
        for (lhs, rhs, note), adj in zip(comparisons, adjusted, strict=True):
            records.append(
                comparison_record(
                    artifact=artifact,
                    family_id=family_id,
                    lhs=lhs,
                    rhs=rhs,
                    family_adjusted_p=adj,
                    note=note,
                )
            )

    add_family(
        "mistral_4k_sentence_vs_token",
        "mistral_4k",
        [
            ("sentence_512_guarded", "token_512_guarded", None),
            ("sentence_1024_guarded", "token_1024_guarded", None),
            ("sentence_1536_guarded", "token_1536_guarded", None),
            ("sentence_2048_guarded", "token_2048_guarded", None),
        ],
    )
    add_family(
        "mistral_8k_sentence_vs_token",
        "mistral_8k",
        [
            ("sentence_512_guarded", "token_512_guarded", None),
            ("sentence_1024_guarded", "token_1024_guarded", None),
            ("sentence_1536_guarded", "token_1536_guarded", None),
            ("sentence_2048_guarded", "token_2048_guarded", None),
        ],
    )
    add_family(
        "qwen_4k_sentence_vs_token",
        "qwen_4k",
        [
            ("sentence_256_guarded", "token_256_guarded", None),
            ("sentence_512_guarded", "token_512_guarded", None),
            ("sentence_1024_guarded", "token_1024_guarded", None),
            ("sentence_1536_guarded", "token_1536_guarded", "Best Qwen token edge appears at 1536, but remains non-decisive."),
            ("sentence_2048_guarded", "token_2048_guarded", None),
        ],
    )
    add_family(
        "gemma_4k_gate",
        "gemma_gate",
        [
            ("sentence_1024_guarded", "token_1024_guarded", None),
            ("sentence_1024_noguards", "sentence_1024_guarded", None),
            ("sentence_1024_guarded", "vanilla_floor", "Negative delta to vanilla is expected here because vanilla is already at ceiling."),
        ],
    )
    add_family(
        "qwen3_4k_gate",
        "qwen3_gate",
        [
            ("sentence_1024_guarded", "token_1024_guarded", "Zero-discordance gate: no retention-policy discrimination on the shipped surface."),
            ("sentence_1024_noguards", "sentence_1024_guarded", None),
        ],
    )
    add_family(
        "llama31_4k_gate",
        "llama_gate",
        [
            ("sentence_1024_guarded", "token_1024_guarded", "Ceiling gate: no retention-policy discrimination on the shipped surface."),
            ("sentence_1024_noguards", "sentence_1024_guarded", None),
            ("sentence_1024_guarded", "vanilla_floor", None),
        ],
    )
    add_family(
        "gemma_4k_cross_baseline",
        "gemma_xbase",
        [
            ("tova_1024_guarded", "token_1024_guarded", None),
            ("h2o_1024_guarded", "token_1024_guarded", None),
            ("tova_1024_guarded", "sentence_1024_guarded", None),
            ("h2o_1024_guarded", "sentence_1024_guarded", None),
            ("tova_1024_guarded", "h2o_1024_guarded", None),
            ("tova_1024_guarded", "vanilla_floor", None),
            ("h2o_1024_guarded", "vanilla_floor", None),
        ],
    )
    add_family(
        "gemma_4k_curve",
        "gemma_curve",
        [
            ("sentence_512_guarded", "token_512_guarded", None),
            ("tova_512_guarded", "sentence_512_guarded", None),
            ("sentence_1536_guarded", "token_1536_guarded", None),
            ("tova_1536_guarded", "sentence_1536_guarded", None),
            ("sentence_2048_guarded", "token_2048_guarded", None),
            ("tova_2048_guarded", "sentence_2048_guarded", "At 2048, TOVA remains numerically above sentence, but the paired test is non-decisive."),
        ],
    )
    add_family(
        "llama31_4k_cross_baseline",
        "llama_xbase",
        [
            ("tova_1024_guarded", "token_1024_guarded", "Five one-sided discordant pairs is still p = 0.0625 under exact two-sided McNemar."),
            ("h2o_1024_guarded", "token_1024_guarded", None),
            ("tova_1024_guarded", "sentence_1024_guarded", None),
            ("h2o_1024_guarded", "sentence_1024_guarded", None),
            ("tova_1024_guarded", "h2o_1024_guarded", None),
            ("tova_1024_guarded", "vanilla_floor", None),
            ("h2o_1024_guarded", "vanilla_floor", None),
        ],
    )
    add_family(
        "cross_family_no_guards_guardrail_checks",
        "cross_family_no_guards",
        [
            ("gemma_token_noguards", "gemma_token_guarded", None),
            ("gemma_sentence_noguards", "gemma_sentence_guarded", None),
            ("gemma_tova_noguards", "gemma_tova_guarded", None),
            ("gemma_h2o_noguards", "gemma_h2o_guarded", None),
            ("llama_token_noguards", "llama_token_guarded", None),
            ("llama_sentence_noguards", "llama_sentence_guarded", None),
            ("llama_tova_noguards", "llama_tova_guarded", None),
            ("llama_h2o_noguards", "llama_h2o_guarded", None),
        ],
    )
    add_family(
        "llama31_4k_threshold_cut",
        "llama_threshold_cut",
        [
            ("llama_token_512_guarded", "llama_token_1024_guarded", None),
            ("llama_sentence_512_guarded", "llama_sentence_1024_guarded", None),
            ("llama_tova_512_guarded", "llama_tova_1024_guarded", None),
            ("llama_h2o_512_guarded", "llama_h2o_1024_guarded", None),
        ],
    )
    add_family(
        "ministral_8b_fast_read",
        "ministral_fast_read",
        [
            ("sentence_1024_guarded", "tova_1024_guarded", None),
            ("vanilla_floor", "tova_1024_guarded", None),
            ("sentence_1024_guarded", "vanilla_floor", None),
        ],
    )

    return records


def build_supporting_observations(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    adaptive = artifacts["adaptive_posthoc"]
    qwen_word = artifacts["qwen_word"]
    qwen_word_rows = row_map(qwen_word)
    word_labels = [label for label in qwen_word_rows if label.startswith("word_")]
    return {
        "adaptive_posthoc_summary": adaptive["summary"],
        "qwen_word_all_zero": all(qwen_word_rows[label]["hit_rate"] == 0.0 for label in word_labels),
    }


def build_claims(records: list[dict[str, Any]], supporting: dict[str, Any]) -> list[dict[str, Any]]:
    by_key = {(record["family_id"], record["comparison"]["lhs"], record["comparison"]["rhs"]): record for record in records}
    claims = [
        {
            "claim_id": "mistral_4k_sentence_advantage_1024",
            "summary": "Mistral 4k sentence-level decisively outperforms token-level at budget 1024.",
            "support": [
                by_key[("mistral_4k_sentence_vs_token", "sentence_1024_guarded", "token_1024_guarded")]
            ],
            "classification": "publication_strong",
        },
        {
            "claim_id": "mistral_8k_midband_sentence_advantage",
            "summary": "Mistral 8k sentence-level wins in the mid-budget band (1024 and 1536), while 2048 remains non-decisive.",
            "support": [
                by_key[("mistral_8k_sentence_vs_token", "sentence_1024_guarded", "token_1024_guarded")],
                by_key[("mistral_8k_sentence_vs_token", "sentence_1536_guarded", "token_1536_guarded")],
                by_key[("mistral_8k_sentence_vs_token", "sentence_2048_guarded", "token_2048_guarded")],
            ],
            "classification": "publication_strong_with_regime_partition",
        },
        {
            "claim_id": "qwen_blocks_universal_sentence_law",
            "summary": "Qwen blocks any universal sentence-law claim, but the evidence is non-discriminative rather than strongly anti-sentence.",
            "support": [
                by_key[("qwen_4k_sentence_vs_token", "sentence_1536_guarded", "token_1536_guarded")],
                by_key[("qwen3_4k_gate", "sentence_1024_guarded", "token_1024_guarded")],
            ],
            "classification": "directional_only",
        },
        {
            "claim_id": "gemma_degradation_regime",
            "summary": "Gemma 4 sits in a degradation regime: sentence improves over token but remains far below vanilla/TOVA/H2O on the tested slice.",
            "support": [
                by_key[("gemma_4k_gate", "sentence_1024_guarded", "token_1024_guarded")],
                by_key[("gemma_4k_gate", "sentence_1024_guarded", "vanilla_floor")],
                by_key[("gemma_4k_cross_baseline", "tova_1024_guarded", "sentence_1024_guarded")],
                by_key[("gemma_4k_cross_baseline", "h2o_1024_guarded", "sentence_1024_guarded")],
                by_key[("gemma_4k_curve", "tova_1536_guarded", "sentence_1536_guarded")],
                by_key[("gemma_4k_curve", "sentence_2048_guarded", "token_2048_guarded")],
            ],
            "classification": "publication_strong",
        },
        {
            "claim_id": "llama_threshold_gate",
            "summary": "Llama 3.1 functions as a threshold gate, not a discriminative retention comparison, on the tested 4k slice.",
            "support": [
                by_key[("llama31_4k_gate", "sentence_1024_guarded", "token_1024_guarded")],
                by_key[("llama31_4k_gate", "sentence_1024_guarded", "vanilla_floor")],
            ],
            "classification": "non_discriminative",
        },
        {
            "claim_id": "llama_vs_gemma_mirror_image",
            "summary": "The Gemma-vs-Llama mirror-image framing survives guardrail removal, but it is still not inferentially symmetric at n=50.",
            "support": [
                by_key[("gemma_4k_cross_baseline", "tova_1024_guarded", "sentence_1024_guarded")],
                by_key[("llama31_4k_cross_baseline", "tova_1024_guarded", "sentence_1024_guarded")],
                by_key[("llama31_4k_cross_baseline", "tova_1024_guarded", "token_1024_guarded")],
                by_key[("cross_family_no_guards_guardrail_checks", "gemma_sentence_noguards", "gemma_sentence_guarded")],
                by_key[("cross_family_no_guards_guardrail_checks", "llama_sentence_noguards", "llama_sentence_guarded")],
            ],
            "classification": "directional_only",
        },
        {
            "claim_id": "guardrail_removal_does_not_reverse_family_split",
            "summary": "Removing guardrails does not reveal a hidden reversal of the Gemma/Llama family split on the tested 1024 slice.",
            "support": [
                by_key[("cross_family_no_guards_guardrail_checks", "gemma_token_noguards", "gemma_token_guarded")],
                by_key[("cross_family_no_guards_guardrail_checks", "gemma_sentence_noguards", "gemma_sentence_guarded")],
                by_key[("cross_family_no_guards_guardrail_checks", "llama_token_noguards", "llama_token_guarded")],
                by_key[("cross_family_no_guards_guardrail_checks", "llama_sentence_noguards", "llama_sentence_guarded")],
            ],
            "classification": "observational_only",
        },
        {
            "claim_id": "llama_threshold_cut_is_method_specific",
            "summary": "On Llama 3.1, the threshold cut below 1024 appears sharply in the attention-weight baselines, while sentence remains at ceiling and token remains near-ceiling.",
            "support": [
                by_key[("llama31_4k_threshold_cut", "llama_tova_512_guarded", "llama_tova_1024_guarded")],
                by_key[("llama31_4k_threshold_cut", "llama_h2o_512_guarded", "llama_h2o_1024_guarded")],
                by_key[("llama31_4k_threshold_cut", "llama_sentence_512_guarded", "llama_sentence_1024_guarded")],
            ],
            "classification": "publication_strong",
        },
        {
            "claim_id": "ministral_fast_read_is_strong_but_narrow",
            "summary": "Ministral 8B provides a strong fifth-family fast-read in the residual-preserving direction, but not yet a full family classification.",
            "support": [
                by_key[("ministral_8b_fast_read", "sentence_1024_guarded", "tova_1024_guarded")],
                by_key[("ministral_8b_fast_read", "vanilla_floor", "tova_1024_guarded")],
                by_key[("ministral_8b_fast_read", "sentence_1024_guarded", "vanilla_floor")],
            ],
            "classification": "publication_strong",
        },
        {
            "claim_id": "word_never_wins_is_local",
            "summary": "Word never wins in the observed regime table, but that remains an observed-regime statement, not a universal theorem.",
            "support": [supporting["adaptive_posthoc_summary"], {"qwen_word_all_zero": supporting["qwen_word_all_zero"]}],
            "classification": "observational_only",
        },
    ]
    return claims


def main() -> None:
    artifacts = {key: load_artifact(spec) for key, spec in ARTIFACT_SPECS.items()}
    records = build_family_records(artifacts)
    supporting = build_supporting_observations(artifacts)
    claims = build_claims(records, supporting)
    output = {
        "note_slug": "vorn-mat-cross-family-stats-2026-05-15",
        "alpha": ALPHA,
        "minimum_one_sided_discordant_pairs_for_exact_two_sided_significance": min_discordant_pairs_for_alpha(ALPHA),
        "artifacts": {key: artifacts[key]["_source"] for key in artifacts},
        "comparisons": records,
        "supporting_observations": supporting,
        "claims": claims,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(OUTPUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
