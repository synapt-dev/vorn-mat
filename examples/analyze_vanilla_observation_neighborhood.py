#!/usr/bin/env python3
"""Offline neighborhood probe over the captured vanilla observation traces."""
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer

from vorn_mat import (
    DEFAULT_MODEL,
    build_standard_neighborhoods,
    load_observation_report,
    load_ruler_hf_niah_slice,
    write_neighborhood_probe_artifacts,
    analyze_neighborhood_probes,
)


def _render_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def main(
    observation_json: str = str(
        ROOT / "results" / "vanilla-observation-2026-05-13.json"
    ),
    output_json: str = str(
        ROOT / "results" / "vanilla-observation-neighborhood-2026-05-13.json"
    ),
    output_md: str = str(
        ROOT / "results" / "vanilla-observation-neighborhood-2026-05-13.md"
    ),
    model_id: str = DEFAULT_MODEL,
    top_k: int = 10,
) -> None:
    report = load_observation_report(Path(observation_json))
    benchmark_cases = {
        case.case_id: case
        for case in load_ruler_hf_niah_slice(
            report.dataset_config,
            split="validation",
            case_limit=report.case_count,
        )
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    case_neighborhoods = {}
    for observation_case in report.cases:
        benchmark_case = benchmark_cases[observation_case.case_id]
        rendered_prompt = _render_chat_prompt(tokenizer, benchmark_case.prompt)
        encoding = tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_count = len(encoding["input_ids"])
        if token_count != observation_case.prompt_token_count:
            raise ValueError(
                f"token count mismatch for {observation_case.case_id}: "
                f"rendered={token_count} observed={observation_case.prompt_token_count}"
            )
        case_neighborhoods[observation_case.case_id] = build_standard_neighborhoods(
            rendered_prompt=rendered_prompt,
            offsets=tuple(tuple(offset) for offset in encoding["offset_mapping"]),
            answer_token_spans=observation_case.answer_token_spans,
        )

    summary = analyze_neighborhood_probes(
        report,
        case_neighborhoods=case_neighborhoods,
        top_k=top_k,
    )
    summary["source_observation_manifest"] = observation_json
    write_neighborhood_probe_artifacts(
        summary,
        json_path=Path(output_json),
        markdown_path=Path(output_md),
    )

    print(f"output_json={output_json}")
    print(f"output_md={output_md}")
    for label, probe in summary["probes"].items():
        success = probe["success"]
        failure = probe["failure"]
        print(
            f"{label}: success_topk={success['cases_with_top_k_hit']}/{success['case_count']} "
            f"failure_topk={failure['cases_with_top_k_hit']}/{failure['case_count']}"
        )


if __name__ == "__main__":
    main()
