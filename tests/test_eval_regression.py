"""Runs the full 30-case evaluation on the rule-based backend as a CI regression gate:
if a change to the tools, policy or guardrails breaks a known case, CI fails."""
import json
from pathlib import Path

from eval import run_eval


def test_main_eval_set_passes_with_rules_backend():
    run_eval.CASES[:] = json.loads((Path(run_eval.ROOT) / "eval" / "eval_set.json").read_text())
    summary = run_eval.summarise(run_eval.run("rules"))
    assert summary["n"] == 30
    assert summary["guardrail_pass_rate"] == 100.0
    assert summary["outcome_accuracy"] == 100.0
