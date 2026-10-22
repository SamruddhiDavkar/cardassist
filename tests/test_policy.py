from datetime import timedelta

import pytest

from cardassist import config
from cardassist.policy import evaluate


def txn(days_ago=5, amount=50.0, status="posted"):
    return {"date": (config.TODAY - timedelta(days=days_ago)).isoformat(),
            "amount": amount, "status": status}


@pytest.mark.parametrize("kwargs,reason,decision,rule", [
    ({}, "wrong amount", "eligible", "eligible"),
    ({"days_ago": 60}, "", "eligible", "eligible"),              # boundary: exactly 60 days
    ({"days_ago": 61}, "", "ineligible", "outside_window"),
    ({"amount": 500.00}, "", "eligible", "eligible"),            # boundary: exactly $500
    ({"amount": 500.01}, "", "needs_review", "over_threshold"),
    ({"status": "pending"}, "", "ineligible", "pending"),
    ({"status": "disputed"}, "", "ineligible", "already_disputed"),
    ({}, "my card was stolen", "needs_review", "possible_fraud"),
    ({}, "this was UNAUTHORIZED", "needs_review", "possible_fraud"),
    ({"days_ago": 90}, "fraud", "ineligible", "outside_window"),  # window rule wins over fraud
])
def test_rules(kwargs, reason, decision, rule):
    d = evaluate(txn(**kwargs), reason)
    assert (d.decision, d.rule) == (decision, rule)
