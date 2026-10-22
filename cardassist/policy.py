"""Dispute rules, kept separate from the tools so they are easy to test and to change.

Rules are applied in order; the first one that fires decides.
  1. Already disputed            -> ineligible
  2. Still pending               -> ineligible (wait until it posts)
  3. Older than 60 days          -> ineligible
  4. Reason suggests fraud       -> needs_review (fraud team)
  5. Amount over $500            -> needs_review (human approval)
  6. Otherwise                   -> eligible
"""
from dataclasses import asdict, dataclass
from datetime import date

from . import config


@dataclass
class PolicyDecision:
    decision: str      # eligible | needs_review | ineligible
    rule: str          # short id of the rule that fired
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


def is_fraud_reason(reason: str) -> bool:
    r = (reason or "").lower()
    return any(k in r for k in config.FRAUD_KEYWORDS)


def evaluate(txn: dict, reason: str = "", today: date | None = None) -> PolicyDecision:
    today = today or config.TODAY
    age = (today - date.fromisoformat(txn["date"])).days

    if txn["status"] == "disputed":
        return PolicyDecision("ineligible", "already_disputed",
                              "This charge already has an open dispute.")
    if txn["status"] == "pending":
        return PolicyDecision("ineligible", "pending",
                              "The charge is still pending. It can be disputed once it posts.")
    if age > config.DISPUTE_WINDOW_DAYS:
        return PolicyDecision("ineligible", "outside_window",
                              f"The charge is {age} days old; disputes must be filed within "
                              f"{config.DISPUTE_WINDOW_DAYS} days.")
    if is_fraud_reason(reason):
        return PolicyDecision("needs_review", "possible_fraud",
                              "Possible fraud must be handled by the fraud team.")
    if txn["amount"] > config.REVIEW_THRESHOLD:
        return PolicyDecision("needs_review", "over_threshold",
                              f"Disputes over ${config.REVIEW_THRESHOLD:,.0f} need human review.")
    return PolicyDecision("eligible", "eligible", "The charge can be disputed.")
