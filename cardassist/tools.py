"""The four tools the agent can call, plus their JSON schemas for function calling.

Every tool returns a JSON-serialisable dict and never raises on bad input: errors come
back as {"error": ...} so the model can recover instead of the loop crashing.
Card numbers are only ever returned masked.
"""
import uuid
from datetime import date, timedelta

from . import config, policy
from .db import connect
from .guardrails import mask_card


def _parse_range(date_range) -> tuple[str, str]:
    """Accept {"start": "...", "end": "..."}, "YYYY-MM-DD:YYYY-MM-DD", "last_N_days" or None."""
    today = config.TODAY
    if not date_range:
        return (today - timedelta(days=90)).isoformat(), today.isoformat()
    if isinstance(date_range, dict):
        start = date_range.get("start") or (today - timedelta(days=90)).isoformat()
        end = date_range.get("end") or today.isoformat()
        date.fromisoformat(start), date.fromisoformat(end)  # validate
        return start, end
    s = str(date_range).strip().lower()
    if s.startswith("last_") and s.endswith("_days"):
        n = int(s[5:-5])
        return (today - timedelta(days=n)).isoformat(), today.isoformat()
    if ":" in s:
        start, end = s.split(":", 1)
        date.fromisoformat(start), date.fromisoformat(end)
        return start, end
    raise ValueError(f"Unrecognised date_range {date_range!r}")


def _get_txn(conn, transaction_id: str):
    row = conn.execute(
        "SELECT t.*, c.card_number FROM transactions t JOIN customers c USING (customer_id) "
        "WHERE transaction_id = ?", (transaction_id,)).fetchone()
    return dict(row) if row else None


def _public(txn: dict) -> dict:
    out = {k: txn[k] for k in ("transaction_id", "date", "merchant", "category", "amount", "status")}
    out["card"] = mask_card(txn["card_number"])
    return out


def get_transactions(customer_id: str, date_range=None, db_path=None) -> dict:
    try:
        start, end = _parse_range(date_range)
    except (ValueError, TypeError) as e:
        return {"error": str(e)}
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT t.*, c.card_number FROM transactions t JOIN customers c USING (customer_id) "
            "WHERE customer_id = ? AND date BETWEEN ? AND ? ORDER BY date DESC",
            (customer_id, start, end)).fetchall()
    # Compact rows: the card is the same for every row, so it is shown once. Shorter tool
    # output fits Llama's 4096-token context comfortably and is much faster on a CPU.
    keys = ("transaction_id", "date", "merchant", "category", "amount", "status")
    card = mask_card(rows[0]["card_number"]) if rows else None
    return {"customer_id": customer_id, "start": start, "end": end, "card": card,
            "count": len(rows), "transactions": [{k: r[k] for k in keys} for r in rows]}


def check_dispute_policy(transaction_id: str, reason: str = "", customer_id: str | None = None,
                         db_path=None) -> dict:
    with connect(db_path) as conn:
        txn = _get_txn(conn, transaction_id)
    if not txn or (customer_id and txn["customer_id"] != customer_id):
        return {"error": f"Transaction {transaction_id} not found for this customer."}
    result = policy.evaluate(txn, reason).to_dict()
    result["transaction"] = _public(txn)
    return result


def open_dispute(transaction_id: str, reason: str, customer_id: str | None = None,
                 db_path=None) -> dict:
    """Opens a dispute. Re-checks the policy itself, so the model cannot skip the rules."""
    with connect(db_path) as conn:
        txn = _get_txn(conn, transaction_id)
        if not txn or (customer_id and txn["customer_id"] != customer_id):
            return {"error": f"Transaction {transaction_id} not found for this customer."}
        decision = policy.evaluate(txn, reason)
        if decision.decision != "eligible":
            return {"error": "Dispute not allowed by policy.", **decision.to_dict()}
        dispute_id = "D-" + uuid.uuid4().hex[:8].upper()
        conn.execute("INSERT INTO disputes VALUES (?,?,?,?,?)",
                     (dispute_id, transaction_id, reason, config.TODAY.isoformat(), "open"))
        conn.execute("UPDATE transactions SET status='disputed' WHERE transaction_id=?",
                     (transaction_id,))
    return {"dispute_id": dispute_id, "status": "open", "transaction": _public(txn),
            "message": "Dispute opened. Expect an update within 10 business days."}


def escalate_to_human(reason: str, customer_id: str | None = None, db_path=None) -> dict:
    ticket_id = "H-" + uuid.uuid4().hex[:8].upper()
    with connect(db_path) as conn:
        conn.execute("INSERT INTO escalations VALUES (?,?,?,?)",
                     (ticket_id, customer_id, reason or "unspecified", config.TODAY.isoformat()))
    return {"ticket_id": ticket_id, "status": "escalated",
            "message": "A specialist will contact the customer within 1 business day."}


TOOL_FUNCTIONS = {
    "get_transactions": get_transactions,
    "check_dispute_policy": check_dispute_policy,
    "open_dispute": open_dispute,
    "escalate_to_human": escalate_to_human,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_transactions",
        "description": "List the customer's card transactions in a date range, newest first.",
        "parameters": {"type": "object", "properties": {
            "date_range": {"type": "string",
                           "description": "'last_N_days' (e.g. 'last_30_days') or "
                                          "'YYYY-MM-DD:YYYY-MM-DD'. Defaults to last 90 days."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "check_dispute_policy",
        "description": "Check whether a transaction can be disputed. Always call before open_dispute.",
        "parameters": {"type": "object", "properties": {
            "transaction_id": {"type": "string"},
            "reason": {"type": "string", "description": "The customer's reason, in their words."}},
            "required": ["transaction_id", "reason"]}}},
    {"type": "function", "function": {
        "name": "open_dispute",
        "description": "Open a dispute on a transaction that check_dispute_policy marked eligible.",
        "parameters": {"type": "object", "properties": {
            "transaction_id": {"type": "string"},
            "reason": {"type": "string"}},
            "required": ["transaction_id", "reason"]}}},
    {"type": "function", "function": {
        "name": "escalate_to_human",
        "description": "Hand the case to a human specialist. Use for possible fraud, amounts over "
                       "$500, unclear requests, or when no matching transaction is found.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"}},
            "required": ["reason"]}}},
]
