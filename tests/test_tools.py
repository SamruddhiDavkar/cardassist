import re

from cardassist.db import connect
from cardassist.tools import (check_dispute_policy, escalate_to_human, get_transactions,
                              open_dispute)


def test_get_transactions_masks_cards(db):
    out = get_transactions("C001", "last_90_days", db_path=db)
    assert out["count"] > 0
    assert re.fullmatch(r"\*{4} \*{4} \*{4} \d{4}", out["card"])
    for t in out["transactions"]:
        assert "card_number" not in t and "card" not in t


def test_get_transactions_only_returns_own_customer(db):
    ids = {t["transaction_id"] for t in get_transactions("C002", None, db_path=db)["transactions"]}
    assert "T9001" not in ids


def test_get_transactions_date_formats(db):
    a = get_transactions("C001", "last_7_days", db_path=db)
    b = get_transactions("C001", {"start": a["start"], "end": a["end"]}, db_path=db)
    c = get_transactions("C001", f"{a['start']}:{a['end']}", db_path=db)
    assert a["count"] == b["count"] == c["count"]


def test_get_transactions_bad_range_returns_error(db):
    assert "error" in get_transactions("C001", "next tuesday", db_path=db)


def test_check_policy(db):
    assert check_dispute_policy("T9001", "don't recognize", db_path=db)["decision"] == "eligible"
    assert check_dispute_policy("T9002", "broken", db_path=db)["rule"] == "over_threshold"
    assert check_dispute_policy("T9003", "", db_path=db)["rule"] == "outside_window"


def test_check_policy_other_customer_is_not_found(db):
    assert "error" in check_dispute_policy("T9001", "", customer_id="C002", db_path=db)


def test_open_dispute_happy_path(db):
    out = open_dispute("T9001", "don't recognize", customer_id="C001", db_path=db)
    assert out["dispute_id"].startswith("D-")
    with connect(db) as c:
        assert c.execute("SELECT status FROM transactions WHERE transaction_id='T9001'"
                         ).fetchone()[0] == "disputed"
    # a second dispute on the same charge is refused
    assert "error" in open_dispute("T9001", "again", customer_id="C001", db_path=db)


def test_open_dispute_enforces_policy_itself(db):
    """Even if the model skips check_dispute_policy, the rules still hold."""
    for tid in ("T9002", "T9003", "T9006", "T9007"):
        assert "error" in open_dispute(tid, "please", customer_id="C001", db_path=db)


def test_escalate_creates_ticket(db):
    out = escalate_to_human("possible fraud", customer_id="C001", db_path=db)
    assert out["ticket_id"].startswith("H-")
    with connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM escalations").fetchone()[0] == 1
