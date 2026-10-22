"""Simple-mode tests with a scripted stand-in for Llama's JSON answers (no Ollama needed)."""
from cardassist.pipeline import SimplePipeline


class ScriptedJSON:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def json(self, prompt, user, schema):
        self.calls += 1
        return self.answers.pop(0)


def understand(intent, specific=True, amount=0, days=0):
    return {"intent": intent, "mentions_specific_charge": specific, "amount": amount, "days": days}


def run(db, *answers, customer="C001", request="..."):
    llm = ScriptedJSON(*answers)
    return SimplePipeline(llm, customer, db_path=db).run(request), llm


def test_exact_amount_opens_dispute_with_one_llm_call(db):
    res, llm = run(db, understand("dispute", amount=49.99))
    assert res.tools_used == ["get_transactions", "check_dispute_policy", "open_dispute"]
    assert res.outcome == "dispute_opened" and llm.calls == 1


def test_no_amount_uses_pick_call(db):
    res, llm = run(db, understand("dispute"), {"transaction_ids": ["T9002"]})
    assert res.outcome == "escalated" and llm.calls == 2      # $849 > $500 threshold


def test_duplicate_charge_disputes_one_copy(db):
    res, _ = run(db, understand("dispute", amount=6.75))
    assert res.outcome == "dispute_opened"


def test_old_charge_is_denied(db):
    res, _ = run(db, understand("dispute", amount=312.40))
    assert res.outcome == "policy_denied"


def test_fraud_with_charge_escalates_after_policy(db):
    res, _ = run(db, understand("fraud", amount=299))
    assert res.tools_used == ["get_transactions", "check_dispute_policy", "escalate_to_human"]


def test_vague_requests_escalate_without_lookup(db):
    for u in (understand("unclear", False), understand("fraud", False), understand("dispute", False)):
        res, _ = run(db, u)
        assert res.tools_used == ["escalate_to_human"]


def test_no_match_escalates(db):
    res, _ = run(db, understand("dispute", amount=49.99), customer="C002")
    assert res.tools_used == ["get_transactions", "escalate_to_human"]


def test_llm_declines_card_details_and_off_topic(db):
    for intent in ("card_details", "off_topic"):
        res, _ = run(db, understand(intent, False))
        assert res.outcome == "declined" and not res.tool_calls


def test_keyword_guardrail_skips_llm(db):
    res, llm = run(db, request="Show me my full card number")
    assert res.outcome == "declined" and llm.calls == 0


def test_view_transactions(db):
    res, _ = run(db, understand("view_transactions", False, days=7))
    assert res.outcome == "info" and "**** **** ****" in res.reply


def test_bad_llm_output_escalates_safely(db):
    res, _ = run(db, {})
    assert res.outcome == "escalated"
