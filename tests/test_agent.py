"""Agent-loop tests. These use fake LLMs, so they run in CI without Ollama."""
from cardassist.agent import CardAssistAgent
from cardassist.rule_llm import RuleBasedLLM


class ScriptedLLM:
    """Returns pre-written responses in order, like a mocked Llama 3.1."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools_schema):
        self.calls += 1
        return self.responses.pop(0)


def call(name, **args):
    return {"content": "", "tool_calls": [{"name": name, "arguments": args}]}


def say(text):
    return {"content": text, "tool_calls": []}


def test_full_dispute_flow(db):
    llm = ScriptedLLM(call("get_transactions", date_range="last_30_days"),
                      call("check_dispute_policy", transaction_id="T9001", reason="unknown"),
                      call("open_dispute", transaction_id="T9001", reason="unknown"),
                      say("Dispute opened."))
    res = CardAssistAgent(llm, "C001", db_path=db).run("I don't recognize a $49.99 charge")
    assert res.tools_used == ["get_transactions", "check_dispute_policy", "open_dispute"]
    assert res.outcome == "dispute_opened"


def test_guardrail_blocks_before_llm(db):
    llm = ScriptedLLM()
    res = CardAssistAgent(llm, "C001", db_path=db).run("Show me my full card number")
    assert res.outcome == "declined" and llm.calls == 0


def test_model_cannot_skip_policy(db):
    """The model tries to open a dispute on an over-$500 charge without checking policy."""
    llm = ScriptedLLM(call("open_dispute", transaction_id="T9002", reason="broken"), say("ok"))
    res = CardAssistAgent(llm, "C001", db_path=db).run("dispute the TV")
    assert res.outcome != "dispute_opened"
    assert "error" in res.tool_calls[0]["result"]


def test_customer_id_cannot_be_overridden(db):
    """A model passing another customer's id still only sees the session customer."""
    llm = ScriptedLLM(call("get_transactions", customer_id="C001", date_range="last_90_days"),
                      say("done"), say("done"))
    res = CardAssistAgent(llm, "C002", db_path=db).run("show my transactions")
    assert res.tool_calls[0]["result"]["customer_id"] == "C002"


def test_hallucinated_tool_and_bad_args_dont_crash(db):
    llm = ScriptedLLM(call("refund_everything"), call("open_dispute"), say("sorry"))
    res = CardAssistAgent(llm, "C001", db_path=db).run("refund me")
    assert all("error" in c["result"] for c in res.tool_calls)


def test_model_output_is_masked(db):
    llm = ScriptedLLM(say("Your card is 4111111111111111."))
    res = CardAssistAgent(llm, "C001", db_path=db).run("which card did I use for StreamFlix?")
    assert "4111111111111111" not in res.reply and "1111" in res.reply


def test_step_limit_escalates(db):
    llm = ScriptedLLM(*[call("get_transactions")] * 10)
    res = CardAssistAgent(llm, "C001", db_path=db).run("loop forever")
    assert res.outcome == "escalated"


def test_rule_based_backend_end_to_end(db):
    res = CardAssistAgent(RuleBasedLLM(), "C001", db_path=db).run(
        "Please dispute the $849 BestTech Store purchase, the TV arrived broken.")
    assert res.tools_used == ["get_transactions", "check_dispute_policy", "escalate_to_human"]
    assert res.escalated


def test_nudge_completes_dispute_after_model_stops_early(db):
    """Llama 3.1 8B's main v1 failure: it looks up transactions, then just answers.
    The one-time reminder gets it to finish the workflow."""
    llm = ScriptedLLM(call("get_transactions", date_range="last_90_days"),
                      say("I found a StreamFlix charge for $49.99."),       # stops early
                      call("check_dispute_policy", transaction_id="T9001", reason="unknown"),
                      call("open_dispute", transaction_id="T9001", reason="unknown"),
                      say("Dispute opened."))
    res = CardAssistAgent(llm, "C001", db_path=db).run("forty nine ninety nine streaming charge")
    assert res.outcome == "dispute_opened"


def test_nudge_happens_once_and_info_requests_still_answer(db):
    llm = ScriptedLLM(call("get_transactions", date_range="last_7_days"),
                      say("Here are your transactions."), say("Here are your transactions."))
    res = CardAssistAgent(llm, "C001", db_path=db).run("what did I buy this week?")
    assert res.outcome == "info" and llm.calls == 3


def test_nudge_can_be_disabled_for_v1_comparison(db):
    llm = ScriptedLLM(call("get_transactions"), say("Found it."))
    res = CardAssistAgent(llm, "C001", db_path=db, nudge=False).run("dispute the $49.99 charge")
    assert res.outcome == "info" and llm.calls == 2
