"""Simple mode: the LLM understands, the code acts.

Instead of letting the model chain 3-4 tool calls by itself (which Llama 3.1 8B does
unreliably), the LLM does two small, structured jobs and Python runs the workflow:

  1. understand()  one short JSON call: what does the customer want?
                   (dispute / fraud / view / card_details / off_topic / unclear, plus amount, days)
  2. pick()        only when several charges could match: which transaction is it?

Everything else (looking up charges, the dispute rules, opening disputes, escalating) is
plain code calling the same four tools, so it is fast, testable and can't skip a step.
Most requests need one LLM call; some need two.
"""
import json

from . import config, guardrails, tools
from .agent import AgentResult, _outcome

INTENTS = ["dispute", "fraud", "view_transactions", "card_details", "off_topic", "unclear"]

UNDERSTAND_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "mentions_specific_charge": {"type": "boolean"},
        "amount": {"type": "number"},      # 0 = not given (plain types keep the grammar simple)
        "days": {"type": "integer"},
    },
    "required": ["intent", "mentions_specific_charge", "amount", "days"],
}

UNDERSTAND_PROMPT = """You classify messages sent to a credit card support assistant. Today is {today}.
Return JSON with:
- intent:
  "dispute"           customer wants to dispute/contest a charge, doesn't recognize it, was charged
                      twice or wrongly, cancelled but was billed, wants money back for a charge
  "fraud"             mentions fraud, stolen card, hacked account, leaked card info, or purchases
                      they never made because someone else used the card
  "view_transactions" only wants to see, list or check their charges
  "card_details"      asks for the card number, digits, CVV/security code, expiry or PIN
  "off_topic"         anything not about their card charges (investing, travel, weather, writing...)
  "unclear"           about their account but too vague to act on
- mentions_specific_charge: true if the message gives an amount, a merchant or type of store,
  or a date for the charge
- amount: the charge amount as a number if given (e.g. "forty nine ninety nine" -> 49.99), else 0
- days: for view_transactions, how many days back to look (this week = 7, month = 30), else 0"""

PICK_SCHEMA = {
    "type": "object",
    "properties": {"transaction_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["transaction_ids"],
}

PICK_PROMPT = """Today is {today}. A customer wrote: "{request}"
Here are their card transactions (id | date | merchant | category | amount | status):
{lines}

Which transaction(s) is the customer talking about? Match on amount, merchant or store type,
and date. Return JSON {{"transaction_ids": [...]}} with every matching id.
Return an empty list if none clearly match."""


class OllamaJSON:
    """Llama 3.1 via Ollama, constrained to a JSON schema."""

    def __init__(self, model: str = config.OLLAMA_MODEL, host: str = config.OLLAMA_HOST):
        import ollama
        self.client = ollama.Client(host=host)
        self.model = model

    def json(self, prompt: str, user: str | None, schema: dict) -> dict:
        messages = [{"role": "system", "content": prompt}]
        if user:
            messages.append({"role": "user", "content": user})
        resp = self.client.chat(model=self.model, messages=messages, format=schema,
                                options={"temperature": 0})
        try:
            return json.loads(resp["message"]["content"])
        except (json.JSONDecodeError, TypeError):
            return {}


class SimplePipeline:
    def __init__(self, llm, customer_id: str, db_path=None):
        self.llm = llm
        self.customer_id = customer_id
        self.db_path = db_path
        self.calls: list[dict] = []

    def _tool(self, name, **args):
        fn = tools.TOOL_FUNCTIONS[name]
        if "customer_id" in fn.__code__.co_varnames:
            args["customer_id"] = self.customer_id
        result = fn(**args, db_path=self.db_path)
        self.calls.append({"name": name, "arguments": args, "result": result})
        return result

    def _done(self, reply, guardrail=None, outcome=None):
        return AgentResult(reply=guardrails.mask_output(reply), tool_calls=self.calls,
                           outcome=outcome or _outcome(self.calls), guardrail=guardrail)

    def _escalate(self, why):
        r = self._tool("escalate_to_human", reason=why)
        return self._done(f"I've passed this to a specialist (ticket {r['ticket_id']}). {r['message']}")

    def run(self, request: str) -> AgentResult:
        blocked = guardrails.check_input(request)
        if blocked:
            return self._done(blocked[1], guardrail=blocked[0], outcome="declined")

        today = config.TODAY.isoformat()
        u = self.llm.json(UNDERSTAND_PROMPT.format(today=today), request, UNDERSTAND_SCHEMA)
        intent = u.get("intent", "unclear")

        if intent == "card_details":
            return self._done(guardrails.DECLINE_CARD_DATA, guardrail="card_data", outcome="declined")
        if intent == "off_topic":
            return self._done(guardrails.DECLINE_OFF_TOPIC, guardrail="off_topic", outcome="declined")
        if intent == "view_transactions":
            days = u.get("days") or 30
            r = self._tool("get_transactions", date_range=f"last_{int(days)}_days")
            if not r.get("transactions"):
                return self._done(f"I found no transactions in the last {days} days.")
            lines = [f"- {t['date']}  {t['merchant']}  ${t['amount']:.2f} ({t['status']})"
                     for t in r["transactions"][:20]]
            return self._done(f"Your last {days} days on card {r['card']}:\n" + "\n".join(lines))
        if intent not in ("dispute", "fraud") or not u.get("mentions_specific_charge"):
            label = "Possible fraud" if intent == "fraud" else "Unclear request"
            return self._escalate(f"{label}, no specific charge: {request}")

        # dispute or fraud about a specific charge: find it
        txns = self._tool("get_transactions", date_range="last_90_days")["transactions"]
        amount = u.get("amount")
        cands = [t for t in txns if abs(t["amount"] - amount) < 0.005] if amount else txns
        approximate = False
        if amount and not cands:     # "about 850 dollars": allow 2%, but let the LLM confirm
            cands = [t for t in txns if abs(t["amount"] - amount) <= 0.02 * amount]
            approximate = True
        # "charged twice": identical copies of one charge -> dispute one of them, no LLM needed
        if amount and len({(t["merchant"], t["date"], t["amount"]) for t in cands}) == 1:
            cands = cands[:1]
        if len(cands) > 1 or (approximate and cands):
            lines = "\n".join(f"{t['transaction_id']} | {t['date']} | {t['merchant']} | "
                              f"{t.get('category', '')} | {t['amount']:.2f} | {t['status']}"
                              for t in cands)
            ids = self.llm.json(PICK_PROMPT.format(today=today, request=request, lines=lines),
                                None, PICK_SCHEMA).get("transaction_ids") or []
            cands = [t for t in cands if t["transaction_id"] in ids]
        if len({(t["merchant"], t["date"], t["amount"]) for t in cands}) == 1:
            cands = cands[:1]
        if not cands:
            return self._escalate(f"No matching transaction found: {request}")
        if len(cands) > 1:
            return self._escalate(f"Ambiguous: {len(cands)} charges could match: {request}")

        txn = cands[0]
        reason = (f"possible fraud: {request}" if intent == "fraud" else request)
        p = self._tool("check_dispute_policy", transaction_id=txn["transaction_id"], reason=reason)
        if p["decision"] == "eligible":
            r = self._tool("open_dispute", transaction_id=txn["transaction_id"], reason=reason)
            return self._done(f"I've opened dispute {r['dispute_id']} for the {txn['merchant']} "
                              f"charge of ${txn['amount']:.2f}. {r['message']}")
        if p["decision"] == "needs_review":
            return self._escalate(f"{p['rule']}: {txn['merchant']} ${txn['amount']:.2f}")
        return self._done(f"I can't open a dispute on the {txn['merchant']} charge of "
                          f"${txn['amount']:.2f}. {p['explanation']}")
