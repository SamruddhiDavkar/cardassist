"""A deterministic, rule-based stand-in for the LLM.

It speaks the same interface as OllamaLLM (reads the message history, returns tool calls
in the same format), so it exercises the full agent loop. Uses:
  * CI, where Ollama can't run
  * a baseline to compare Llama 3.1 against in the evaluation
It is a keyword parser, not a language model, and it is brittle on purpose: if Llama 3.1
can't beat it, the LLM isn't earning its place.
"""
import json
import re
from datetime import date, timedelta

from . import config, policy

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
INFO_RE = re.compile(r"^(show|list|what did i|did i|how much|when did|see)\b|\bmy transactions\b",
                     re.IGNORECASE)
DUPLICATE_RE = re.compile(r"\b(twice|double|duplicate|two times)\b", re.IGNORECASE)
AMOUNT_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{2})?)")
STOPWORDS = {"the", "a", "an", "i", "my", "me", "from", "at", "on", "for", "charge", "charged",
             "dispute", "this", "that", "there", "is", "was", "of", "by", "to", "and", "it",
             "can", "you", "please", "purchase", "don't", "dont", "recognize", "never",
             "what", "want", "charges", "card", "didn't", "got", "even", "though", "but"}


def _last_user(messages):
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


def _tool_results(messages):
    """Pair each tool message with the call that produced it."""
    names = [tc["function"]["name"] for m in messages if m["role"] == "assistant"
             for tc in m.get("tool_calls") or []]
    results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    return list(zip(names, results))


def _parse_amount(text):
    m = AMOUNT_RE.search(text)
    return float(m.group(1).replace(",", "")) if m else None


def _parse_date(text, today):
    t = text.lower()
    for i, d in enumerate(WEEKDAYS):
        if f"last {d}" in t:
            back = (today.weekday() - i) % 7 or 7
            return today - timedelta(days=back)
    for i, mo in enumerate(MONTHS, start=1):
        m = re.search(rf"\b{mo}\s+(\d{{1,2}})\b", t)
        if m:
            return date(today.year, i, int(m.group(1)))
    if "yesterday" in t:
        return today - timedelta(days=1)
    return None


def _merchant_hits(text, txns):
    words = {w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOPWORDS and len(w) > 2}
    merchants = {t["merchant"] for t in txns}
    return {m for m in merchants if words & set(m.lower().split())}


def _call(name, **args):
    return {"content": "", "tool_calls": [{"name": name, "arguments": args}]}


def _say(text):
    return {"content": text, "tool_calls": []}


class RuleBasedLLM:
    def chat(self, messages, tools_schema=None):
        text = _last_user(messages)
        done = _tool_results(messages)
        today = config.TODAY

        if not done:
            return self._first_step(text)

        name, result = done[-1]
        if "error" in result and name != "open_dispute":
            return _call("escalate_to_human", reason=f"Tool error: {result['error']}")

        if name == "get_transactions":
            return self._after_lookup(text, result, today)
        if name == "check_dispute_policy":
            d, txn = result["decision"], result["transaction"]
            if d == "eligible":
                return _call("open_dispute", transaction_id=txn["transaction_id"], reason=text)
            if d == "needs_review":
                return _call("escalate_to_human",
                             reason=f"{result['rule']}: {txn['merchant']} ${txn['amount']:.2f}")
            return _say(f"I can't open a dispute on the {txn['merchant']} charge of "
                        f"${txn['amount']:.2f}. {result['explanation']}")
        if name == "open_dispute":
            if "dispute_id" in result:
                t = result["transaction"]
                return _say(f"I've opened dispute {result['dispute_id']} for the {t['merchant']} "
                            f"charge of ${t['amount']:.2f}. {result['message']}")
            return _say(f"I couldn't open the dispute: {result.get('explanation', result['error'])}")
        if name == "escalate_to_human":
            return _say(f"I've passed this to a specialist (ticket {result['ticket_id']}). "
                        f"{result['message']}")
        return _say("Done.")

    def _first_step(self, text):
        has_detail = _parse_amount(text) is not None or bool(
            re.search(r"\b[A-Z][a-z]+[A-Z]?[a-z]*\b", text[1:]))
        if policy.is_fraud_reason(text) and _parse_amount(text) is None:
            return _call("escalate_to_human", reason=f"Possible fraud, no specific charge: {text}")
        if INFO_RE.search(text):
            days = re.search(r"last (\d+) days", text.lower())
            rng = f"last_{days.group(1)}_days" if days else (
                "last_7_days" if "this week" in text.lower() else "last_90_days")
            return _call("get_transactions", date_range=rng)
        if not has_detail:
            return _call("escalate_to_human", reason=f"Unclear request: {text}")
        return _call("get_transactions", date_range="last_90_days")

    def _after_lookup(self, text, result, today):
        txns = result["transactions"]
        if INFO_RE.search(text) and not re.search(r"\bdispute|recogni", text, re.I):
            hits = _merchant_hits(text, txns)
            shown = [t for t in txns if not hits or t["merchant"] in hits]
            if not shown:
                return _say("I didn't find any matching transactions in that period.")
            lines = [f"- {t['date']}  {t['merchant']}  ${t['amount']:.2f} ({t['status']})"
                     for t in shown[:15]]
            return _say(f"Here are {len(shown)} transaction(s) on card {result['card']}:\n"
                        + "\n".join(lines))

        amount = _parse_amount(text)
        hits = _merchant_hits(text, txns)
        cands = [t for t in txns
                 if (amount is None or abs(t["amount"] - amount) < 0.005)
                 and (not hits or t["merchant"] in hits)]
        if amount is None and not hits:
            cands = []
        when = _parse_date(text, today)
        if when and len(cands) > 1:
            dated = [t for t in cands if t["date"] == when.isoformat()]
            cands = dated or cands
        if len(cands) == 2 and DUPLICATE_RE.search(text) and \
                cands[0]["date"] == cands[1]["date"]:
            cands = [cands[0]]           # dispute one copy of the duplicate
        if not cands:
            return _call("escalate_to_human", reason=f"No matching transaction found: {text}")
        if len(cands) > 1:
            return _call("escalate_to_human", reason=f"Ambiguous: {len(cands)} matching charges")
        return _call("check_dispute_policy", transaction_id=cands[0]["transaction_id"], reason=text)
