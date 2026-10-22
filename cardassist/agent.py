"""The agent loop.

    user message
      -> input guardrails (decline card-data / injection / off-topic requests)
      -> LLM chooses tools; tools run on SQLite; results go back to the LLM
      -> repeat until the LLM answers in text (max MAX_AGENT_STEPS)
      -> output guardrail (mask card numbers)

The customer_id comes from the logged-in session and is injected into every tool call,
so the model can never read or act on another customer's data.

Any object with `.chat(messages, tools) -> {"content": str, "tool_calls": [...]}` works
as the LLM, which is how the tests swap in a fake model.
"""
import json
from dataclasses import dataclass, field

from . import config, guardrails, tools

SYSTEM_PROMPT = """You are CardAssist, a customer-service agent for credit card charges and disputes.
Today is {today}. You are helping customer {customer_id}; their identity is already verified.

How to work:
1. To find a charge, call get_transactions (use date_range 'last_90_days' unless the customer gives dates).
2. Pick the transaction that matches the amount, merchant and date the customer describes.
3. Call check_dispute_policy with that transaction_id and the customer's reason.
4. If the decision is "eligible", call open_dispute. If it is "needs_review", call escalate_to_human.
   If it is "ineligible", do not call any more tools; explain the rule to the customer.
5. Call escalate_to_human directly when: the customer mentions fraud, a stolen card or a hacked
   account without naming a specific charge; the request is too unclear to identify a charge;
   no transaction matches; or several transactions match and you cannot tell which one.
6. If the customer only asks to see transactions, call get_transactions and summarise them.

Rules:
- Never show full card numbers, CVV or expiry dates. Refer to cards by last four digits only.
- Only help with card charges and disputes. Politely decline anything else (investments,
  general questions, writing tasks).
- Never open a dispute without checking the policy first. Never invent transaction IDs.
- Looking up transactions is never the last step of a dispute: after get_transactions, continue
  with check_dispute_policy or escalate_to_human. Don't ask the customer to confirm first.
- Keep answers short and tell the customer what happened and what comes next."""


NEXT_STEP_REMINDER = """Check before you answer. Re-read the customer's message:
- If they want to dispute a charge, don't recognize it, were charged twice, or want money back:
  find the matching transaction_id in the get_transactions result and call check_dispute_policy now.
- If no transaction matches, several match, or they mention fraud: call escalate_to_human now.
- Only if they just wanted to see their transactions: reply to them in text."""


@dataclass
class AgentResult:
    reply: str
    tool_calls: list = field(default_factory=list)   # [{"name":..., "arguments":..., "result":...}]
    outcome: str = "info"   # dispute_opened | escalated | policy_denied | declined | info | no_action
    guardrail: str | None = None

    @property
    def tools_used(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]

    @property
    def escalated(self) -> bool:
        return self.outcome == "escalated"


class OllamaLLM:
    """Llama 3.1 through a local Ollama server."""

    def __init__(self, model: str = config.OLLAMA_MODEL, host: str = config.OLLAMA_HOST):
        import ollama
        self.client = ollama.Client(host=host)
        self.model = model

    def chat(self, messages, tools_schema):
        resp = self.client.chat(model=self.model, messages=messages, tools=tools_schema,
                                options={"temperature": 0})
        msg = resp["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": fn["name"], "arguments": dict(args)})
        return {"content": msg.get("content") or "", "tool_calls": calls}


def _run_tool(name: str, args: dict, customer_id: str, db_path) -> dict:
    fn = tools.TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name!r}."}
    args = {k: v for k, v in (args or {}).items() if k in fn.__code__.co_varnames}
    args.pop("db_path", None)
    if "customer_id" in fn.__code__.co_varnames:
        args["customer_id"] = customer_id           # always the session's customer
    try:
        return fn(**args, db_path=db_path)
    except TypeError as e:                          # missing/bad arguments from the model
        return {"error": f"Bad arguments for {name}: {e}"}


def _outcome(calls: list[dict]) -> str:
    if not calls:
        return "no_action"          # the model answered in text only (declined or asked a question)
    for c in reversed(calls):
        r = c["result"]
        if c["name"] == "open_dispute" and "dispute_id" in r:
            return "dispute_opened"
        if c["name"] == "escalate_to_human" and "ticket_id" in r:
            return "escalated"
    for c in reversed(calls):
        if c["name"] == "check_dispute_policy" and c["result"].get("decision") == "ineligible":
            return "policy_denied"
    return "info"


class CardAssistAgent:
    def __init__(self, llm, customer_id: str, db_path=None, nudge: bool = True):
        self.llm = llm
        self.customer_id = customer_id
        self.db_path = db_path
        self.nudge = nudge          # False reproduces the v1 behaviour, for before/after evals

    def run(self, user_message: str) -> AgentResult:
        blocked = guardrails.check_input(user_message)
        if blocked:
            category, reply = blocked
            return AgentResult(reply=reply, outcome="declined", guardrail=category)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                today=config.TODAY.isoformat(), customer_id=self.customer_id)},
            {"role": "user", "content": user_message},
        ]
        calls: list[dict] = []
        reply = ""
        nudged = False
        for _ in range(config.MAX_AGENT_STEPS):
            resp = self.llm.chat(messages, tools.TOOL_SCHEMAS)
            if not resp["tool_calls"]:
                # Small models often stop after the lookup instead of finishing the workflow.
                # Remind them once of the next step; an info-only request just answers again.
                if self.nudge and not nudged and [c["name"] for c in calls] == ["get_transactions"]:
                    nudged = True
                    messages.append({"role": "assistant", "content": resp["content"]})
                    messages.append({"role": "system", "content": NEXT_STEP_REMINDER})
                    continue
                reply = resp["content"]
                break
            messages.append({"role": "assistant", "content": resp["content"],
                             "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in resp["tool_calls"]]})
            for c in resp["tool_calls"]:
                result = _run_tool(c["name"], c["arguments"], self.customer_id, self.db_path)
                calls.append({"name": c["name"], "arguments": c["arguments"], "result": result})
                messages.append({"role": "tool", "tool_name": c["name"], "content": json.dumps(result)})
        else:
            reply = "I couldn't finish this request, so I've passed it to a specialist."
            result = tools.escalate_to_human("agent step limit reached", self.customer_id, self.db_path)
            calls.append({"name": "escalate_to_human", "arguments": {}, "result": result})

        outcome = _outcome(calls)
        if not calls and not reply.strip():
            reply = "Sorry, I didn't understand. Could you describe the charge?"
        return AgentResult(reply=guardrails.mask_output(reply), tool_calls=calls, outcome=outcome)
