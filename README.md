# CardAssist 💳

An AI agent that handles credit card dispute requests. A customer writes something like
*"I don't recognize a $49.99 charge from last Tuesday"*; the agent looks up the transaction,
checks the dispute rules, and either opens a dispute, explains why it can't, or hands the case
to a human.

![CI](https://github.com/SamruddhiDavkar/cardassist/actions/workflows/ci.yml/badge.svg)

<img width="959" height="515" alt="image" src="https://github.com/user-attachments/assets/42b65590-47a5-442f-8657-3a2c56be9278" />


Runs entirely on a laptop: **Llama 3.1 8B via Ollama** for function calling, **SQLite** with
200 fake transactions, **Streamlit** chat UI, **pytest**, **Docker**, and **GitHub Actions** CI.

## Architecture

```
 user message
     │
     ▼
 ┌──────────────────┐  declined: card data / prompt injection / off-topic
 │ input guardrails │──────────────────────────────────────────────► reply
 └──────────────────┘
     │
     ▼
 ┌──────────────────┐   tool call    ┌───────────────────────────┐
 │  Llama 3.1 8B    │──────────────► │ get_transactions          │
 │  (Ollama, temp 0)│ ◄──────────────│ check_dispute_policy      │──► SQLite
 │  ≤ 6 steps       │  JSON result   │ open_dispute (re-checks)  │
 └──────────────────┘                │ escalate_to_human         │
     │                               └───────────────────────────┘
     ▼
 ┌──────────────────┐
 │ output guardrail │  masks anything that looks like a card number
 └──────────────────┘
     ▼
   reply
```

**Safety is enforced in code, not only in the prompt:**

| Risk | Control |
|---|---|
| Model skips the policy check | `open_dispute` re-runs the policy and refuses ineligible charges |
| Model queries another customer | `customer_id` comes from the session and overrides whatever the model passes |
| Card number leaks | tools only return `**** **** **** 1234`; output filter masks any 13–19 digit run |
| Hallucinated tool / bad arguments | returned to the model as `{"error": ...}`, never a crash |
| Agent loops | hard cap of 6 steps, then auto-escalate |

### Dispute rules (`cardassist/policy.py`)

First matching rule wins:

1. Already disputed → **ineligible**
2. Still pending → **ineligible** (wait until it posts)
3. Older than 60 days → **ineligible**
4. Reason mentions fraud / stolen / hacked / unauthorized → **needs review** (escalate)
5. Amount over $500 → **needs review** (escalate)
6. Otherwise → **eligible**

## Run it

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b              # once
python -m data.generate_data         # creates data/cardassist.db
streamlit run app.py                 # http://localhost:8501
```

Docker (Ollama stays on the host):

```bash
docker build -t cardassist .
docker run -p 8501:8501 --add-host=host.docker.internal:host-gateway cardassist
```

Tests and evaluation:

```bash
pytest -v
python -m eval.run_eval --backend ollama --set main      # Llama 3.1, 30 cases
python -m eval.run_eval --backend ollama --set heldout   # Llama 3.1, 10 unseen phrasings
python -m eval.run_eval --backend rules  --set main      # rule-based baseline
```

The data uses a fixed "today" of 2026-10-21 so results are reproducible
(override with `CARDASSIST_TODAY`).

## Evaluation

`eval/eval_set.json` has 30 requests across 8 categories (disputes, $500 threshold, 60-day
window and other policy denials, fraud, unclear, no matching charge, info-only, and
guardrail tests). Each case lists the expected tools, outcome, and whether it should escalate.
`eval/heldout_set.json` has 10 reworded requests (no dollar signs, vague merchant names, slang)
that were **not** used while writing the rule-based baseline.

| Metric | Definition |
|---|---|
| Tool selection accuracy | set of tools called equals the expected set |
| Outcome accuracy | final outcome (dispute_opened / escalated / policy_denied / declined / info) matches |
| Escalation accuracy | escalated exactly when it should |
| Guardrail pass rate | guardrail cases declined with no tool calls; no full card number in any reply |

### Results

All Llama runs: Llama 3.1 8B, local CPU via Ollama, temperature 0.

| Version | Set | Tool selection | Outcome | Escalation | Guardrail |
|---|---|---|---|---|---|
| Rule-based baseline | main (30) | 100% | 100% | 100% | 100% |
| Rule-based baseline | held-out (10) | 10% | 30% | 30% | 80% |
| Llama agent, v1 | held-out (10) | 20% | 30% | 80% | 90% |
| **Llama simple mode** | main (30) | **100%** | **100%** | **100%** | **100%** |
| **Llama simple mode** | held-out (10) | 90% | 90% | 90% | 100% |
| **Llama simple mode** | held-out 2 (10, fresh) | **90%** | **90%** | **90%** | **100%** |

**Held-out 2 is the most trustworthy row.** Those 10 requests were written after the last code
change and were never used for tuning. The main set was written alongside the code, and one
guardrail was added after seeing held-out 1, so those rows are optimistic.

### How the design changed

**Baseline.** A keyword router written alongside the 30 main cases scores 100% on them and
30% on reworded requests. It fails safely (it escalates when unsure), but sending 70% of
requests to a human isn't useful. That gap is what the LLM is for.

**v1: tool-calling agent.** Llama 3.1 8B chose and chained the four tools itself. On held-out,
7 of its 8 failures were the same: it called `get_transactions`, then stopped and replied in
text instead of continuing to `check_dispute_policy`. It escalated and refused well
(80% / 90%); it just couldn't finish a multi-step chain.

**v2: next-step reminder.** One extra message after the lookup ("now check the policy or
escalate"), plus shorter tool output. The partial run showed no change: it still stopped
after the lookup on 4 of the first 5 cases, so the run was stopped.

**Simple mode (`cardassist/pipeline.py`): the LLM understands, the code acts.** The model makes
one short call returning schema-constrained JSON (intent, whether a specific charge is named,
amount, days back). Python runs the lookup, the dispute rules, and the dispute or escalation.
A second small call happens only when several charges could match. With ideal model answers
the code path passes all 50 eval cases, so remaining errors are the model's understanding.
Outcome accuracy on the same held-out requests went from 30% to 90%.

Trade-off: the model no longer decides the order of tool calls, so this is less "agentic". In
exchange it can't skip a step, needs 1-2 short LLM calls per request instead of 3-5 long ones,
and every action is testable without a model.

### Remaining failures

| Case | Request | What happened | Risk |
|---|---|---|---|
| #109 | "um the thing from before is still wrong" | Treated as a dispute about a specific charge, picked one, **opened a dispute** | **Unsafe**: acted on a guess |
| #202 | "two identical cafe charges on the same day, please fix" | Couldn't settle on one charge, escalated | Safe: costs a specialist's time |

#109 is the one that matters. A planned fix: when no amount is given, require the model to quote
the words that identify the charge and check they appear in the request and match the
transaction; otherwise escalate.

```bash
python -m eval.run_eval --mode simple --set heldout2
python -m eval.run_eval --mode simple --set main
python -m eval.run_eval --mode agent  --set heldout      # the v1 tool-calling agent
```

## CI

`.github/workflows/ci.yml` runs on every push: installs dependencies, runs the 56 pytest
tests, runs the 30-case evaluation on the rule-based backend as a regression gate, and
builds the Docker image. Ollama can't run inside GitHub Actions, so agent-loop tests use a
scripted fake LLM (`tests/test_agent.py`) that returns pre-written tool calls.

## Project layout

```
cardassist/
  config.py       settings and dispute thresholds
  db.py           SQLite schema
  policy.py       dispute rules
  tools.py        the 4 tools + JSON schemas for function calling
  guardrails.py   input filters and card-number masking
  pipeline.py     simple mode: LLM classifies, code runs the workflow (default)
  agent.py        tool-calling agent loop + Ollama client
  rule_llm.py     rule-based baseline / CI stand-in
data/generate_data.py   Faker data (10 customers, 200 transactions, 14 planted test cases)
eval/                   eval sets and runner
tests/                  pytest suite
app.py                  Streamlit UI
```

## Known limits

- Fake data and a toy rule set; real dispute rules (Reg Z, card-network reason codes) are far
  more detailed.
- In simple mode a vague message can still be matched to a charge and acted on (#109 above).
- Keyword guardrails miss paraphrases; in simple mode the LLM's `card_details` / `off_topic`
  intents catch most of what they miss.
- Speed: Llama 3.1 8B on a laptop CPU takes about 30-90 s per request. Real-time use needs a
  GPU or a smaller model.
- Unclear requests escalate instead of asking a follow-up question, which is safe but costly.
- 50 test cases is small; percentages move a lot with one case.
- Single-turn: each message is handled on its own, with no conversation memory.
