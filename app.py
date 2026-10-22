"""Streamlit chat UI.   streamlit run app.py"""
import streamlit as st

from cardassist import config
from cardassist.agent import CardAssistAgent, OllamaLLM
from cardassist.db import connect
from cardassist.pipeline import OllamaJSON, SimplePipeline
from cardassist.rule_llm import RuleBasedLLM
from data.generate_data import build

st.set_page_config(page_title="CardAssist", page_icon="💳")
st.title("💳 CardAssist")
st.caption("Credit card dispute agent. Demo data only, no real cards.")

if not config.DB_PATH.exists():
    build()

with st.sidebar:
    with connect() as c:
        customers = [r["customer_id"] for r in c.execute("SELECT customer_id FROM customers")]
    customer = st.selectbox("Signed in as", customers, index=0)
    backend = st.radio("Mode", ["Simple: Llama understands, code acts (recommended)",
                                "Agent: Llama chains the tools itself", "Rule-based baseline"])
    show_trace = st.toggle("Show tool calls", value=True)
    if st.button("Reset demo data"):
        build()
        st.session_state.history = []
    st.markdown("**Try:**\n- I don't recognize a $49.99 charge from last Tuesday\n"
                "- Dispute the $849 BestTech Store purchase\n- Show my last 7 days\n"
                "- Give me investment advice")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_trace and msg.get("trace"):
            with st.expander(f"Tool calls · outcome: {msg['outcome']}"):
                st.json(msg["trace"])

if prompt := st.chat_input("Describe the charge…"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        if backend.startswith("Simple"):
            result = SimplePipeline(OllamaJSON(), customer).run(prompt)
        else:
            llm = OllamaLLM() if backend.startswith("Agent") else RuleBasedLLM()
            result = CardAssistAgent(llm, customer).run(prompt)
        reply, trace, outcome = result.reply, result.tool_calls, result.outcome
    except Exception as e:  # e.g. Ollama not running
        reply, trace, outcome = f"⚠️ Couldn't reach the model: {e}", [], "error"
    st.session_state.history.append(
        {"role": "assistant", "content": reply, "trace": trace, "outcome": outcome})
    st.rerun()
