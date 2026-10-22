"""Input and output guardrails.

Input:  block requests for card data, prompt-injection attempts, and clearly
        off-topic requests before they reach the model.
Output: mask anything that looks like a card number in whatever the agent says.
"""
import re

# 13-19 digits, optionally separated by spaces or dashes
PAN_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

CARD_DATA_PATTERNS = [
    r"\b(full|whole|complete|entire)\b.*\b(card|account)\s*(number|#|no)",
    r"\b(card|account)\s*(number|#|no)\b.*\b(show|tell|give|read|reveal|display|what)",
    r"\b(show|tell|give|read|reveal|display|what('s| is))\b.*\b(card|account)\s*(number|#|no)",
    r"\bcvv\b|\bcvc\b|\bsecurity code\b|\bpin\b",
    r"\bexpiration date\b|\bexpiry\b",
    r"\b(all|every|16|sixteen)\b.*\bdigits\b",
]

INJECTION_PATTERNS = [
    r"ignore (all |your |the )?(previous|prior|above) (instructions|rules|prompt)",
    r"disregard (your|the) (rules|instructions|policy)",
    r"you are now\b", r"system prompt", r"developer mode", r"jailbreak",
    r"\ball (of )?my transactions\b.*\bdispute|\bdispute\b.*\ball (of )?my transactions",
]

OFF_TOPIC_PATTERNS = [
    r"\binvest(ing|ment|ments)?\b", r"\bstocks?\b", r"\bcrypto(currency)?\b", r"\bbitcoin\b",
    r"\bweather\b", r"\bpoem\b", r"\bsong\b", r"\brecipe\b", r"\bjoke\b",
    r"\bhomework\b", r"\bsports?\b", r"\bmovie recommendation", r"\bwrite (me )?(a|an) (essay|story)",
]

DECLINE_CARD_DATA = ("For your security I can't show full card numbers, CVV codes or other card "
                     "credentials. I can only show the last four digits. I'm happy to help with a "
                     "charge or a dispute instead.")
DECLINE_INJECTION = ("I can only act on your card charges one request at a time, within the "
                     "dispute rules. Tell me which charge you'd like help with.")
DECLINE_OFF_TOPIC = ("I can only help with charges and disputes on your card, so I can't help "
                     "with that. Is there a charge you'd like me to look into?")


def _any(patterns, text):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def check_input(text: str) -> tuple[str, str] | None:
    """Return (category, reply) if the request must be declined, else None."""
    if _any(INJECTION_PATTERNS, text):
        return "injection", DECLINE_INJECTION
    if _any(CARD_DATA_PATTERNS, text):
        return "card_data", DECLINE_CARD_DATA
    if _any(OFF_TOPIC_PATTERNS, text):
        return "off_topic", DECLINE_OFF_TOPIC
    return None


def mask_card(number: str) -> str:
    digits = re.sub(r"\D", "", number)
    return f"**** **** **** {digits[-4:]}"


def mask_output(text: str) -> str:
    return PAN_RE.sub(lambda m: mask_card(m.group()), text or "")


def contains_pan(text: str) -> bool:
    return bool(PAN_RE.search(text or ""))
