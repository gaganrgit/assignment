"""
generate_dataset.py
Generates raw_conversations.jsonl — a messy synthetic dataset of 100 EMI collection
call centre conversations with intentionally injected quality issues.

Injected issues (documented):
  - Empty/whitespace-only turns: conv_011, conv_022, conv_033, conv_044, conv_055
  - Duplicate consecutive turns: conv_012, conv_023, conv_034, conv_045, conv_056
  - Fewer than 2 turns: conv_013, conv_024, conv_035
  - Missing/invalid metadata (null outcome, negative duration): conv_014, conv_025, conv_036, conv_047, conv_058
  - Language label mismatches: conv_015, conv_026, conv_037, conv_048, conv_059
  - Garbled/mixed encoding characters: conv_016, conv_027, conv_038, conv_049, conv_060
"""

import json
import random

random.seed(42)

OUTCOMES = ["payment_committed", "callback_scheduled", "escalated", "no_resolution"]

# Clean turn pools by language
HINDI_TURNS = [
    ("agent", "Namaste, main Kapture Finance se bol raha hoon. Kya aap Mr. Sharma hain?"),
    ("customer", "Haan, main hi bol raha hoon."),
    ("agent", "Sir, aapki EMI is mahine pending hai. Kya aap aaj payment kar sakte hain?"),
    ("customer", "Mujhe thoda time chahiye, kal tak karunga."),
    ("agent", "Theek hai sir, hum aapko kal reminder bhejenge."),
    ("customer", "Accha, main koshish karunga."),
    ("agent", "Sir, late payment se penalty lagti hai, please aaj hi process kar lijiye."),
    ("customer", "Mera account abhi short hai, kya aap 2 din de sakte hain?"),
    ("agent", "Bilkul sir, 2 din ka extension de dete hain, lekin please confirm karein."),
    ("customer", "Haan, pakka karta hoon. Thank you."),
]

HINGLISH_TURNS = [
    ("agent", "Hello sir, main Kapture CX se calling kar raha hoon aapki EMI ke baare mein."),
    ("customer", "Haan bolo, kya hua?"),
    ("agent", "Sir, aapki 3,500 rupees ki EMI due hai. Kya aap aaj pay kar sakte ho?"),
    ("customer", "Yaar abhi thoda tight chal raha hai, ek hafte baad karunga."),
    ("agent", "Sir, ek hafte mein extra charges lag jayenge. Kya partial payment possible hai?"),
    ("customer", "Okay, 1,000 aaj de sakta hoon, baaki baad mein."),
    ("agent", "Perfect sir, partial payment accept ho jayegi. Kaunsa payment mode prefer karoge?"),
    ("customer", "UPI se kar deta hoon."),
    ("agent", "Great sir! Main abhi UPI link bhejta hoon aapke registered number pe."),
    ("customer", "Done bhai, payment ho gayi. Thanks."),
    ("agent", "Thank you sir! Confirmation SMS aa jayega shortly."),
]

ENGLISH_TURNS = [
    ("agent", "Good morning, this is Kapture Finance calling regarding your overdue EMI."),
    ("customer", "Yes, I know. I have been meaning to call you."),
    ("agent", "Your EMI of Rs 4,200 is overdue by 10 days. Can we process the payment today?"),
    ("customer", "I just got paid today, I can do it right now."),
    ("agent", "That's great to hear. Would you prefer net banking or UPI?"),
    ("customer", "Let me use my card actually."),
    ("agent", "Sure, I can send you a payment link on your registered mobile number."),
    ("customer", "Please do. I'll pay as soon as I get it."),
    ("agent", "Link has been sent. Is there anything else I can help you with?"),
    ("customer", "No, that's all. Thank you."),
    ("agent", "Thank you for your cooperation. Have a good day!"),
]

GARBLED_SAMPLES = [
    "Kal\x92 tak payment kar\x85 deta\x00 hoon",
    "Sir aapki EMI \xff\xfe pending hai",
    "Main abhi \x80\x81 nahin kar sakta",
    "Payment \x93link\x94 bhejo please",
    "Okay \xc3\xa9\xc3\xa0 kar deta hoon",
]


def make_turns(pool, n=None):
    if n is None:
        n = random.randint(4, 8)
    selected = random.sample(pool, min(n, len(pool)))
    # Ensure alternating roles
    result = []
    for i, (role, text) in enumerate(selected):
        result.append({"role": role, "text": text})
    return result


def clean_conv(conv_id, language):
    pool = {"hindi": HINDI_TURNS, "hinglish": HINGLISH_TURNS, "english": ENGLISH_TURNS}[language]
    return {
        "conversation_id": conv_id,
        "language": language,
        "turns": make_turns(pool),
        "metadata": {
            "call_duration_seconds": random.randint(60, 480),
            "outcome": random.choice(OUTCOMES),
        },
    }


conversations = []

# ── 60 clean conversations ────────────────────────────────────────────────────
languages = ["hindi", "hinglish", "english"]
for i in range(1, 61):
    lang = languages[(i - 1) % 3]
    conv_id = f"conv_{i:03d}"
    # Skip IDs reserved for dirty data (011-060 that overlap with dirty IDs below)
    if i in {11,12,13,14,15,16,22,23,24,25,26,27,33,34,35,36,37,38,44,45,47,48,49,55,56,58,59,60}:
        continue
    conversations.append(clean_conv(conv_id, lang))

# ── Dirty conversations (documented) ─────────────────────────────────────────

# 1. Empty / whitespace-only turns (conv_011, 022, 033, 044, 055)
for cid in ["conv_011", "conv_022", "conv_033", "conv_044", "conv_055"]:
    lang = random.choice(languages)
    pool = {"hindi": HINDI_TURNS, "hinglish": HINGLISH_TURNS, "english": ENGLISH_TURNS}[lang]
    turns = make_turns(pool, 4)
    # Inject an empty turn
    turns.insert(random.randint(1, len(turns)), {"role": "agent", "text": "   "})
    conversations.append({
        "conversation_id": cid,
        "language": lang,
        "turns": turns,
        "metadata": {"call_duration_seconds": random.randint(60, 480), "outcome": random.choice(OUTCOMES)},
    })

# 2. Duplicate consecutive turns (conv_012, 023, 034, 045, 056)
for cid in ["conv_012", "conv_023", "conv_034", "conv_045", "conv_056"]:
    lang = random.choice(languages)
    pool = {"hindi": HINDI_TURNS, "hinglish": HINGLISH_TURNS, "english": ENGLISH_TURNS}[lang]
    turns = make_turns(pool, 4)
    # Inject duplicate consecutive turn
    dup_idx = random.randint(0, len(turns) - 1)
    turns.insert(dup_idx + 1, turns[dup_idx].copy())
    conversations.append({
        "conversation_id": cid,
        "language": lang,
        "turns": turns,
        "metadata": {"call_duration_seconds": random.randint(60, 480), "outcome": random.choice(OUTCOMES)},
    })

# 3. Fewer than 2 turns (conv_013, 024, 035)
for cid in ["conv_013", "conv_024", "conv_035"]:
    lang = random.choice(languages)
    pool = {"hindi": HINDI_TURNS, "hinglish": HINGLISH_TURNS, "english": ENGLISH_TURNS}[lang]
    n_turns = random.choice([0, 1])
    turns = make_turns(pool, n_turns) if n_turns else []
    conversations.append({
        "conversation_id": cid,
        "language": lang,
        "turns": turns,
        "metadata": {"call_duration_seconds": random.randint(60, 480), "outcome": random.choice(OUTCOMES)},
    })

# 4. Missing / invalid metadata (conv_014, 025, 036, 047, 058)
invalid_meta_cases = {
    "conv_014": {"call_duration_seconds": None, "outcome": random.choice(OUTCOMES)},
    "conv_025": {"call_duration_seconds": -120, "outcome": random.choice(OUTCOMES)},
    "conv_036": {"call_duration_seconds": 200, "outcome": None},
    "conv_047": {"call_duration_seconds": 0, "outcome": "unknown_outcome"},
    "conv_058": {},  # missing both fields
}
for cid, meta in invalid_meta_cases.items():
    lang = random.choice(languages)
    pool = {"hindi": HINDI_TURNS, "hinglish": HINGLISH_TURNS, "english": ENGLISH_TURNS}[lang]
    conversations.append({
        "conversation_id": cid,
        "language": lang,
        "turns": make_turns(pool),
        "metadata": meta,
    })

# 5. Language label mismatches (conv_015, 026, 037, 048, 059)
mismatch_cases = [
    ("conv_015", "hindi", ENGLISH_TURNS),    # labelled hindi, clearly english
    ("conv_026", "english", HINDI_TURNS),    # labelled english, clearly hindi
    ("conv_037", "hinglish", ENGLISH_TURNS), # labelled hinglish, all english
    ("conv_048", "hindi", ENGLISH_TURNS),
    ("conv_059", "english", HINDI_TURNS),
]
for cid, wrong_lang, pool in mismatch_cases:
    conversations.append({
        "conversation_id": cid,
        "language": wrong_lang,
        "turns": make_turns(pool),
        "metadata": {"call_duration_seconds": random.randint(60, 480), "outcome": random.choice(OUTCOMES)},
    })

# 6. Garbled / mixed encoding characters (conv_016, 027, 038, 049, 060)
for cid in ["conv_016", "conv_027", "conv_038", "conv_049", "conv_060"]:
    lang = random.choice(languages)
    pool = {"hindi": HINDI_TURNS, "hinglish": HINGLISH_TURNS, "english": ENGLISH_TURNS}[lang]
    turns = make_turns(pool, 4)
    # Replace one turn text with garbled sample
    garbled_idx = random.randint(0, len(turns) - 1)
    turns[garbled_idx]["text"] = random.choice(GARBLED_SAMPLES)
    conversations.append({
        "conversation_id": cid,
        "language": lang,
        "turns": turns,
        "metadata": {"call_duration_seconds": random.randint(60, 480), "outcome": random.choice(OUTCOMES)},
    })

# ── Fill remaining slots up to exactly 100 ───────────────────────────────────
existing_ids = {c["conversation_id"] for c in conversations}
filler_i = 61
while len(conversations) < 100:
    cid = f"conv_{filler_i:03d}"
    if cid not in existing_ids:
        lang = languages[(filler_i - 1) % 3]
        conversations.append(clean_conv(cid, lang))
        existing_ids.add(cid)
    filler_i += 1

# Shuffle so dirty conversations aren't all at the end
random.shuffle(conversations)

# Write JSONL
output_path = "raw_conversations.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for conv in conversations:
        f.write(json.dumps(conv, ensure_ascii=False) + "\n")

print(f"Generated {len(conversations)} conversations → {output_path}")
