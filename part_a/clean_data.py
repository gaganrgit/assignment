"""
clean_data.py
-------------
Reads  : raw_conversations.jsonl
Outputs:
  cleaned_conversations.jsonl  — valid, training-ready conversations
  rejected_conversations.jsonl — conversations that failed checks (with rejection_reason)

Quality checks performed:
  1. JSON parsability / required top-level fields
  2. Empty or whitespace-only turns
  3. Duplicate consecutive turns (same role + same text)
  4. Fewer than 2 turns (after cleaning empty turns)
  5. Missing or invalid metadata (null/missing fields, negative/zero duration,
     unrecognised outcome values)
  6. Language label mismatch (detected via simple keyword heuristics)
  7. Garbled / high-ratio non-printable / control characters in turn text
  8. Duplicate conversation IDs across the dataset
  9. Turns with unrecognised roles (anything other than "agent" / "customer")
"""

import json
import re
import unicodedata
from pathlib import Path
from collections import Counter

# ── Constants ─────────────────────────────────────────────────────────────────

RAW_PATH      = Path("raw_conversations.jsonl")
CLEAN_PATH    = Path("cleaned_conversations.jsonl")
REJECTED_PATH = Path("rejected_conversations.jsonl")

VALID_LANGUAGES = {"hindi", "hinglish", "english"}
VALID_OUTCOMES  = {"payment_committed", "callback_scheduled", "escalated", "no_resolution"}
VALID_ROLES     = {"agent", "customer"}

# Keyword heuristics for language detection
HINDI_KEYWORDS    = {"haan", "nahin", "kya", "aap", "main", "hai", "kal", "sir",
                     "theek", "accha", "namaste", "nahi", "karunga", "please",
                     "abhi", "karein", "raha", "hoon", "bhi", "toh"}
ENGLISH_ONLY_WORDS = {"your", "you", "the", "this", "is", "are", "have", "been",
                      "can", "will", "would", "please", "thank", "payment", "today",
                      "overdue", "bank", "morning", "good", "help", "process"}

# Max share of non-printable / control characters before flagging as garbled
GARBLED_THRESHOLD = 0.10   # 10 %
MIN_TURNS_REQUIRED = 2


# ── Helper functions ──────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file; skip lines that are not valid JSON (logged as parse errors)."""
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                # Can't recover — treat as rejected later, but keep raw text
                records.append({"_parse_error": str(exc), "_raw_line": lineno})
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def is_garbled(text: str) -> bool:
    """Return True if the ratio of non-printable / replacement characters is too high."""
    if not text:
        return False
    bad = sum(
        1 for ch in text
        if unicodedata.category(ch) in ("Cc", "Cs")   # control / surrogate
        or ch in ("\ufffd", "\x00", "\xff", "\xfe")
    )
    return (bad / len(text)) > GARBLED_THRESHOLD


def detect_language(turns: list[dict]) -> str:
    """
    Heuristic language detection based on combined turn text.
    Returns 'hindi', 'hinglish', 'english', or 'unknown'.
    """
    all_text = " ".join(t.get("text", "") for t in turns).lower()
    words = set(re.findall(r"[a-zA-Z]+", all_text))

    hindi_hits   = len(words & HINDI_KEYWORDS)
    english_hits = len(words & ENGLISH_ONLY_WORDS)

    if hindi_hits == 0 and english_hits == 0:
        return "unknown"
    if hindi_hits > 0 and english_hits > 3:
        return "hinglish"
    if english_hits > hindi_hits * 2:
        return "english"
    return "hindi"


def validate_metadata(meta) -> list[str]:
    """
    Returns a list of issues found in the metadata dict.
    Empty list means metadata is valid.
    """
    issues = []
    if not isinstance(meta, dict):
        issues.append("metadata is not a dict")
        return issues

    duration = meta.get("call_duration_seconds")
    outcome  = meta.get("outcome")

    if duration is None:
        issues.append("call_duration_seconds is missing/null")
    elif not isinstance(duration, (int, float)):
        issues.append(f"call_duration_seconds has invalid type: {type(duration).__name__}")
    elif duration <= 0:
        issues.append(f"call_duration_seconds is non-positive ({duration})")

    if outcome is None:
        issues.append("outcome is missing/null")
    elif outcome not in VALID_OUTCOMES:
        issues.append(f"outcome '{outcome}' is not a recognised value")

    return issues


# ── Core cleaning logic ───────────────────────────────────────────────────────

def clean_turns(turns: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Clean a turn list in-place (non-destructively).
    Returns (cleaned_turns, list_of_issues_found).
    """
    issues = []

    # 1. Remove turns with unrecognised roles
    unknown_roles = [t for t in turns if t.get("role") not in VALID_ROLES]
    if unknown_roles:
        issues.append(f"removed {len(unknown_roles)} turn(s) with unrecognised role")
    turns = [t for t in turns if t.get("role") in VALID_ROLES]

    # 2. Remove empty / whitespace-only turns
    empty = [t for t in turns if not t.get("text", "").strip()]
    if empty:
        issues.append(f"removed {len(empty)} empty/whitespace turn(s)")
    turns = [t for t in turns if t.get("text", "").strip()]

    # 3. Remove duplicate consecutive turns
    deduped = []
    for turn in turns:
        if deduped and deduped[-1]["role"] == turn["role"] and deduped[-1]["text"] == turn["text"]:
            if "removed duplicate consecutive turn" not in " ".join(issues):
                issues.append("removed duplicate consecutive turn(s)")
        else:
            deduped.append(turn)
    turns = deduped

    return turns, issues


def process_conversation(conv: dict, seen_ids: set) -> tuple[str, list[str]]:
    """
    Validate a conversation record.
    Returns ('accept' | 'reject', list_of_rejection_reasons).
    Mutates conv in place when fixing recoverable issues (turn cleaning).
    """
    reasons = []

    # ── Parse error ───────────────────────────────────────────────────────────
    if "_parse_error" in conv:
        return "reject", [f"JSON parse error on line {conv.get('_raw_line')}: {conv['_parse_error']}"]

    # ── Required top-level fields ─────────────────────────────────────────────
    required_fields = ["conversation_id", "language", "turns", "metadata"]
    missing = [f for f in required_fields if f not in conv]
    if missing:
        reasons.append(f"missing required fields: {missing}")

    if reasons:
        return "reject", reasons

    conv_id  = conv["conversation_id"]
    language = conv.get("language", "")
    turns    = conv.get("turns", [])
    metadata = conv.get("metadata", {})

    # ── Duplicate conversation ID ─────────────────────────────────────────────
    if conv_id in seen_ids:
        reasons.append(f"duplicate conversation_id '{conv_id}'")
    else:
        seen_ids.add(conv_id)

    # ── Language label ────────────────────────────────────────────────────────
    if language not in VALID_LANGUAGES:
        reasons.append(f"language '{language}' is not valid")

    # ── Turn cleaning (recoverable issues logged but not reject-worthy alone) ──
    if not isinstance(turns, list):
        reasons.append("turns field is not a list")
        return "reject", reasons

    cleaned_turns, turn_issues = clean_turns(turns)
    conv["turns"] = cleaned_turns   # update in place

    # ── Minimum turn count (after cleaning) ───────────────────────────────────
    if len(cleaned_turns) < MIN_TURNS_REQUIRED:
        reasons.append(
            f"fewer than {MIN_TURNS_REQUIRED} valid turns after cleaning "
            f"(got {len(cleaned_turns)})"
        )

    # ── Garbled text ──────────────────────────────────────────────────────────
    garbled_turns = [t for t in cleaned_turns if is_garbled(t.get("text", ""))]
    if garbled_turns:
        reasons.append(f"{len(garbled_turns)} turn(s) contain garbled/encoding-corrupted text")

    # ── Language mismatch ─────────────────────────────────────────────────────
    if cleaned_turns and language in VALID_LANGUAGES:
        detected = detect_language(cleaned_turns)
        if detected != "unknown":
            mismatch = False
            if language == "hindi"    and detected == "english":
                mismatch = True
            elif language == "english" and detected == "hindi":
                mismatch = True
            elif language == "hinglish" and detected == "english":
                mismatch = True
            if mismatch:
                reasons.append(
                    f"language label mismatch: labelled '{language}' "
                    f"but detected '{detected}'"
                )

    # ── Metadata validation ───────────────────────────────────────────────────
    meta_issues = validate_metadata(metadata)
    reasons.extend(meta_issues)

    if reasons:
        return "reject", reasons
    return "accept", []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading {RAW_PATH} …")
    raw_records = load_jsonl(RAW_PATH)
    print(f"  Loaded {len(raw_records)} records.")

    cleaned  = []
    rejected = []
    seen_ids: set[str] = set()

    rejection_counter = Counter()

    for conv in raw_records:
        verdict, reasons = process_conversation(conv, seen_ids)
        if verdict == "accept":
            cleaned.append(conv)
        else:
            conv_copy = dict(conv)
            conv_copy["rejection_reason"] = "; ".join(reasons)
            rejected.append(conv_copy)
            for r in reasons:
                # Bucket the reason by its first meaningful phrase
                bucket = r.split("(")[0].split(":")[0].strip()
                rejection_counter[bucket] += 1

    write_jsonl(cleaned,  CLEAN_PATH)
    write_jsonl(rejected, REJECTED_PATH)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Raw conversations  : {len(raw_records)}")
    print(f"  Cleaned (accepted) : {len(cleaned)}")
    print(f"  Rejected           : {len(rejected)}")
    print(f"\n  Rejection reasons (may overlap):")
    for reason, count in rejection_counter.most_common():
        print(f"    {count:3d}  {reason}")
    print(f"\nOutputs written:")
    print(f"  {CLEAN_PATH}")
    print(f"  {REJECTED_PATH}")


if __name__ == "__main__":
    main()
