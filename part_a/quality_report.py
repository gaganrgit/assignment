"""
quality_report.py
-----------------
Reads raw_conversations.jsonl and cleaned_conversations.jsonl (+ rejected)
and prints a comprehensive quality / statistics report for the ML team.
"""

import json
from pathlib import Path
from collections import Counter

RAW_PATH      = Path("raw_conversations.jsonl")
CLEAN_PATH    = Path("cleaned_conversations.jsonl")
REJECTED_PATH = Path("rejected_conversations.jsonl")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def avg_turns(records: list[dict]) -> float:
    if not records:
        return 0.0
    return sum(len(r.get("turns", [])) for r in records) / len(records)


def language_dist(records: list[dict]) -> Counter:
    return Counter(r.get("language", "unknown") for r in records)


def outcome_dist(records: list[dict]) -> Counter:
    return Counter(r.get("metadata", {}).get("outcome", "unknown") for r in records)


def avg_duration(records: list[dict]) -> float:
    durations = []
    for r in records:
        d = r.get("metadata", {}).get("call_duration_seconds")
        if isinstance(d, (int, float)) and d > 0:
            durations.append(d)
    return sum(durations) / len(durations) if durations else 0.0


def pct(part: int, total: int) -> str:
    return f"{part / total * 100:.1f}%" if total else "N/A"


def print_distribution(title: str, counter: Counter, total: int) -> None:
    print(f"\n  {title}:")
    for key, count in sorted(counter.items()):
        print(f"    {key:<30s} {count:4d}  ({pct(count, total)})")


def separator(char="─", width=62) -> None:
    print(char * width)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    raw      = load_jsonl(RAW_PATH)
    cleaned  = load_jsonl(CLEAN_PATH)
    rejected = load_jsonl(REJECTED_PATH)

    n_raw     = len(raw)
    n_cleaned = len(cleaned)
    n_rejected = len(rejected)

    separator("=")
    print("  KAPTURE CX — DATA QUALITY REPORT")
    separator("=")

    # ── 1. Volume ─────────────────────────────────────────────────────────────
    print("\n1. VOLUME SUMMARY")
    separator()
    print(f"  Raw conversations    : {n_raw}")
    print(f"  Cleaned (kept)       : {n_cleaned}  ({pct(n_cleaned, n_raw)} of raw)")
    print(f"  Rejected             : {n_rejected}  ({pct(n_rejected, n_raw)} of raw)")

    # ── 2. Rejection breakdown ────────────────────────────────────────────────
    print("\n2. REJECTION BREAKDOWN")
    separator()
    if rejected:
        reason_counter: Counter = Counter()
        for r in rejected:
            raw_reasons = r.get("rejection_reason", "")
            for part in raw_reasons.split(";"):
                bucket = part.strip().split("(")[0].split(":")[0].strip()
                if bucket:
                    reason_counter[bucket] += 1
        for reason, count in reason_counter.most_common():
            print(f"  {count:3d} ({pct(count, n_rejected)})  {reason}")
    else:
        print("  No rejections.")

    # ── 3. Language distribution ──────────────────────────────────────────────
    print("\n3. LANGUAGE DISTRIBUTION")
    separator()
    raw_lang   = language_dist(raw)
    clean_lang = language_dist(cleaned)
    print(f"  {'Language':<15} {'Before':>8} {'After':>8}  {'Change':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8}  {'-'*8}")
    all_langs = sorted(set(raw_lang) | set(clean_lang))
    for lang in all_langs:
        b = raw_lang.get(lang, 0)
        a = clean_lang.get(lang, 0)
        print(f"  {lang:<15} {b:>8} {a:>8}  {a - b:>+8}")

    # ── 4. Outcome distribution ───────────────────────────────────────────────
    print("\n4. OUTCOME DISTRIBUTION")
    separator()
    raw_out   = outcome_dist(raw)
    clean_out = outcome_dist(cleaned)
    all_outcomes = sorted((set(raw_out) | set(clean_out)), key=lambda x: str(x))
    print(f"  {'Outcome':<30} {'Before':>8} {'After':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    for out in all_outcomes:
        b = raw_out.get(out, 0)
        a = clean_out.get(out, 0)
        print(f"  {str(out):<30} {b:>8} {a:>8}")

    # ── 5. Turn statistics ────────────────────────────────────────────────────
    print("\n5. TURN STATISTICS")
    separator()
    print(f"  Avg turns/conv (raw)     : {avg_turns(raw):.2f}")
    print(f"  Avg turns/conv (cleaned) : {avg_turns(cleaned):.2f}")

    raw_turn_counts = Counter(len(r.get("turns", [])) for r in raw)
    print("\n  Turn-count distribution (raw):")
    for k in sorted(raw_turn_counts):
        print(f"    {k} turns : {raw_turn_counts[k]} conversation(s)")

    # ── 6. Duration statistics ────────────────────────────────────────────────
    print("\n6. CALL DURATION STATISTICS")
    separator()
    print(f"  Avg duration (raw)     : {avg_duration(raw):.1f} s")
    print(f"  Avg duration (cleaned) : {avg_duration(cleaned):.1f} s")

    # Count metadata anomalies in raw
    null_duration = sum(
        1 for r in raw
        if r.get("metadata", {}).get("call_duration_seconds") in (None, 0)
        or (isinstance(r.get("metadata", {}).get("call_duration_seconds"), (int, float))
            and r["metadata"]["call_duration_seconds"] < 0)
    )
    print(f"  Conversations with invalid duration (raw) : {null_duration}")

    # ── 7. Specific quality-issue counts (raw) ────────────────────────────────
    print("\n7. SPECIFIC QUALITY ISSUE COUNTS (RAW DATASET)")
    separator()

    def count_empty_turns(records):
        return sum(
            1 for r in records
            if any(not t.get("text", "").strip() for t in r.get("turns", []))
        )

    def count_dup_consecutive(records):
        count = 0
        for r in records:
            turns = r.get("turns", [])
            for i in range(1, len(turns)):
                if turns[i]["role"] == turns[i-1]["role"] and turns[i]["text"] == turns[i-1]["text"]:
                    count += 1
                    break
        return count

    def count_short(records, min_turns=2):
        return sum(1 for r in records if len(r.get("turns", [])) < min_turns)

    def count_missing_meta(records):
        return sum(
            1 for r in records
            if not isinstance(r.get("metadata"), dict)
            or r["metadata"].get("outcome") not in {"payment_committed", "callback_scheduled",
                                                     "escalated", "no_resolution"}
            or not isinstance(r["metadata"].get("call_duration_seconds"), (int, float))
            or r["metadata"]["call_duration_seconds"] <= 0
        )

    print(f"  Convs with empty/whitespace turns    : {count_empty_turns(raw)}")
    print(f"  Convs with dup consecutive turns     : {count_dup_consecutive(raw)}")
    print(f"  Convs with fewer than 2 turns        : {count_short(raw)}")
    print(f"  Convs with invalid metadata          : {count_missing_meta(raw)}")

    # ── 8. Training-readiness summary ─────────────────────────────────────────
    print("\n8. TRAINING-READINESS SUMMARY")
    separator()
    print(f"  Conversations ready for training : {n_cleaned}")
    print(f"  Estimated % of raw data usable   : {pct(n_cleaned, n_raw)}")
    lang_balance = clean_lang
    dominant = max(lang_balance, key=lang_balance.get)
    print(f"  Most common language in clean    : {dominant} ({lang_balance[dominant]} convs)")
    print(f"  Unique outcomes in clean data    : {len(set(clean_out.keys()) - {'unknown', 'None'})}")

    separator("=")
    print("  END OF REPORT")
    separator("=")


if __name__ == "__main__":
    main()
