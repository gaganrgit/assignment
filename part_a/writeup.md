# Part A — Writeup

## Assumptions Made During Cleaning

1. **Minimum 2 turns = training-ready.** A conversation with fewer than 2 turns has no meaningful input-output pair for an LLM to learn from. I set this as a hard rejection threshold rather than a warning.

2. **Turn cleaning before count check.** Empty/whitespace turns and duplicate consecutive turns are cleaned first; only *then* is the turn count checked. This avoids falsely rejecting conversations that would be valid after minor noise removal.

3. **Language detection is heuristic, not ML-based.** I used a keyword vocabulary approach: Hindi/Hinglish conversations contain words like "haan", "kya", "aap", "theek", while purely English ones contain words like "your", "the", "have", "been". A mismatch is flagged only when the detected language clearly contradicts the label (e.g., labelled "hindi" but every turn is fluent English). Borderline cases are given the benefit of the doubt.

4. **Garbled text = structural corruption.** A turn is considered garbled if more than 10% of its characters are non-printable control characters, Unicode surrogates, or replacement characters (`\ufffd`). This threshold was chosen to tolerate isolated special characters while catching clearly corrupted text.

5. **Zero-duration calls are invalid.** A `call_duration_seconds` of 0 is treated the same as a negative value — a real call cannot have zero duration. This is stricter than the spec, but a zero-duration call is almost certainly a data pipeline error.

6. **Missing metadata dict entirely = hard reject.** If the `metadata` field is absent or not a dict, the whole conversation is rejected rather than partially recovered.

---

## Hardest Issue to Detect Programmatically

**Language label mismatch** was the hardest.

The other issues have definitive signals: empty text, duplicate strings, negative numbers. Language mismatch is inherently ambiguous — Hinglish by definition mixes Hindi and English, so a "hinglish" conversation might score high on English keywords. Distinguishing *correctly labelled Hinglish* from *mislabelled English* requires context.

My approach:
- Count hits against a curated Hindi-keyword set and an English-only-word set.
- If a conversation labelled `hindi` has zero Hindi keywords and many English words, flag it.
- Avoid flagging `hinglish` unless the text is *entirely* English with no Hindi lexical items at all.

This heuristic has false-negative risk (it won't catch subtle mismatches), but it avoids false positives that would incorrectly discard valid Hinglish data.

---

## Scaling to 100,000 Conversations

At 100 conversations the pipeline runs in under a second in a single Python process. At 100,000 the following changes would matter:

1. **Parallel processing.** Use `multiprocessing.Pool` or a framework like Apache Beam / Spark to process conversations in parallel across CPU cores or a cluster.

2. **Replace keyword-heuristic language detection with a proper model.** `langdetect` or `fastText`'s language identification model handles 170+ languages accurately, is fast (microseconds per string), and handles Hinglish better than hand-crafted keyword lists.

3. **Streaming I/O.** Reading 100,000 JSON lines into memory is fine (~100 MB), but writing cleaned and rejected files should be streamed (one record at a time) to avoid buffering the entire output.

4. **Deduplication at scale.** The current duplicate-ID check uses an in-memory `set`. At large scale, a hash set still works up to ~10 M records; beyond that, a probabilistic structure (Bloom filter) or a database index is more memory-efficient.

5. **Observability.** Replace `print()` with structured logging (Python `logging` module, or a service like Weights & Biases artifact tracking) so quality metrics are persisted and comparable across data pipeline runs.

6. **Schema versioning.** Use `pydantic` or `jsonschema` to define the conversation schema formally, so that schema evolution is explicit and breaking changes are caught automatically.
