#!/usr/bin/env python3
"""
derive_dataset.py
-----------------
Stage 2 of the deriver-fine-tune data pipeline.

Reads conversations.jsonl (from generate_conversations.py) and, for each turn,
asks a teacher model (Claude) to extract Honcho-style EXPLICIT conclusions about
the speaking peer. Each (turn-in-context -> conclusions JSON) pair becomes one
training row, written in chat-message format ready for an Unsloth instruct QLoRA
notebook.

Target output schema per turn (mirrors Honcho's "minimal deriver"):
    {"explicit": [{"content": "<peer> ...", "salience": <1-10 int>}, ...]}

IMPORTANT — match your runtime:
  This builds the derivation prompt from the confirmed minimal-deriver schema
  (single "explicit" array of {content, salience}). If YOUR installed Honcho's
  src/deriver/prompts.py (minimal_deriver_prompt) also emits a "deductive" array,
  add it: extend DERIVER_SYSTEM_PROMPT's definitions + schema and add "deductive"
  to ALLOWED_KEYS below. The adapter must learn to answer the EXACT prompt your
  deriver sends, so paste your real prompt over DERIVER_SYSTEM_PROMPT if it differs.

Training-row format (chat / ShareGPT-style, what Unsloth instruct notebooks expect):
    {"messages": [
        {"role": "system", "content": "<deriver prompt>"},
        {"role": "user", "content": "<context + current turn>"},
        {"role": "assistant", "content": "<conclusions JSON>"}
    ]}

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python derive_dataset.py --in conversations.jsonl \\
        --train-out train.jsonl --eval-out eval.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time

from anthropic import Anthropic

# A strong teacher matters more here than in generation — quality of the targets
# is the ceiling on your fine-tune. Opus is defensible for the derivation pass.
DEFAULT_MODEL = "claude-opus-4-8"

ALLOWED_KEYS = {"explicit"}  # add "deductive" if your deriver version emits it
MAX_CONTENT_CHARS = 150       # Plastic curated short; long conclusions hurt embeddings
MAX_CONTEXT_TURNS = 12        # prior turns shown for reference resolution

# Words that signal speculation — forbidden in EXPLICIT conclusions (must be certain).
SPECULATION_WORDS = re.compile(
    r"\b(likely|probably|might|maybe|perhaps|possibly|seems?|appears?|presumably|"
    r"could be|may be|i think|suggests?)\b", re.IGNORECASE
)
# Conclusions starting with a bare pronoun are not self-contained.
LEADING_PRONOUN = re.compile(r"^\s*(she|he|they|it|her|his|their)\b", re.IGNORECASE)

# This mirrors the intent of Honcho's minimal_deriver_prompt. Replace with your
# installed version's actual prompt for exact runtime parity.
DERIVER_SYSTEM_PROMPT = """You extract EXPLICIT factual conclusions about a specific person \
from a conversation turn. Explicit conclusions are facts the person directly states or that \
are unambiguously present in their message — never guesses, inferences, or speculation.

Rules for each conclusion:
- ATOMIC: exactly one fact. Never combine ("has a dog and a dog bed" -> two conclusions).
- SELF-CONTAINED: start with the person's name; resolve all pronouns and references using \
the conversation context. Never write fragments like "responding to the comment about X".
- SHORT: well under 150 characters.
- CERTAIN: no speculation words (likely, probably, seems, might, etc.). If it isn't certain \
from what was said, omit it.
- A turn with no factual content (a greeting, filler) yields an empty list. Do not invent facts.

Assign each conclusion a "salience" integer 1-10 for how important/identifying it is:
  1-3 = trivial/transient, 4-6 = ordinary personal detail, 7-10 = core identity or stable fact.

Return ONLY a JSON object, no prose, no code fences:
{"explicit": [{"content": "<Name> ...", "salience": <int 1-10>}, ...]}"""


def build_user_message(peer, context_turns, current_turn):
    ctx = "\n".join(f"{t['speaker']}: {t['content']}" for t in context_turns)
    ctx_block = f"Conversation so far:\n{ctx}\n\n" if ctx else ""
    return (
        f"{ctx_block}"
        f"Extract explicit conclusions about {peer} from this message:\n"
        f"{peer}: {current_turn}"
    )


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def clean_and_validate(obj, peer):
    """Return a sanitized conclusions dict, or None if it's unusable."""
    if not isinstance(obj, dict) or set(obj.keys()) - ALLOWED_KEYS:
        return None
    out = {}
    for key in ALLOWED_KEYS:
        items = obj.get(key, [])
        if not isinstance(items, list):
            return None
        kept = []
        for it in items:
            if not isinstance(it, dict):
                continue
            content = (it.get("content") or "").strip()
            sal = it.get("salience")
            if not content or len(content) > MAX_CONTENT_CHARS:
                continue
            if not isinstance(sal, int) or not (1 <= sal <= 10):
                continue
            if LEADING_PRONOUN.match(content):
                continue
            if key == "explicit" and SPECULATION_WORDS.search(content):
                continue
            kept.append({"content": content, "salience": sal})
        out[key] = kept
    # An all-empty result is valid (low-content turn) but we keep only a fraction
    # of those so the set isn't dominated by empties; caller decides.
    return out


def main():
    ap = argparse.ArgumentParser(description="Derive Honcho-style training pairs from conversations.")
    ap.add_argument("--in", dest="infile", default="conversations.jsonl")
    ap.add_argument("--train-out", default="train.jsonl")
    ap.add_argument("--eval-out", default="eval.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--keep-empty-frac", type=float, default=0.15,
                    help="fraction of zero-conclusion turns to keep (teaches restraint)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")
    if not os.path.exists(args.infile):
        sys.exit(f"Input not found: {args.infile} (run generate_conversations.py first)")

    random.seed(args.seed)
    client = Anthropic()
    rows = []

    with open(args.infile, encoding="utf-8") as f:
        conversations = [json.loads(line) for line in f if line.strip()]

    for ci, conv in enumerate(conversations):
        turns = conv["turns"]
        for idx, turn in enumerate(turns):
            peer = turn["speaker"]
            context = turns[max(0, idx - MAX_CONTEXT_TURNS):idx]
            user_msg = build_user_message(peer, context, turn["content"])

            conclusions = None
            for attempt in range(3):
                try:
                    resp = client.messages.create(
                        model=args.model,
                        max_tokens=1000,
                        #temperature=0.2,  # low temp = consistent, format-faithful targets
                        system=DERIVER_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": user_msg}],
                    )
                    parsed = extract_json(resp.content[0].text)
                    conclusions = clean_and_validate(parsed, peer)
                    if conclusions is None:
                        raise ValueError("failed validation")
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        print(f"  [skip] conv {ci} turn {idx}: {e}", file=sys.stderr)
                    else:
                        time.sleep(2 * (attempt + 1))

            if conclusions is None:
                continue

            is_empty = all(len(v) == 0 for v in conclusions.values())
            if is_empty and random.random() > args.keep_empty_frac:
                continue  # downsample empties

            rows.append({
                "messages": [
                    {"role": "system", "content": DERIVER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant",
                     "content": json.dumps(conclusions, ensure_ascii=False)},
                ]
            })
        print(f"[conv {ci + 1}/{len(conversations)}] running total: {len(rows)} rows")

    random.shuffle(rows)
    n_eval = int(len(rows) * args.eval_frac)
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]

    for path, data in ((args.train_out, train_rows), (args.eval_out, eval_rows)):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(train_rows)} train rows -> {args.train_out}")
    print(f"Wrote {len(eval_rows)} eval rows -> {args.eval_out}")
    if train_rows:
        print("\n--- sample training row ---")
        print(json.dumps(train_rows[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
