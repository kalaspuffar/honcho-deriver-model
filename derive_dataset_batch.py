#!/usr/bin/env python3
"""
derive_dataset_batch.py
-----------------------
Batch-API version of derive_dataset.py. Same validation rules, same chat-format
output split into train/eval — but every per-turn derivation goes into ONE
Message Batch at 50% off, processed asynchronously.

How it differs from the sync version:
  - One request per (conversation, turn) is built up front with a unique custom_id.
  - A single batch is submitted; we poll until it ends, then stream results.
  - Results are matched back to their turn via custom_id (order isn't guaranteed).
  - The (identical) deriver system prompt is marked for prompt caching.

Schema match note (unchanged from the sync script): targets the confirmed
"minimal deriver" shape {"explicit": [{content, salience}]}. If your installed
Honcho also emits "deductive", add it to ALLOWED_KEYS and to the prompt, and
paste your real minimal_deriver_prompt over DERIVER_SYSTEM_PROMPT.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python derive_dataset_batch.py --in conversations.jsonl \\
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

DEFAULT_MODEL = "claude-opus-4-8"   # switch to claude-sonnet-4-6 to cut cost further
BATCH_MAX_REQUESTS = 100_000
ALLOWED_KEYS = {"explicit"}         # add "deductive" if your deriver version emits it
MAX_CONTENT_CHARS = 150
MAX_CONTEXT_TURNS = 12

SPECULATION_WORDS = re.compile(
    r"\b(likely|probably|might|maybe|perhaps|possibly|seems?|appears?|presumably|"
    r"could be|may be|i think|suggests?)\b", re.IGNORECASE
)
LEADING_PRONOUN = re.compile(r"^\s*(she|he|they|it|her|his|their)\b", re.IGNORECASE)

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
    return (f"{ctx_block}Extract explicit conclusions about {peer} from this message:\n"
            f"{peer}: {current_turn}")


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def clean_and_validate(obj):
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
    return out


def wait_for_batch(client, batch_id, poll_interval):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        c = batch.request_counts
        print(f"  ...processing (succeeded={c.succeeded} errored={c.errored} "
              f"processing={c.processing})", file=sys.stderr)
        time.sleep(poll_interval)


def main():
    ap = argparse.ArgumentParser(description="Derive training pairs via the Message Batches API.")
    ap.add_argument("--in", dest="infile", default="conversations.jsonl")
    ap.add_argument("--train-out", default="train.jsonl")
    ap.add_argument("--eval-out", default="eval.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--keep-empty-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll-interval", type=int, default=30)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")
    if not os.path.exists(args.infile):
        sys.exit(f"Input not found: {args.infile} (run the generator first)")

    random.seed(args.seed)
    client = Anthropic()

    with open(args.infile, encoding="utf-8") as f:
        conversations = [json.loads(line) for line in f if line.strip()]

    # Build one request per turn; remember the user message by custom_id so we
    # can reconstruct the training row from the (unordered) results.
    requests, user_map = [], {}
    for ci, conv in enumerate(conversations):
        turns = conv["turns"]
        for idx, turn in enumerate(turns):
            peer = turn["speaker"]
            context = turns[max(0, idx - MAX_CONTEXT_TURNS):idx]
            user_msg = build_user_message(peer, context, turn["content"])
            cid = f"c{ci}t{idx}"
            user_map[cid] = user_msg
            requests.append({
                "custom_id": cid,
                "params": {
                    "model": args.model,
                    "max_tokens": 1000,
                    #"temperature": 0.2,
                    "system": [{"type": "text", "text": DERIVER_SYSTEM_PROMPT,
                                "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user", "content": user_msg}],
                },
            })

    if len(requests) > BATCH_MAX_REQUESTS:
        sys.exit(f"{len(requests)} requests exceeds the {BATCH_MAX_REQUESTS} per-batch limit; "
                 f"split your conversations file into chunks and run per chunk.")

    print(f"Submitting batch of {len(requests)} derivation requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch {batch.id} created. Polling every {args.poll_interval}s...")
    wait_for_batch(client, batch.id, args.poll_interval)

    rows = []
    for entry in client.messages.batches.results(batch.id):
        if entry.result.type != "succeeded":
            print(f"  [skip] {entry.custom_id}: {entry.result.type}", file=sys.stderr)
            continue
        try:
            parsed = extract_json(entry.result.message.content[0].text)
            conclusions = clean_and_validate(parsed)
            if conclusions is None:
                raise ValueError("failed validation")
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {entry.custom_id}: {e}", file=sys.stderr)
            continue

        is_empty = all(len(v) == 0 for v in conclusions.values())
        if is_empty and random.random() > args.keep_empty_frac:
            continue

        rows.append({
            "messages": [
                {"role": "system", "content": DERIVER_SYSTEM_PROMPT},
                {"role": "user", "content": user_map[entry.custom_id]},
                {"role": "assistant", "content": json.dumps(conclusions, ensure_ascii=False)},
            ]
        })

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
