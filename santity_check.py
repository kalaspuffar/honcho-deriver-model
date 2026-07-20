#!/usr/bin/env python3
"""
sanity_check.py
---------------
Quick correctness check for a fine-tuned deriver model served by Ollama.
Zero dependencies (stdlib only). Sends a handful of crafted turns through the
local model with your deriver system prompt and validates the output:

  - valid JSON with only the "explicit" key
  - each conclusion atomic, self-contained (no leading pronoun), short
  - salience an integer 1-10
  - empty/filler turns -> empty list (restraint)
  - multi-fact turns -> several atomic conclusions with VARIED salience
  - reference turns -> pronouns resolved to the named person

Usage:
    python sanity_check.py --model deriver-qwen3-8b
    # optional: keep your real prompt in a file and load it
    python sanity_check.py --model deriver-qwen3-8b --system deriver_system.txt
"""

import argparse
import json
import re
import sys
import urllib.request

# Keep this identical to the system prompt you TRAINED with. Override via --system.
DEFAULT_SYSTEM = """You extract EXPLICIT factual conclusions about a specific person \
from a conversation turn. Explicit conclusions are facts the person directly states or that \
are unambiguously present in their message — never guesses, inferences, or speculation.

Rules for each conclusion:
- ATOMIC: exactly one fact.
- SELF-CONTAINED: start with the person's name; resolve all pronouns and references.
- SHORT: well under 150 characters.
- CERTAIN: no speculation words. If it isn't certain, omit it.
- A turn with no factual content yields an empty list. Do not invent facts.

Assign each conclusion a "salience" integer 1-10.
Return ONLY a JSON object: {"explicit": [{"content": "<Name> ...", "salience": <int 1-10>}, ...]}"""

LEADING_PRONOUN = re.compile(r"^\s*(she|he|they|it|her|his|their)\b", re.IGNORECASE)

# Each case: a user message (context + extract instruction + turn) and expectations.
CASES = [
    {
        "name": "fact_dense",
        "user": ("Extract explicit conclusions about Maya from this message:\n"
                 "Maya: Exhausting move, but the new place in Portland finally has a yard, "
                 "so the dog's thrilled."),
        "expect_empty": False, "min_conclusions": 1,
    },
    {
        "name": "filler_empty",
        "user": ("Extract explicit conclusions about Tom from this message:\n"
                 "Tom: lol yeah totally, anyway gotta run"),
        "expect_empty": True,
    },
    {
        "name": "multi_fact",
        "user": ("Extract explicit conclusions about Priya from this message:\n"
                 "Priya: I turned 34 last week, just started as a nurse at St. Mary's, "
                 "my son Leo starts kindergarten in the fall, and we adopted a rescue cat."),
        "expect_empty": False, "min_conclusions": 4, "require_salience_variation": True,
    },
    {
        "name": "reference_resolution",
        "context": ["Dana: How's your sister doing these days?"],
        "user": ("Conversation so far:\nDana: How's your sister doing these days?\n\n"
                 "Extract explicit conclusions about Sam from this message:\n"
                 "Sam: She just passed the bar, so she's a lawyer now in Chicago."),
        "expect_empty": False, "min_conclusions": 1,
        # The facts are about the sister, not Sam — a good model extracts little/nothing
        # about SAM here. This case mostly checks it doesn't wrongly attribute to Sam.
        "must_not_contain": ["Sam is a lawyer", "Sam passed"],
    },
]


def call_model(url, model, system, user):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["message"]["content"]


def parse_explicit(text):
    """Return list of conclusions, or raise with a reason."""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    if stripped != text.strip():
        raise ValueError("output wrapped in code fences (chat-template/stop-token issue)")
    obj = json.loads(stripped)
    if set(obj.keys()) != {"explicit"}:
        raise ValueError(f"unexpected keys: {set(obj.keys())}")
    return obj["explicit"]


def check_case(case, raw):
    problems = []
    try:
        items = parse_explicit(raw)
    except Exception as e:  # noqa: BLE001
        return [f"invalid output: {e}"]

    if case.get("expect_empty") and items:
        problems.append(f"expected empty list, got {len(items)} conclusions (over-extraction)")
    if not case.get("expect_empty"):
        if len(items) < case.get("min_conclusions", 1):
            problems.append(f"expected >= {case.get('min_conclusions', 1)} conclusions, got {len(items)}")

    for it in items:
        c = (it.get("content") or "")
        s = it.get("salience")
        if not isinstance(s, int) or not (1 <= s <= 10):
            problems.append(f"bad salience {s!r} on: {c!r}")
        if len(c) > 150:
            problems.append(f"conclusion too long ({len(c)} chars): {c!r}")
        if LEADING_PRONOUN.match(c):
            problems.append(f"not self-contained (leading pronoun): {c!r}")
        if " and " in c.lower():
            problems.append(f"possible non-atomic (contains 'and'): {c!r}")  # soft

    if case.get("require_salience_variation") and len({it.get("salience") for it in items}) <= 1:
        problems.append("all conclusions share one salience value (no variation)")

    for bad in case.get("must_not_contain", []):
        if any(bad.lower() in (it.get("content") or "").lower() for it in items):
            problems.append(f"wrongly attributed: contains {bad!r}")

    return problems


def main():
    ap = argparse.ArgumentParser(description="Sanity-test a deriver model in Ollama.")
    ap.add_argument("--model", default="deriver-qwen3-8b")
    ap.add_argument("--url", default="http://localhost:11434/api/chat")
    ap.add_argument("--system", help="path to a file with the deriver system prompt")
    args = ap.parse_args()

    system = DEFAULT_SYSTEM
    if args.system:
        with open(args.system, encoding="utf-8") as f:
            system = f.read()

    passed = 0
    for case in CASES:
        try:
            raw = call_model(args.url, args.model, system, case["user"])
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {case['name']}: could not call model: {e}", file=sys.stderr)
            continue
        problems = check_case(case, raw)
        if problems:
            print(f"[FAIL] {case['name']}")
            for p in problems:
                print(f"        - {p}")
            print(f"        raw: {raw.strip()[:200]}")
        else:
            print(f"[PASS] {case['name']}")
            passed += 1

    print(f"\n{passed}/{len(CASES)} cases passed")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()