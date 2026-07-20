#!/usr/bin/env python3
"""
generate_conversations.py
-------------------------
Stage 1 of the deriver-fine-tune data pipeline.

Generates diverse, two-person synthetic conversations and writes them to a
JSONL file (one conversation per line). These are the RAW SOURCE conversations;
a second script (derive_dataset.py) turns each turn into Honcho-style
{"explicit": [{content, salience}]} training targets.

Why synthetic: control over diversity, volume, no privacy/leakage concerns.
The generator deliberately samples across settings/relationships/domains/registers
so the eventual adapter doesn't overfit one "LLM voice", and it asks for some
low-content turns so the deriver later learns NOT to hallucinate facts from
greetings.

Output schema (one JSON object per line):
{
  "id": "conv_0001",
  "scenario": {"setting": "...", "relationship": "...", "domain": "...", "register": "..."},
  "peers": ["Maya", "Tom"],
  "turns": [
    {"speaker": "Tom", "content": "How was your weekend?"},
    {"speaker": "Maya", "content": "Pretty good! ..."}
  ]
}

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python generate_conversations.py --count 50 --out conversations.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time

from anthropic import Anthropic

# Sonnet is a strong, cost-reasonable teacher for bulk generation.
# Swap to "claude-opus-4-8" if you want maximum quality per conversation.
DEFAULT_MODEL = "claude-sonnet-4-6"

# Diversity axes — sampled per conversation so the corpus isn't monotonous.
SETTINGS = [
    "coffee break at the office", "text messages over a weekend", "catching up after months apart",
    "waiting for a flight", "a video call between two cities", "chatting while cooking dinner",
    "a gym session", "walking the dog", "a long car ride", "a hobby club meetup",
    "DMs on a hobby forum", "a hospital waiting room", "moving apartments",
]
RELATIONSHIPS = [
    "two old friends", "new coworkers", "siblings", "a couple", "former classmates reconnecting",
    "neighbors", "a mentor and mentee", "members of the same band", "a parent and adult child",
    "gym buddies", "online friends who've never met in person",
]
DOMAINS = [
    "family and parenting", "fitness and health", "career and work", "travel", "food and cooking",
    "music and hobbies", "moving and housing", "pets", "education and study", "sports",
    "money and saving", "relationships", "tech and gadgets",
]
REGISTERS = [
    "warm and casual", "terse and texty (short messages, abbreviations)", "excited and rambling",
    "dry and understated", "anxious and oversharing", "playful and teasing",
]

SYSTEM_PROMPT = """You write realistic two-person conversations for a memory-extraction \
dataset. The conversations must contain plenty of EXTRACTABLE personal facts about the \
participants (jobs, family, locations, hobbies, plans, preferences, health, history) woven \
naturally into normal dialogue — never a fact-dump or interview.

Rules:
- Exactly two named speakers.
- Vary turn length: some turns are one short line (including pure greetings or filler with NO \
facts at all), others are 2-4 sentences dense with life details.
- Facts should sometimes require context to interpret (pronouns, "she", "that place") so a \
later extraction step has to resolve references from earlier turns.
- Keep it natural for the given setting, relationship, domain, and register.
- Do NOT narrate or add stage directions. Only spoken/typed turns.

Return ONLY a JSON object, no prose, no code fences, with this exact shape:
{
  "peers": ["Name1", "Name2"],
  "turns": [
    {"speaker": "Name1", "content": "..."},
    {"speaker": "Name2", "content": "..."}
  ]
}"""


def build_user_prompt(scenario, n_turns):
    return (
        f"Write a conversation of about {n_turns} turns.\n"
        f"Setting: {scenario['setting']}\n"
        f"Relationship: {scenario['relationship']}\n"
        f"Primary domain of life details: {scenario['domain']}\n"
        f"Register/style: {scenario['register']}\n"
        f"Invent two fitting first names. Remember: most life facts should attach to the "
        f"participants, and include at least one or two near-empty turns with no facts."
    )


def extract_json(text):
    """Pull a JSON object out of a model reply, tolerating stray code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in reply")
    return json.loads(text[start:end + 1])


def valid_conversation(obj):
    if not isinstance(obj, dict):
        return False
    peers = obj.get("peers")
    turns = obj.get("turns")
    if not (isinstance(peers, list) and len(peers) == 2 and all(isinstance(p, str) for p in peers)):
        return False
    if not (isinstance(turns, list) and len(turns) >= 4):
        return False
    for t in turns:
        if not (isinstance(t, dict) and isinstance(t.get("speaker"), str)
                and isinstance(t.get("content"), str) and t["content"].strip()):
            return False
        if t["speaker"] not in peers:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic two-person conversations.")
    ap.add_argument("--count", type=int, default=50, help="number of conversations to generate")
    ap.add_argument("--out", default="conversations.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--turns-min", type=int, default=8)
    ap.add_argument("--turns-max", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    random.seed(args.seed)
    client = Anthropic()
    written = 0

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.count):
            scenario = {
                "setting": random.choice(SETTINGS),
                "relationship": random.choice(RELATIONSHIPS),
                "domain": random.choice(DOMAINS),
                "register": random.choice(REGISTERS),
            }
            n_turns = random.randint(args.turns_min, args.turns_max)

            for attempt in range(3):
                try:
                    resp = client.messages.create(
                        model=args.model,
                        max_tokens=2000,
                        temperature=1.0,  # high temp = more varied conversations
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": build_user_prompt(scenario, n_turns)}],
                    )
                    obj = extract_json(resp.content[0].text)
                    if not valid_conversation(obj):
                        raise ValueError("conversation failed shape validation")
                    break
                except Exception as e:  # noqa: BLE001 - retry on any failure
                    if attempt == 2:
                        print(f"  [skip] conv {i}: {e}", file=sys.stderr)
                        obj = None
                    else:
                        time.sleep(2 * (attempt + 1))
            if obj is None:
                continue

            record = {
                "id": f"conv_{written:04d}",
                "scenario": scenario,
                "peers": obj["peers"],
                "turns": obj["turns"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"[{written}/{args.count}] {record['peers']} — {scenario['domain']} "
                  f"({len(record['turns'])} turns)")

    print(f"\nWrote {written} conversations to {args.out}")
    # Show one sample so you can eyeball the format.
    if written:
        with open(args.out, encoding="utf-8") as f:
            sample = json.loads(f.readline())
        print("\n--- sample conversation ---")
        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
