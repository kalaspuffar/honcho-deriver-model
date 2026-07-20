#!/usr/bin/env python3
"""
generate_conversations_batch.py
-------------------------------
Batch-API version of generate_conversations.py. Same output format and CLI,
but submits all generations as ONE Message Batch (50% cheaper, async). Most
batches finish in under an hour; hard ceiling is 24h.

How it differs from the sync version:
  - All N generation requests are built up front, each with a unique custom_id.
  - One batch is submitted; we poll until processing_status == "ended".
  - Results come back UNORDERED, matched to their scenario via custom_id.
  - The (identical) system prompt is marked for prompt caching for extra savings.

Output schema (one JSON object per line) is identical to the sync script:
{"id", "scenario": {...}, "peers": [...], "turns": [{"speaker","content"}, ...]}

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python generate_conversations_batch.py --count 100 --out conversations.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time

from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
BATCH_MAX_REQUESTS = 100_000  # Anthropic per-batch ceiling

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
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in reply")
    return json.loads(text[start:end + 1])


def valid_conversation(obj):
    if not isinstance(obj, dict):
        return False
    peers, turns = obj.get("peers"), obj.get("turns")
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
    ap = argparse.ArgumentParser(description="Generate conversations via the Message Batches API.")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--out", default="conversations.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--turns-min", type=int, default=8)
    ap.add_argument("--turns-max", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll-interval", type=int, default=30, help="seconds between status checks")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")
    if args.count > BATCH_MAX_REQUESTS:
        sys.exit(f"--count exceeds the {BATCH_MAX_REQUESTS} per-batch limit; split into chunks.")

    random.seed(args.seed)
    client = Anthropic()

    # Build all requests up front; remember each scenario by custom_id.
    requests, scenario_map = [], {}
    for i in range(args.count):
        cid = f"conv_{i:05d}"
        scenario = {
            "setting": random.choice(SETTINGS),
            "relationship": random.choice(RELATIONSHIPS),
            "domain": random.choice(DOMAINS),
            "register": random.choice(REGISTERS),
        }
        scenario_map[cid] = scenario
        n_turns = random.randint(args.turns_min, args.turns_max)
        requests.append({
            "custom_id": cid,
            "params": {
                "model": args.model,
                "max_tokens": 2000,
                #"temperature": 1.0,
                # System prompt is identical across all requests -> cache it.
                "system": [{"type": "text", "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": build_user_prompt(scenario, n_turns)}],
            },
        })

    print(f"Submitting batch of {len(requests)} generation requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch {batch.id} created. Polling every {args.poll_interval}s "
          f"(most finish < 1h, max 24h)...")
    wait_for_batch(client, batch.id, args.poll_interval)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for entry in client.messages.batches.results(batch.id):
            if entry.result.type != "succeeded":
                print(f"  [skip] {entry.custom_id}: {entry.result.type}", file=sys.stderr)
                continue
            try:
                obj = extract_json(entry.result.message.content[0].text)
                if not valid_conversation(obj):
                    raise ValueError("failed shape validation")
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {entry.custom_id}: {e}", file=sys.stderr)
                continue
            record = {
                "id": entry.custom_id,
                "scenario": scenario_map.get(entry.custom_id, {}),
                "peers": obj["peers"],
                "turns": obj["turns"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nWrote {written} conversations to {args.out}")
    if written:
        with open(args.out, encoding="utf-8") as f:
            sample = json.loads(f.readline())
        print("\n--- sample conversation ---")
        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
