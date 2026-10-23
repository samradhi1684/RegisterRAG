#!/usr/bin/env python3
"""
Download the full 2WikiMultiHopQA train + validation sets, save them
unmodified as JSON, and build a fixed 100-question evaluation subset
from the validation set, stratified equally across the four official
question types (25 each), seed=42.

Official question types:
    comparison, inference, compositional, bridge_comparison

Outputs (in --out-dir, default ./wiki2_data):
    2wiki_train.json     full train set, all original fields
    2wiki_dev.json       full validation set, all original fields
    2wiki_100.json       100 type-stratified questions, all original fields
    2wiki_100.md         human-readable view of the 100 questions

Original fields per example (HuggingFace / official schema):
    _id, type, question, context [[title, [sentences]]],
    supporting_facts [[title, sent_id]], answer,
    evidences [[fact, relation, entity]]

Sources:
    default             HuggingFace "xanhho/2WikiMultihopQA" ("full_support" config)
    --official-dir      folder holding the official JSON files from the
                        2WikiMultiHopQA repo: train.json, dev.json

Usage:
    pip install datasets
    python prepare_2wiki.py
    python prepare_2wiki.py --official-dir ./2wiki_official
"""

import argparse
import json
import random
import sys
from pathlib import Path

SEED = 42
TYPES = ("comparison", "inference", "compositional", "bridge_comparison")
N_PER_TYPE = 25  # 4 × 25 = 100


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_hf(dataset_name: str, split: str) -> list:
    from datasets import load_dataset

    hf_split = "validation" if split == "validation" else split

    print(f"[hf] loading {dataset_name} / split={hf_split}")

    ds = load_dataset(dataset_name, split=hf_split)

    rows = []
    for row in ds:
        row = dict(row)

        # Convert HF columnar structures to normal JSON-friendly lists
        if isinstance(row.get("context"), dict):
            row["context"] = list(
                zip(
                    row["context"]["title"],
                    row["context"]["sentences"]
                )
            )

        if isinstance(row.get("supporting_facts"), dict):
            row["supporting_facts"] = list(
                zip(
                    row["supporting_facts"]["title"],
                    row["supporting_facts"]["sent_id"]
                )
            )

        rows.append(row)

    return rows

def load_official(folder: str, split: str) -> list:
    fname = {"train": "train.json", "validation": "dev.json"}[split]
    path = Path(folder) / fname
    print(f"[official] reading {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else list(data.values())


def get_split(split: str, args) -> list:
    if args.official_dir:
        return load_official(args.official_dir, split)
    return load_hf(args.hf_dataset, split)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def qid(ex: dict) -> str:
    return ex.get("_id", ex.get("id", ""))


def supporting_pairs(ex: dict) -> list:
    sf = ex["supporting_facts"]
    if isinstance(sf, dict):
        return list(zip(sf["title"], sf["sent_id"]))
    return [(t, i) for t, i in sf]


def context_paragraphs(ex: dict) -> list:
    ctx = ex["context"]
    if isinstance(ctx, dict):
        return list(zip(ctx["title"], ctx["sentences"]))
    return [(t, s) for t, s in ctx]


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_eval_set(val: list) -> list:
    rng = random.Random(SEED)

    # audit type distribution first
    dist = {}
    for ex in val:
        dist[ex["type"]] = dist.get(ex["type"], 0) + 1
    print(f"validation type distribution: {dist}")

    missing = [t for t in TYPES if t not in dist]
    if missing:
        sys.exit(f"Missing types in validation set: {missing}")

    by_type = {t: [] for t in TYPES}
    for ex in val:
        if ex["type"] in by_type:
            by_type[ex["type"]].append(ex)

    picked = []
    for t in TYPES:
        pool = sorted(by_type[t], key=qid)
        if len(pool) < N_PER_TYPE:
            sys.exit(f"Not enough '{t}' questions: {len(pool)} < {N_PER_TYPE}")
        picked.extend(rng.sample(pool, N_PER_TYPE))

    rng.shuffle(picked)  # mix types, still deterministic under seed=42
    return picked


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def to_markdown(items: list) -> str:
    counts = {t: sum(1 for x in items if x["type"] == t) for t in TYPES}
    dist = ", ".join(f"{n} x {t}" for t, n in counts.items())

    out = [
        "# 2WikiMultiHopQA - 100-question evaluation set",
        "",
        "- Source split: validation",
        f"- Sampling: seed={SEED}, stratified by type ({dist})",
        f"- Total questions: {len(items)}",
        "- In the context, sentences tagged **[SF]** are supporting facts.",
        "",
        "---",
        "",
    ]

    for n, ex in enumerate(items, 1):
        sf = supporting_pairs(ex)
        sf_set = {(t, int(i)) for t, i in sf}
        evidences = ex.get("evidences") or []

        out.append(f"## {n}. {ex['question']}")
        out.append("")
        out.append(f"- **ID:** `{qid(ex)}`")
        out.append(f"- **Answer:** {ex['answer']}")
        out.append(f"- **Type:** {ex['type']}")
        out.append("")

        out.append("**Supporting facts** (title, sentence index):")
        out.append("")
        for title, sid in sf:
            out.append(f"- {title} [{sid}]")
        out.append("")

        if evidences:
            out.append("**Evidences** (fact, relation, entity):")
            out.append("")
            for ev in evidences:
                if isinstance(ev, dict):
                    out.append(f"- {ev.get('fact')} | {ev.get('relation')} | {ev.get('entity')}")
                else:
                    out.append(f"- {' | '.join(str(x) for x in ev)}")
            out.append("")

        out.append("**Context:**")
        out.append("")
        for title, sents in context_paragraphs(ex):
            out.append(f"### {title}")
            out.append("")
            for i, s in enumerate(sents):
                text = s.strip()
                tag = " **[SF]**" if (title, i) in sf_set else ""
                out.append(f"- `[{i}]`{tag} {text}")
            out.append("")

        out.append("---")
        out.append("")

    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"saved {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="wiki2_data")
    ap.add_argument("--hf-dataset", default="framolfese/2WikiMultihopQA")
    ap.add_argument("--official-dir", default=None,
                    help="folder with official train.json and dev.json")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = get_split("train", args)
    save_json(train, out / "2wiki_train.json")
    print(f"train examples: {len(train)}")

    val = get_split("validation", args)
    save_json(val, out / "2wiki_dev.json")
    print(f"validation examples: {len(val)}")

    subset = sample_eval_set(val)
    save_json(subset, out / "2wiki_100.json")

    (out / "2wiki_100.md").write_text(to_markdown(subset), encoding="utf-8")
    print(f"saved {out / '2wiki_100.md'}")

    # sanity checks
    assert len(subset) == 100
    assert len({qid(x) for x in subset}) == 100
    for t in TYPES:
        assert sum(1 for x in subset if x["type"] == t) == N_PER_TYPE
    print(f"OK: 100 unique questions, {N_PER_TYPE} each of {TYPES}")


if __name__ == "__main__":
    main()