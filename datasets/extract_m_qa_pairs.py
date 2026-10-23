#!/usr/bin/env python3
"""
Download the full MuSiQue (answerable, "MuSiQue-Ans") train + validation sets,
save them unmodified as JSON, and build a fixed 100-question evaluation subset
from the validation set, stratified by hop count (2 / 3 / 4 hops), seed=42.

Outputs (in --out-dir, default ./musique_data):
    musique_train.json        full train set, all original fields
    musique_dev.json          full validation set, all original fields
    musique_100.json          100 hop-stratified questions, all original fields
    musique_100.md            human-readable view of the 100 questions

MuSiQue has no "type"/"difficulty" fields. The hop count is derived from
len(question_decomposition) (cross-checked against the id prefix, e.g.
"2hop__...", "3hop1__...", "4hop2__...") and is used as the stratum.

Original fields per example:
    id, paragraphs[{idx, title, paragraph_text, is_supporting}], question,
    question_decomposition[{id, question, answer, paragraph_support_idx}],
    answer, answer_aliases, answerable

Sources:
    default          HuggingFace dataset (--hf-dataset, default dgslibisey/MuSiQue)
    --official-dir   folder holding the official JSONL files from the MuSiQue repo:
                     musique_ans_v1.0_train.jsonl, musique_ans_v1.0_dev.jsonl

Usage:
    pip install datasets
    python prepare_musique.py
    python prepare_musique.py --allocation 2:50,3:30,4:20
    python prepare_musique.py --official-dir ./musique_official
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

SEED = 42
DEFAULT_ALLOCATION = "2:34,3:33,4:33"  # equal split of 100; extra question goes to 2-hop


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_hf(dataset_name: str, split: str) -> list:
    from datasets import load_dataset  # pip install datasets

    print(f"[hf] loading {dataset_name} ({split})")
    ds = load_dataset(dataset_name, split=split)
    return [dict(row) for row in ds]  # every field kept as-is


def load_official(folder: str, split: str) -> list:
    fname = {
        "train": "musique_ans_v1.0_train.jsonl",
        "validation": "musique_ans_v1.0_dev.jsonl",
    }[split]
    path = Path(folder) / fname
    print(f"[official] reading {path}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_split(split: str, args) -> list:
    if args.official_dir:
        return load_official(args.official_dir, split)
    return load_hf(args.hf_dataset, split)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def n_hops(ex: dict) -> int:
    hops = len(ex["question_decomposition"])
    m = re.match(r"(\d)hop", ex["id"])
    if m and int(m.group(1)) != hops:
        print(f"  warning: {ex['id']} id prefix says {m.group(1)} hops, "
              f"decomposition has {hops}; using decomposition")
    return hops


def parse_allocation(s: str) -> dict:
    alloc = {}
    for part in s.split(","):
        h, n = part.split(":")
        alloc[int(h)] = int(n)
    total = sum(alloc.values())
    if total != 100:
        sys.exit(f"Allocation must sum to 100, got {total}")
    return alloc


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_eval_set(val: list, allocation: dict) -> list:
    rng = random.Random(SEED)
    by_hop = {h: [] for h in allocation}
    for ex in val:
        h = n_hops(ex)
        if h in by_hop:
            by_hop[h].append(ex)

    picked = []
    for h in sorted(allocation):
        # sort by id so the sample doesn't depend on source ordering
        pool = sorted(by_hop[h], key=lambda x: x["id"])
        if len(pool) < allocation[h]:
            sys.exit(f"Not enough {h}-hop questions: {len(pool)} < {allocation[h]}")
        picked.extend(rng.sample(pool, allocation[h]))

    rng.shuffle(picked)  # mix hop counts, still deterministic under seed=42
    return picked


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def to_markdown(items: list, allocation: dict) -> str:
    counts = {h: sum(1 for x in items if n_hops(x) == h) for h in sorted(allocation)}
    dist = ", ".join(f"{n} x {h}-hop" for h, n in counts.items())

    out = [
        "# MuSiQue (answerable) - 100-question evaluation set",
        "",
        "- Source split: validation",
        f"- Sampling: seed={SEED}, stratified by hop count ({dist})",
        f"- Total questions: {len(items)}",
        "- MuSiQue has no type/difficulty labels; hop count is used instead.",
        "- Paragraphs marked **[SUPPORTING]** are the gold supporting paragraphs.",
        "",
        "---",
        "",
    ]

    for n, ex in enumerate(items, 1):
        paras = ex["paragraphs"]
        by_idx = {p["idx"]: p for p in paras}
        supporting = [p for p in paras if p["is_supporting"]]

        out.append(f"## {n}. {ex['question']}")
        out.append("")
        out.append(f"- **ID:** `{ex['id']}`")
        out.append(f"- **Answer:** {ex['answer']}")
        aliases = ex.get("answer_aliases") or []
        out.append(f"- **Answer aliases:** {', '.join(aliases) if aliases else '(none)'}")
        out.append(f"- **Hops:** {n_hops(ex)}")
        out.append(f"- **Answerable:** {ex.get('answerable')}")
        out.append("")

        out.append("**Question decomposition:**")
        out.append("")
        for step in ex["question_decomposition"]:
            p = by_idx.get(step["paragraph_support_idx"])
            title = p["title"] if p else "?"
            out.append(
                f"{step['id']}. {step['question']} -> **{step['answer']}** "
                f"_(paragraph {step['paragraph_support_idx']}: {title})_"
            )
        out.append("")

        out.append("**Supporting paragraphs:**")
        out.append("")
        for p in supporting:
            out.append(f"- [{p['idx']}] {p['title']}")
        out.append("")

        out.append(f"**Full context ({len(paras)} paragraphs):**")
        out.append("")
        for p in paras:
            tag = " **[SUPPORTING]**" if p["is_supporting"] else ""
            out.append(f"### [{p['idx']}] {p['title']}{tag}")
            out.append("")
            out.append(p["paragraph_text"].strip())
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
    ap.add_argument("--out-dir", default="musique_data")
    ap.add_argument("--hf-dataset", default="dgslibisey/MuSiQue")
    ap.add_argument("--official-dir", default=None,
                    help="folder with official musique_ans_v1.0_{train,dev}.jsonl")
    ap.add_argument("--allocation", default=DEFAULT_ALLOCATION,
                    help='questions per hop count, must sum to 100, e.g. "2:34,3:33,4:33"')
    args = ap.parse_args()

    allocation = parse_allocation(args.allocation)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = get_split("train", args)
    save_json(train, out / "musique_train.json")
    print(f"train examples: {len(train)}")

    val = get_split("validation", args)
    save_json(val, out / "musique_dev.json")
    print(f"validation examples: {len(val)}")
    dist = {}
    for ex in val:
        dist[n_hops(ex)] = dist.get(n_hops(ex), 0) + 1
    print(f"validation hop distribution: {dict(sorted(dist.items()))}")

    subset = sample_eval_set(val, allocation)
    save_json(subset, out / "musique_100.json")

    (out / "musique_100.md").write_text(to_markdown(subset, allocation), encoding="utf-8")
    print(f"saved {out / 'musique_100.md'}")

    # sanity checks
    assert len(subset) == 100
    assert len({x["id"] for x in subset}) == 100
    for h, n in allocation.items():
        assert sum(1 for x in subset if n_hops(x) == h) == n
    print("OK: 100 unique questions, hop counts match allocation")


if __name__ == "__main__":
    main()