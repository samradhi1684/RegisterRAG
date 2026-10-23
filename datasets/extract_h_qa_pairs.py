#!/usr/bin/env python3
"""
Download the full HotpotQA *distractor* train + validation sets, save them
unmodified as JSON, and build a fixed 100-question evaluation subset
(50 bridge + 50 comparison, seed=42) from the validation set.

Outputs (in --out-dir, default ./hotpotqa_data):
    hotpot_train_distractor.json   full train set, all original fields
    hotpot_dev_distractor.json     full validation set, all original fields
    hotpot_100.json                50 bridge + 50 comparison, all original fields
    hotpot_100.md                  human-readable view of the 100 questions

Primary source: the official HotpotQA JSON files (original schema:
_id, question, answer, type, level, supporting_facts=[[title, sent_id]],
context=[[title, [sentences]]]).
Fallback: HuggingFace `hotpotqa/hotpot_qa` ("distractor" config), which has the
same fields but a columnar schema (supporting_facts / context as dicts of lists).
Both are handled by the markdown writer.

Usage:
    python prepare_hotpotqa.py
    python prepare_hotpotqa.py --out-dir data --source hf
"""

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

SEED = 42
N_PER_TYPE = 50

OFFICIAL_URLS = {
    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "validation": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_official(split: str) -> list:
    url = OFFICIAL_URLS[split]
    print(f"[official] downloading {split}: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_hf(split: str) -> list:
    from datasets import load_dataset  # pip install datasets

    print(f"[hf] downloading {split} (distractor)")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    return [dict(row) for row in ds]  # every field kept as-is


def get_split(split: str, source: str) -> list:
    if source == "official":
        return download_official(split)
    if source == "hf":
        return download_hf(split)
    # auto: try official first, fall back to HF
    try:
        return download_official(split)
    except Exception as e:  # noqa: BLE001
        print(f"  official download failed ({e!r}); falling back to HuggingFace")
        return download_hf(split)


# --------------------------------------------------------------------------- #
# Schema helpers (handle official + HF layouts)
# --------------------------------------------------------------------------- #
def qid(ex: dict) -> str:
    return ex.get("_id", ex.get("id"))


def supporting_pairs(ex: dict) -> list:
    """Return [(title, sent_id), ...] regardless of schema."""
    sf = ex["supporting_facts"]
    if isinstance(sf, dict):  # HF: {"title": [...], "sent_id": [...]}
        return list(zip(sf["title"], sf["sent_id"]))
    return [(t, i) for t, i in sf]  # official: [[title, sent_id], ...]


def context_paragraphs(ex: dict) -> list:
    """Return [(title, [sentences]), ...] regardless of schema."""
    ctx = ex["context"]
    if isinstance(ctx, dict):  # HF: {"title": [...], "sentences": [[...], ...]}
        return list(zip(ctx["title"], ctx["sentences"]))
    return [(t, s) for t, s in ctx]  # official: [[title, [sentences]], ...]


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_eval_set(val: list) -> list:
    rng = random.Random(SEED)
    by_type = {"bridge": [], "comparison": []}
    for ex in val:
        if ex["type"] in by_type:
            by_type[ex["type"]].append(ex)

    picked = []
    for t in ("bridge", "comparison"):
        # sort by id so the sample doesn't depend on source file ordering
        pool = sorted(by_type[t], key=qid)
        if len(pool) < N_PER_TYPE:
            sys.exit(f"Not enough '{t}' questions: {len(pool)} < {N_PER_TYPE}")
        picked.extend(rng.sample(pool, N_PER_TYPE))

    rng.shuffle(picked)  # mix the two types, still deterministic under seed=42
    return picked


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def to_markdown(items: list) -> str:
    n_bridge = sum(1 for x in items if x["type"] == "bridge")
    n_comp = sum(1 for x in items if x["type"] == "comparison")

    out = [
        "# HotpotQA (distractor) - 100-question evaluation set",
        "",
        f"- Source split: validation (distractor)",
        f"- Sampling: seed={SEED}, {n_bridge} bridge + {n_comp} comparison",
        f"- Total questions: {len(items)}",
        "- In the context, sentences marked with **[SF]** are supporting facts.",
        "",
        "---",
        "",
    ]

    for n, ex in enumerate(items, 1):
        sf = supporting_pairs(ex)
        sf_set = set(sf)

        out.append(f"## {n}. {ex['question']}")
        out.append("")
        out.append(f"- **ID:** `{qid(ex)}`")
        out.append(f"- **Answer:** {ex['answer']}")
        out.append(f"- **Type:** {ex['type']}")
        out.append(f"- **Difficulty:** {ex['level']}")
        out.append("")
        out.append("**Supporting facts** (title, sentence index):")
        out.append("")
        for title, sid in sf:
            out.append(f"- {title} [{sid}]")
        out.append("")
        out.append("**Context:**")
        out.append("")
        for title, sents in context_paragraphs(ex):
            out.append(f"### {title}")
            out.append("")
            for i, s in enumerate(sents):
                text = s.strip()
                if (title, i) in sf_set:
                    out.append(f"- **[SF]** `[{i}]` {text}")
                else:
                    out.append(f"- `[{i}]` {text}")
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
    ap.add_argument("--out-dir", default="hotpotqa_data")
    ap.add_argument("--source", choices=["auto", "official", "hf"], default="auto")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = get_split("train", args.source)
    save_json(train, out / "hotpot_train_distractor.json")
    print(f"train examples: {len(train)}")

    val = get_split("validation", args.source)
    save_json(val, out / "hotpot_dev_distractor.json")
    print(f"validation examples: {len(val)}")

    subset = sample_eval_set(val)
    save_json(subset, out / "hotpot_100.json")

    md = to_markdown(subset)
    (out / "hotpot_100.md").write_text(md, encoding="utf-8")
    print(f"saved {out / 'hotpot_100.md'}")

    # sanity checks
    types = [x["type"] for x in subset]
    assert len(subset) == 100
    assert types.count("bridge") == 50 and types.count("comparison") == 50
    assert len({qid(x) for x in subset}) == 100
    print("OK: 100 unique questions (50 bridge, 50 comparison)")


if __name__ == "__main__":
    main()