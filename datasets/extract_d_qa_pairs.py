import json
import random
import argparse
from collections import defaultdict

def extract_stratified(
    input_path: str,
    output_path: str,
    n_docs: int = 20,
    n_questions: int = 100,
    seed: int = 42,
    fields: list = None
):
    """
    Extract n_questions with:
      - Exactly n_docs unique document IDs in the output
      - Equal stratification across all question_type values
      - Questions distributed as evenly as possible across the 20 docs

    Strategy:
      1. Find all doc IDs that have AT LEAST ONE question of EVERY type
         (so every type can be represented within the same 20 docs).
      2. Randomly pick n_docs from that eligible set.
      3. Within those docs, for each type: distribute the per-type quota
         evenly across the 20 docs and sample accordingly.
    """
    fields = fields or [
        "document_id",
        "chunk",
        "question",
        "answer",
        "context",
        "conditions",
        "question_type",
        "program_category",
    ]

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total records in file: {len(data)}")

    # ── Build index: doc_id -> {qtype -> [records]} ────────────────────────
    index = defaultdict(lambda: defaultdict(list))
    for record in data:
        index[record["document_id"]][record.get("question_type", "unknown")].append(record)

    all_types  = sorted({r.get("question_type", "unknown") for r in data})
    n_types    = len(all_types)
    print(f"Question types found ({n_types}): {all_types}")

    if n_questions % n_types != 0:
        nearest_down = n_types * (n_questions // n_types)
        nearest_up   = nearest_down + n_types
        raise ValueError(
            f"n_questions ({n_questions}) must be divisible by {n_types} types. "
            f"Try {nearest_down} or {nearest_up}."
        )

    per_type_quota = n_questions // n_types          # e.g. 20
    per_type_per_doc = per_type_quota // n_docs      # base questions per doc per type
    # e.g. 20 // 20 = 1 per doc per type (minimum)

    print(f"Per-type quota  : {per_type_quota} questions × {n_types} types = {n_questions} total")
    print(f"Per doc per type: ~{per_type_per_doc} (base, remainder spread across docs)")

    # ── Step 1: Find eligible docs (have >= per_type_per_doc+1 of EVERY type) ──
    # We need each doc to contribute at least per_type_per_doc questions per type.
    # Minimum requirement: at least 1 question per type per doc.
    min_per_type = max(1, per_type_per_doc)

    eligible_docs = [
        doc_id for doc_id, type_map in index.items()
        if all(len(type_map.get(qt, [])) >= min_per_type for qt in all_types)
    ]

    print(f"Docs with ≥{min_per_type} question(s) of every type: {len(eligible_docs)}")

    if len(eligible_docs) < n_docs:
        # Fallback: relax to docs that have at least 1 of every type
        eligible_docs = [
            doc_id for doc_id, type_map in index.items()
            if all(len(type_map.get(qt, [])) >= 1 for qt in all_types)
        ]
        print(f"Relaxed to docs with ≥1 of every type: {len(eligible_docs)}")
        if len(eligible_docs) < n_docs:
            raise ValueError(
                f"Only {len(eligible_docs)} docs have at least 1 question of every type, "
                f"but n_docs={n_docs} requested. "
                f"Try reducing n_docs to ≤{len(eligible_docs)}."
            )

    # ── Step 2: Randomly pick n_docs ──────────────────────────────────────
    random.seed(seed)
    chosen_docs = sorted(random.sample(eligible_docs, n_docs))
    print(f"Chosen doc IDs: {chosen_docs}")

    # ── Step 3: For each type, distribute per_type_quota across chosen_docs ──
    # Base: per_type_per_doc per doc. Remainder goes to docs with most questions
    # of that type (so we never ask for more than a doc has).
    extracted = []

    for qtype in all_types:
        # Available count per chosen doc for this type
        avail = {doc_id: len(index[doc_id].get(qtype, [])) for doc_id in chosen_docs}

        # Sort docs by availability descending to assign remainder greedily
        sorted_docs = sorted(chosen_docs, key=lambda d: avail[d], reverse=True)

        base      = per_type_quota // n_docs       # floor
        remainder = per_type_quota % n_docs        # extra slots

        quotas = {}
        for i, doc_id in enumerate(sorted_docs):
            quota = base + (1 if i < remainder else 0)
            # Cap at what's available
            quotas[doc_id] = min(quota, avail[doc_id])

        # If any doc was capped below its quota, redistribute shortfall
        total = sum(quotas.values())
        shortfall = per_type_quota - total
        if shortfall > 0:
            for doc_id in sorted_docs:
                if shortfall == 0:
                    break
                headroom = avail[doc_id] - quotas[doc_id]
                add = min(headroom, shortfall)
                quotas[doc_id] += add
                shortfall -= add

        if shortfall > 0:
            raise ValueError(
                f"Cannot fill {per_type_quota} questions for type '{qtype}' "
                f"across the chosen {n_docs} docs (short by {shortfall}). "
                f"Try a different seed or fewer questions."
            )

        # Sample
        for doc_id, take in quotas.items():
            if take > 0:
                sampled = random.sample(index[doc_id][qtype], take)
                extracted.extend(
                    {field: r[field] for field in fields if field in r}
                    for r in sampled
                )

    random.shuffle(extracted)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)

    # ── Verification ───────────────────────────────────────────────────────
    unique_docs = set(r["document_id"] for r in extracted)
    type_counts = defaultdict(int)
    per_doc     = defaultdict(int)
    for r in extracted:
        type_counts[r.get("question_type", "unknown")] += 1
        per_doc[r["document_id"]] += 1

    print(f"\n✓ Extracted {len(extracted)} questions")
    print(f"✓ Unique document IDs: {len(unique_docs)} (target: {n_docs})")

    print("\nStratification check (question_type → count):")
    all_equal = True
    for qt in sorted(type_counts):
        ok = "✓" if type_counts[qt] == per_type_quota else "✗"
        if type_counts[qt] != per_type_quota:
            all_equal = False
        print(f"  {ok} {qt}: {type_counts[qt]}")
    print("  → perfectly stratified ✓" if all_equal else "  → stratification FAILED ✗")

    print(f"\nQuestions per document_id (across all types):")
    for doc_id in sorted(per_doc):
        print(f"  doc {doc_id}: {per_doc[doc_id]} questions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract stratified questions from exactly N unique docs in RePolicyQA"
    )
    parser.add_argument("--input",       default="test.json",           help="Path to test.json")
    parser.add_argument("--output",      default="repolicyqa_100.json", help="Output file path")
    parser.add_argument("--n_docs",      type=int, default=20,          help="Number of unique docs (hard constraint)")
    parser.add_argument("--n_questions", type=int, default=100,         help="Total questions (must be divisible by #types)")
    parser.add_argument("--seed",        type=int, default=42,          help="Random seed")
    args = parser.parse_args()

    extract_stratified(
        input_path=args.input,
        output_path=args.output,
        n_docs=args.n_docs,
        n_questions=args.n_questions,
        seed=args.seed,
    )