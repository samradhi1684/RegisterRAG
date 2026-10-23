import json
import random
import argparse
from collections import defaultdict

def split_and_replace(
    original_json: str,       # repolicyqa_100.json
    source_json: str,         # test_data.json (full dataset)
    output_85: str,           # repolicyqa_85_plus_15.json  (final 100 w/ replacements)
    output_15_missing: str,   # repolicyqa_15_missing.json  (the 15 you don't have)
    missing_doc_ids: list,    # [22515, 22706, 22718]
    seed: int = 42,
):
    # ── Load files ─────────────────────────────────────────────────────────
    with open(original_json, "r", encoding="utf-8") as f:
        original = json.load(f)

    with open(source_json, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    missing_set  = set(missing_doc_ids)
    original_doc_ids = set(r["document_id"] for r in original)

    # ── Split original into 85 (keep) and 15 (missing) ────────────────────
    keep_85  = [r for r in original if r["document_id"] not in missing_set]
    gone_15  = [r for r in original if r["document_id"] in missing_set]

    print(f"Original total     : {len(original)}")
    print(f"Records to keep    : {len(keep_85)}  (docs NOT in missing list)")
    print(f"Records to replace : {len(gone_15)}  (docs {sorted(missing_doc_ids)})")

    # ── Verify the 15 are perfectly stratified (5 per type × 3 docs) ──────
    gone_types = defaultdict(int)
    gone_docs  = defaultdict(int)
    for r in gone_15:
        gone_types[r["question_type"]] += 1
        gone_docs[r["document_id"]]    += 1
    print(f"\nMissing 15 — by type : {dict(gone_types)}")
    print(f"Missing 15 — by doc  : {dict(gone_docs)}")

    # ── Find 3 replacement docs from the full dataset ─────────────────────
    # Constraints:
    #   - Not in the original 20 docs
    #   - Not in missing_doc_ids
    #   - Must have ≥1 question of every type (so we can pick 1 per type per doc)

    all_types = sorted(set(r["question_type"] for r in original))
    n_types   = len(all_types)   # 5

    # Index full dataset
    full_index = defaultdict(lambda: defaultdict(list))
    for record in full_data:
        full_index[record["document_id"]][record.get("question_type","unknown")].append(record)

    # Eligible replacement docs: not already used, has all types
    eligible = [
        doc_id for doc_id, type_map in full_index.items()
        if doc_id not in original_doc_ids
        and doc_id not in missing_set
        and all(len(type_map.get(qt, [])) >= 1 for qt in all_types)
    ]

    n_replacement_docs = len(missing_doc_ids)   # 3
    print(f"\nEligible replacement docs (not in original 20, has all types): {len(eligible)}")

    if len(eligible) < n_replacement_docs:
        raise ValueError(
            f"Only {len(eligible)} eligible replacement docs found, need {n_replacement_docs}."
        )

    random.seed(seed)
    replacement_docs = sorted(random.sample(eligible, n_replacement_docs))
    print(f"Chosen replacement doc IDs: {replacement_docs}")

    # ── Sample 5 questions (1 per type) from each replacement doc ─────────
    new_15 = []
    for doc_id in replacement_docs:
        for qtype in all_types:
            record = random.choice(full_index[doc_id][qtype])
            new_15.append(record)

    print(f"\nReplacement 15 — by type : ", end="")
    rep_types = defaultdict(int)
    rep_docs  = defaultdict(int)
    for r in new_15:
        rep_types[r["question_type"]] += 1
        rep_docs[r["document_id"]]    += 1
    print(dict(rep_types))
    print(f"Replacement 15 — by doc  : {dict(rep_docs)}")

    # ── Build final 100 = 85 kept + 15 new ───────────────────────────────
    final_100 = keep_85 + new_15
    random.shuffle(final_100)

    # ── Write outputs ──────────────────────────────────────────────────────
    with open(output_15_missing, "w", encoding="utf-8") as f:
        json.dump(gone_15, f, indent=2, ensure_ascii=False)

    with open(output_85, "w", encoding="utf-8") as f:
        json.dump(final_100, f, indent=2, ensure_ascii=False)

    # ── Final verification ─────────────────────────────────────────────────
    final_type_counts = defaultdict(int)
    final_doc_counts  = defaultdict(int)
    for r in final_100:
        final_type_counts[r["question_type"]] += 1
        final_doc_counts[r["document_id"]]    += 1

    print(f"\n── Output: {output_15_missing} ──")
    print(f"  Records: {len(gone_15)}  (the 15 questions from missing docs)")

    print(f"\n── Output: {output_85} (85 kept + 15 replacements = 100) ──")
    print(f"  Total records      : {len(final_100)}")
    print(f"  Unique doc IDs     : {len(final_doc_counts)}  (target: 20)")
    print(f"  Stratification check:")
    all_ok = True
    for qt in sorted(final_type_counts):
        ok = "✓" if final_type_counts[qt] == 20 else "✗"
        if final_type_counts[qt] != 20: all_ok = False
        print(f"    {ok} {qt}: {final_type_counts[qt]}")
    print("    → perfectly stratified ✓" if all_ok else "    → stratification FAILED ✗")
    print(f"\n  Questions per doc:")
    for doc_id in sorted(final_doc_counts):
        tag = " ← NEW" if doc_id in replacement_docs else ""
        print(f"    doc {doc_id}: {final_doc_counts[doc_id]} questions{tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split out missing-doc questions and replace with new docs"
    )
    parser.add_argument("--input",           required=True,  help="repolicyqa_100.json")
    parser.add_argument("--source",          required=True,  help="Full test_data.json")
    parser.add_argument("--output_100",      default="repolicyqa_100_replaced.json")
    parser.add_argument("--output_missing",  default="repolicyqa_15_missing.json")
    parser.add_argument("--missing_doc_ids", nargs="+", type=int,
                        default=[22515, 22706, 22718],
                        help="Doc IDs you don't have access to")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_and_replace(
        original_json      = args.input,
        source_json        = args.source,
        output_85          = args.output_100,
        output_15_missing  = args.output_missing,
        missing_doc_ids    = args.missing_doc_ids,
        seed               = args.seed,
    )