#!/usr/bin/env python3
"""
Run four RAG systems on a corpus of markdown policy documents + a JSON question file:

    vector           fixed-size chunks -> dense retrieval              -> generator
    vector_rerank    fixed-size chunks -> dense top-N -> cross-encoder -> generator
    sentence         sentences         -> dense retrieval              -> generator
    sentence_rerank  sentences         -> dense top-N -> cross-encoder -> generator

All four use the SAME corpus, questions, embedder, top-k, reranker settings, generator, prompt,
seeds and evaluation: they are the shared driver in `vector_rag.py` with a different chunker
(`vector_rag.FixedSizeChunker` / `sentence_rag.SentenceChunker`) and with/without `--reranker`.
Keep vector_rag.py, sentence_rag.py and policy_rag.py in the same folder.

Question file: JSON array (or JSONL). Fields used: question, answer (gold answer), context
(gold evidence, "- " bullets), document_id (source document), chunk (source document text).
Every other field (conditions, question_type, program_category, ...) is copied into
`meta` of each results.jsonl record and is available for breakdowns.

Gold source document is resolved per question, in this order:
  1. cleaned `chunk` text equals a cleaned markdown file          ("content")
  2. a markdown file name contains the question's document_id     ("filename")
  3. the markdown title equals the title in `chunk`               ("title")
Unresolved questions get no document-level metrics (counts are printed and saved).
Naming the files <document_id>.md makes this unambiguous.

Metrics (in addition to what the other scripts log): the gold answers here are sentences, so
EM is expected to be near 0. Use F1 / ROUGE-L / answer token recall for answer quality and
`evidence_coverage` (token-level share of the gold `context` found in the retrieved units)
for retrieval quality. All are non-LLM, deterministic metrics.

Example:
  python policy_rag.py --md-dir policies/ --questions questions.json \
      --embedder BAAI/bge-base-en-v1.5 --reranker ms-marco-MiniLM-L-6-v2 --retrieve-k 50 \
      --top-k 5 --generator qwen2.5:14b --seed 42
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sympy import root

import sentence_rag as sr
import vector_rag as vr

PREPROCESSING_VERSION = "md-clean-v2"    # v2: fixed-size chunks -> pysbd inside each chunk
SYSTEMS = {                 # name -> (chunker kind, uses reranker)
                            # sentence = ChunkSentenceChunker (chunk first, then pysbd inside each chunk)
    "vector": ("fixed", False),
    "vector_rerank": ("fixed", True),
    "sentence": ("sentence", False),
    "sentence_rerank": ("sentence", True),
}

# ============================================================================
# Markdown -> plain text (deterministic, one block per line)
# ============================================================================
_FIELD = re.compile(r"^\s*\*{2,3}[^*\n]+\*{2,3}\s*:")      # ***key*** : value
_BULLET = re.compile(r"^\s*[•\-\*]\s+")
_HEAD = re.compile(r"^\s{0,3}(#{1,6})\s*(.*?)\s*#*\s*$")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _inline(s: str) -> str:
    s = _LINK.sub(r"\1", s)
    s = re.sub(r"\*+", "", s)                  # emphasis markers
    return " ".join(s.split())


def clean_markdown(raw: str, fallback_title: str) -> tuple[str, str]:
    """Returns (title, text). Blocks (paragraph / field line / list item / heading) become
    separate lines; hard-wrapped lines inside a paragraph are joined. Emphasis markers and
    short structural headings (level >= 2, <= 3 words, e.g. '## Summary :') are removed."""
    blocks: list[str] = []
    cur: list[str] = []
    title: Optional[str] = None

    def flush() -> None:
        if cur:
            b = _BULLET.sub("", " ".join(cur), count=1)
            b = _inline(b)
            if b:
                blocks.append(b)
            cur.clear()

    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            flush()
            continue
        m = _HEAD.match(line)
        if m:
            flush()
            level, htext = len(m.group(1)), _inline(m.group(2))
            if level == 1 and title is None and htext:
                title = htext
            if htext and not (level >= 2 and len(htext.split()) <= 3):
                blocks.append(htext)
            continue
        if _FIELD.match(line) or _BULLET.match(line):
            flush()
        cur.append(line.strip())
    flush()
    return (title or fallback_title), "\n".join(blocks)


# ============================================================================
# Corpus + questions
# ============================================================================
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_markdown_corpus(md_dir: str) -> tuple[dict[str, vr.Doc], dict]:
    files = sorted(Path(md_dir).rglob("*.md"))
    if not files:
        raise SystemExit(f"No .md files found under {md_dir}")
    docs: dict[str, vr.Doc] = {}
    idx = {"stem": {}, "text": {}, "title": {}}
    h = hashlib.sha256()
    for f in files:
        raw = f.read_text(encoding="utf-8")
        h.update(f.name.encode()); h.update(b"\0"); h.update(raw.encode()); h.update(b"\0")
        title, text = clean_markdown(raw, f.stem)
        if not text:
            continue
        did = vr.make_doc_id(title, text)
        docs.setdefault(did, vr.Doc(did, title, text))
        idx["stem"][f.stem] = did
        idx["text"][_sha(text)] = did
        idx["title"].setdefault(" ".join(title.lower().split()), set()).add(did)
    info = {"md_dir": str(md_dir), "n_md_files": len(files), "n_docs": len(docs),
            "md_files_sha256": h.hexdigest(), "preprocessing": PREPROCESSING_VERSION}
    return docs, {"idx": idx, "info": info}


def resolve_gold_doc(rec: dict, idx: dict) -> tuple[Optional[str], str]:
    chunk = rec.get("chunk")
    if chunk:
        title, text = clean_markdown(chunk, "")
        if _sha(text) in idx["text"]:
            return idx["text"][_sha(text)], "content"
    did = rec.get("document_id")
    if did is not None:
        sid = str(did)
        if sid in idx["stem"]:
            return idx["stem"][sid], "filename"
        pat = re.compile(rf"(?<!\d){re.escape(sid)}(?!\d)")
        hits = {d for stem, d in idx["stem"].items() if pat.search(stem)}
        if len(hits) == 1:
            return next(iter(hits)), "filename"
    if chunk:
        cands = idx["title"].get(" ".join(clean_markdown(chunk, "")[0].lower().split()), set())
        if len(cands) == 1:
            return next(iter(cands)), "title"
    return None, "unresolved"


def parse_evidence(context) -> list[str]:
    out = []
    for line in str(context or "").splitlines():
        line = re.sub(r"^\s*[-•*]\s*", "", line).strip()
        if line:
            out.append(" ".join(line.split()))
    return out


def load_policy(args: argparse.Namespace):
    docs, c = load_markdown_corpus(args.md_dir)
    idx, info = c["idx"], c["info"]
    records = vr.read_json_records(args.questions)
    indexed = list(enumerate(records))                       # qid keeps the position in the file
    if args.limit is not None and args.limit < len(indexed):
        indexed = (indexed[:args.limit] if args.sample_seed is None
                   else random.Random(args.sample_seed).sample(indexed, args.limit))
    all_doc_ids = list(docs)
    examples, counts = [], {"content": 0, "filename": 0, "title": 0, "unresolved": 0}
    unresolved = []
    for i, rec in indexed:
        for key in ("question", "answer"):
            if key not in rec:
                raise SystemExit(f"Question #{i} has no '{key}' field")
        gold, how = resolve_gold_doc(rec, idx)
        counts[how] += 1
        qid = f"q{i:04d}"
        if gold is None:
            unresolved.append(qid)
        meta = {k: v for k, v in rec.items() if k not in ("question", "answer", "chunk")}
        meta["gold_doc_resolution"] = how
        if args.log_source_chunk and "chunk" in rec:
            meta["chunk"] = rec["chunk"]
        examples.append(vr.Example(qid, " ".join(str(rec["question"]).split()),
                                   [str(rec["answer"])], all_doc_ids,
                                   [gold] if gold else [], meta, parse_evidence(rec.get("context"))))
    info.update({"questions_file": str(args.questions),
                 "questions_sha256": vr.file_sha256(args.questions),
                 "n_questions_in_file": len(records), "n_questions_used": len(examples),
                 "gold_doc_resolution": counts, "limit": args.limit,
                 "sample_seed": args.sample_seed})
    if unresolved:
        vr.logger.warning("%d/%d questions have no resolvable gold document (e.g. %s); they get "
                          "no document-level metrics. Name files <document_id>.md to fix.",
                          len(unresolved), len(examples), unresolved[:5])
    return examples, docs, info


# ============================================================================
# Reporting
# ============================================================================
def _fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def _group_means(recs: list[dict], field: str, metrics: list[str]) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in recs:
        groups.setdefault(str(r.get("meta", {}).get(field)), []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        row = {"n": len(rs)}
        for m in metrics:
            xs = [r["metrics"].get(m) for r in rs if r["metrics"].get(m) is not None]
            row[m] = float(np.mean(xs)) if xs else None
        out[g] = row
    return out


def write_comparison(root: Path, systems: list[str]) -> None:
    cols = [("EM", "EM"), ("F1", "F1"), ("ROUGE-L", "ROUGE-L"),
            ("answer_token_recall", "AnsRecall"), ("evidence_coverage@k", "EvCov@k"),
            ("evidence_precision@k", "EvPrec@k"), ("retrieval_doc_recall@k", "DocRecall@k")]
    summaries = {s: json.loads((root / s / "summary.json").read_text(encoding="utf-8")) for s in systems}
    lines = ["| system | " + " | ".join(c[1] for c in cols) + " | latency_s (mean) | prompt tok/q |",
             "|---|" + "---|" * (len(cols) + 2)]
    for s, sm in summaries.items():
        n = max(sm["n_questions"], 1)
        lines.append(f"| {s} | " + " | ".join(_fmt(sm.get(c[0])) for c in cols)
                     + f" | {_fmt(sm['latency_s_mean'].get('total_s'))}"
                     + f" | {sm['tokens_total']['prompt_est_cl100k'] / n:.0f} |")
    breakdown = {}
    metrics = ["f1", "rougeL", "answer_recall", "evidence_coverage", "doc_recall"]
    for s in systems:

        recs = [
            json.loads(line)
            for line in (
                root / s / "results.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        breakdown[s] = {fld: _group_means(recs, fld, metrics) for fld in ("question_type", "program_category")}
    for fld in ("question_type", "program_category"):
        groups = sorted({g for s in systems for g in breakdown[s][fld]})
        lines += ["", f"ROUGE-L by {fld} (n in brackets)", "",
                  f"| {fld} | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
        for g in groups:
            n = next((breakdown[s][fld][g]["n"] for s in systems if g in breakdown[s][fld]), 0)
            lines.append(f"| {g} [{n}] | " + " | ".join(
                _fmt(breakdown[s][fld].get(g, {}).get("rougeL")) for s in systems) + " |")
    (root / "comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (root / "comparison.json").write_text(
        json.dumps(
            {"summaries": summaries, "breakdown": breakdown},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\n" + "\n".join(lines[:2 + len(systems)]) + f"\n\nFull tables: {root / 'comparison.md'}")


# ============================================================================
# Two-stage chunker: fixed-size -> pysbd sentence split inside each chunk
# ============================================================================
class ChunkSentenceChunker:
    """Stage-1: split each document into fixed-size token/word chunks using the same
    FixedSizeChunker and parameters as the Vector RAG systems (same chunk_size, overlap,
    unit). Stage-2: run pysbd sentence segmentation inside each chunk. Each sentence
    becomes one retrieval unit.

    chunk_id format: "<doc_id>:c<chunk_idx>:s<sent_idx>"

    Why chunk first, then split sentences:
      - the two vector systems and the two sentence systems cover identical text; the only
        difference is the granularity of the retrieval unit (fair comparison);
      - sentence boundaries are detected within a bounded context window, not across an
        entire long document, so pysbd's accuracy is higher;
      - token budget per "neighbourhood" is the same as for the chunk systems.
    """

    def __init__(self, fixed: vr.FixedSizeChunker, splitter: sr.SentenceSplitter):
        self.fixed = fixed
        self.splitter = splitter

    def chunk_corpus(self, docs) -> list[vr.Chunk]:
        units: list[vr.Chunk] = []
        for d in docs:
            for ci, chunk_text in enumerate(self.fixed.split(d.text)):
                sents = [sn for sn in self.splitter.split(chunk_text)
                         if any(ch.isalnum() for ch in sn)]
                if not sents:
                    sents = [chunk_text]
                for si, sent in enumerate(sents):
                    units.append(vr.Chunk(
                        chunk_id=f"{d.doc_id}:c{ci}:s{si}",
                        doc_id=d.doc_id,
                        title=d.title,
                        chunk_index=ci * 10000 + si,    # unique, preserves order
                        text=sent,
                    ))
        return units

    def describe(self) -> dict:
        return {
            "chunker": "chunk_sentence",
            **self.fixed.describe(),                 # chunk_size, overlap, unit
            **{f"sent_{k}": v                        # splitter, version, language
               for k, v in self.splitter.describe().items()},
        }


def dump_inspection_files(root: Path, docs: dict, chunkers: dict,
                          seed: int, n: int = 30, per_doc: bool = True) -> None:
    """Writes three inspection artefacts for each chunker kind:

    corpus_docs.jsonl           — cleaned text of every markdown document (one per line)
    units_sample_<kind>.txt     — N random retrieval units (quick sanity check)
    units_per_doc/<kind>/       — one file per document with ALL its units in order
                                  (read these to judge whether the split is sensible before
                                  trusting any metric)
    """
    with open(root / "corpus_docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs.values():
            f.write(json.dumps({"doc_id": d.doc_id, "title": d.title, "text": d.text},
                               ensure_ascii=False) + "\n")

    for kind, chunker in chunkers.items():
        all_chunks = chunker.chunk_corpus(docs.values())

        # ---- random sample ----
        sample = random.Random(seed).sample(all_chunks, min(n, len(all_chunks)))
        with open(root / f"units_sample_{kind}.txt", "w", encoding="utf-8") as f:
            f.write(f"{len(all_chunks)} units total | {len(sample)} random samples "
                    f"(seed {seed})\n\n")
            for c in sample:
                f.write(f"[{c.title}]  ({c.chunk_id})\n{c.text}\n\n{'─'*60}\n\n")

        # ---- per-document files ----
        if per_doc:
            doc_dir = root / "units_per_doc" / kind
            doc_dir.mkdir(parents=True, exist_ok=True)
            by_doc: dict[str, list[vr.Chunk]] = {}
            for c in all_chunks:
                by_doc.setdefault(c.doc_id, []).append(c)
            for did, chunks in by_doc.items():
                title = chunks[0].title
                safe = re.sub(r"[^A-Za-z0-9 _-]+", "", title)[:80].strip().replace(" ", "_")
                fname = doc_dir / f"{safe}__{did[:8]}.txt"
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(f"Document : {title}\n")
                    f.write(f"doc_id   : {did}\n")
                    f.write(f"chunker  : {kind}\n")
                    f.write(f"n_units  : {len(chunks)}\n")
                    f.write("\n")
                    for c in chunks:
                        f.write(f"[{c.chunk_id}]\n{c.text}\n\n")


# ============================================================================
# CLI
# ============================================================================
def add_policy_data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--md-dir", required=True, help="folder with the markdown policy documents")
    g.add_argument("--questions", required=True, help="JSON array / JSONL question file")
    g.add_argument("--limit", type=int, default=None, help="use only N questions")
    g.add_argument("--sample-seed", type=int, default=None,
                   help="if set with --limit, random sample with this seed; else first N")
    g.add_argument("--log-source-chunk", action="store_true",
                   help="also copy each question's full 'chunk' text into its log record")


def add_all_chunking_args(p: argparse.ArgumentParser) -> None:
    vr.add_fixed_chunking_args(p)
    sr.add_sentence_args(p)


def parse_args(argv=None) -> argparse.Namespace:
    p = vr.build_parser("Vector RAG / SentenceRAG, with and without cross-encoder reranking, "
                        "on markdown policy documents", add_all_chunking_args,
                        add_data_args=add_policy_data_args)
    g = p.add_argument_group("systems")
    g.add_argument("--systems", nargs="+", choices=list(SYSTEMS), default=list(SYSTEMS),
                   help="which systems to run (all share every other setting)")
    p.set_defaults(top_k=5, reranker="ms-marco-MiniLM-L-6-v2", prompt_version="v2",
                   max_new_tokens=256)
    args = vr.finalize_args(p, p.parse_args(argv))
    args.dataset, args.data, args.corpus_scope = "policy", args.questions, "pooled"
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = Path(args.output_dir) if args.output_dir else Path("runs") / f"policy_{ts}"
    root.mkdir(parents=True, exist_ok=True)

    examples, docs, info = load_policy(args)
    fixed_chunker = vr.FixedSizeChunker(args.chunk_size, args.chunk_overlap, args.chunk_unit)
    splitter      = sr.SPLITTERS[args.splitter](args.language)
    # vector systems: fixed-size chunks;  sentence systems: same chunks -> pysbd inside each
    chunkers = {
        "fixed":    fixed_chunker,
        "sentence": ChunkSentenceChunker(fixed_chunker, splitter),
    }
    needed = {SYSTEMS[s][0] for s in args.systems}
    dump_inspection_files(root, docs, {k: v for k, v in chunkers.items() if k in needed}, args.seed)
    (root / "data_info.json").write_text(json.dumps(info, indent=2))

    for name in args.systems:
        kind, rerank = SYSTEMS[name]
        a = copy.deepcopy(args)
        a.output_dir = str(root / name)
        a.reranker = args.reranker if rerank else None
        vr.run(a, chunker=chunkers[kind], method=name,
               loader=lambda: (examples, docs), data_info=info)

    write_comparison(root, args.systems)


if __name__ == "__main__":
    main()