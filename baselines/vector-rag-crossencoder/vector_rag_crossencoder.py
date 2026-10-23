#!/usr/bin/env python3
"""
Vector RAG baseline (fixed-size chunking -> dense retrieval -> generation).

Datasets : 2WikiMultiHopQA, MuSiQue (answerable)
Embedders: all-MiniLM-L6-v2 | BAAI/bge-base-en-v1.5
Reranker : optional cross-encoder (--reranker). Stage 1 retrieves --retrieve-k (default 50)
           chunks by dense similarity; stage 2 rescored by the cross-encoder; the best
           --top-k go to the generator. Without --reranker the pipeline is the plain baseline.
Generators: qwen2.5:14b (Ollama) | gpt-4o-mini (OpenAI API)

Design rules (so results are comparable across experiments):
  * Chunking depends ONLY on (corpus, chunk_size, chunk_overlap, chunk_unit). It uses a
    fixed tokenizer (tiktoken cl100k_base) that is independent of the embedder, so every
    embedder sees byte-identical chunks. A SHA-256 fingerprint of all chunks is written to
    config.json / summary.json -- compare it across runs to prove chunking was identical.
  * Retrieval is exact (brute-force cosine over L2-normalised vectors), deterministic
    tie-breaking, no ANN approximation.
  * The prompt is a single constant (PROMPT_VERSION) shared by all generators.
  * Generation is greedy (temperature 0) by default.

Extending: add a class implementing `Retriever` (e.g. BM25, hybrid) and register it in
`RETRIEVERS` (rerankers implement `Reranker`); the CLI `--retriever` choice, logging and metrics pick it up automatically.

Requirements:
  pip install numpy sentence-transformers tiktoken requests openai tqdm python-dotenv
  (openai is only needed for gpt-4o-mini; requests only for Ollama; python-dotenv optional,
   loads OPENAI_API_KEY from a .env file in the working directory)

Example:
  python vector_rag.py --dataset 2wiki --data data/2wiki_dev.json \
      --embedder BAAI/bge-base-en-v1.5 --generator qwen2.5:14b \
      --top-k 5 --chunk-size 200 --chunk-overlap 40 --limit 500 --sample-seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import re
import string
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

logger = logging.getLogger("vector_rag")

PROMPT_VERSION = "v1"
SYSTEM_PROMPT = (
    "You are a question answering assistant. Answer the question using only the "
    "provided context. Reply with ONLY the final answer (a short phrase, entity, date, "
    "number, or yes/no) and no explanation."
)


# ============================================================================
# Data structures
# ============================================================================
@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str


@dataclass
class Example:
    qid: str
    question: str
    answers: list[str]          # gold answer + aliases (MuSiQue)
    doc_ids: list[str]          # this question's own paragraphs (used for per_question scope)
    gold_doc_ids: list[str]     # supporting paragraphs (for retrieval metrics only)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str               # f"{doc_id}:{chunk_index}"
    doc_id: str
    title: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class Hit:
    chunk_idx: int              # index into the global chunk list
    score: float


@dataclass(frozen=True)
class Ranked:
    """A candidate chunk with its first-stage and (optional) reranking scores/ranks (1-based)."""
    chunk_idx: int
    dense_rank: int
    dense_score: float
    rerank_rank: Optional[int] = None
    rerank_score: Optional[float] = None


@dataclass
class GenResult:
    text: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency_s: float
    attempts: int = 1
    extra: dict = field(default_factory=dict)


# ============================================================================
# Dataset loading (2Wiki: HotpotQA-style; MuSiQue: paragraphs list)
# ============================================================================
def _norm_ws(s: str) -> str:
    return " ".join(str(s).split())


def make_doc_id(title: str, text: str) -> str:
    return hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()[:16]


def read_json_records(path: str) -> list[dict]:
    """Accepts a JSON array or JSON-lines file."""
    raw = Path(path).read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _register_doc(docs: dict[str, Doc], title: str, text: str) -> Optional[str]:
    title, text = _norm_ws(title), _norm_ws(text)
    if not text:
        return None
    did = make_doc_id(title, text)
    docs.setdefault(did, Doc(did, title, text))
    return did


def parse_2wiki(rec: dict, docs: dict[str, Doc]) -> Example:
    qid = str(rec.get("_id", rec.get("id")))
    doc_ids: list[str] = []
    for title, sents in rec["context"]:
        did = _register_doc(docs, title, " ".join(sents))
        if did is not None:
            doc_ids.append(did)
    gold_titles = {_norm_ws(t) for t, _ in rec.get("supporting_facts", [])}
    gold = [d for d in doc_ids if docs[d].title in gold_titles]
    return Example(qid, _norm_ws(rec["question"]), [str(rec["answer"])], doc_ids, gold)


def parse_musique(rec: dict, docs: dict[str, Doc]) -> Example:
    doc_ids, gold = [], []
    for p in rec["paragraphs"]:
        did = _register_doc(docs, p["title"], p["paragraph_text"])
        if did is None:
            continue
        doc_ids.append(did)
        if p.get("is_supporting"):
            gold.append(did)
    answers = [str(rec["answer"])] + [str(a) for a in rec.get("answer_aliases", []) or []]
    return Example(str(rec["id"]), _norm_ws(rec["question"]), answers, doc_ids, gold)


def load_examples(path: str, dataset: str, limit: Optional[int],
                  sample_seed: Optional[int]) -> tuple[list[Example], dict[str, Doc]]:
    records = read_json_records(path)
    if dataset == "musique":
        n0 = len(records)
        records = [r for r in records if r.get("answerable", True)]
        if len(records) < n0:
            logger.info("MuSiQue: dropped %d unanswerable questions", n0 - len(records))
    if limit is not None and limit < len(records):
        if sample_seed is None:
            records = records[:limit]
        else:
            records = random.Random(sample_seed).sample(records, limit)
    parser = parse_2wiki if dataset == "2wiki" else parse_musique
    docs: dict[str, Doc] = {}
    examples = [parser(r, docs) for r in records]
    qids = [e.qid for e in examples]
    if len(set(qids)) != len(qids):
        raise ValueError("Duplicate question ids in the selected data")
    return examples, docs


# ============================================================================
# Chunking (identical for every experiment)
# ============================================================================
@lru_cache(maxsize=1)
def get_enc():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


class FixedSizeChunker:
    """Sliding window of `chunk_size` units with `overlap` units of overlap.
    Texts with <= chunk_size units are kept verbatim as a single chunk."""

    def __init__(self, chunk_size: int, overlap: int, unit: str = "tokens"):
        if chunk_size <= 0 or not (0 <= overlap < chunk_size):
            raise ValueError("Need chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
        if unit not in ("tokens", "words"):
            raise ValueError("unit must be 'tokens' or 'words'")
        self.chunk_size, self.overlap, self.unit = chunk_size, overlap, unit
        self._enc = get_enc() if unit == "tokens" else None

    def _windows(self, n: int) -> Iterable[tuple[int, int]]:
        stride = self.chunk_size - self.overlap
        start = 0
        while True:
            end = min(start + self.chunk_size, n)
            yield start, end
            if end >= n:
                break
            start += stride

    def split(self, text: str) -> list[str]:
        if self.unit == "words":
            units = text.split()
            if len(units) <= self.chunk_size:
                return [text]
            pieces = [" ".join(units[s:e]) for s, e in self._windows(len(units))]
        else:
            toks = self._enc.encode(text, disallowed_special=())
            if len(toks) <= self.chunk_size:
                return [text]
            pieces = [self._enc.decode(toks[s:e]).strip() for s, e in self._windows(len(toks))]
        return [p for p in pieces if p]

    def chunk_corpus(self, docs: Iterable[Doc]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for d in docs:
            for i, t in enumerate(self.split(d.text)):
                chunks.append(Chunk(f"{d.doc_id}:{i}", d.doc_id, d.title, i, t))
        return chunks


def fingerprint_chunks(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode("utf-8"))
        h.update(b"\0")
        h.update(c.text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


# ============================================================================
# Embedding
# ============================================================================
EMBEDDERS = {
    "all-MiniLM-L6-v2": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "", "doc_prefix": "",
    },
    "BAAI/bge-base-en-v1.5": {
        "hf_name": "BAAI/bge-base-en-v1.5",
        # Official BGE v1.5 retrieval instruction: queries only, never passages.
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
    },
}


class Embedder:
    def __init__(self, key: str, device: Optional[str] = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.key, self.spec, self.batch_size = key, EMBEDDERS[key], batch_size
        self.model = SentenceTransformer(self.spec["hf_name"], device=device)
        self.max_seq_length = int(self.model.max_seq_length)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        p = self.spec["doc_prefix"]
        emb = self.model.encode([p + t for t in texts], batch_size=self.batch_size,
                                normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=True)
        return emb.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        emb = self.model.encode([self.spec["query_prefix"] + query],
                                normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)
        return emb[0].astype(np.float32)

    def count_truncated(self, texts: list[str]) -> tuple[int, int]:
        """(#texts longer than the model's max_seq_length, longest length in wordpieces)."""
        tok, p = self.model.tokenizer, self.spec["doc_prefix"]
        n_trunc, longest = 0, 0
        for i in range(0, len(texts), 1000):
            ids = tok([p + t for t in texts[i:i + 1000]], add_special_tokens=True,
                      truncation=False)["input_ids"]
            lens = [len(x) for x in ids]
            n_trunc += sum(l > self.max_seq_length for l in lens)
            longest = max(longest, max(lens, default=0))
        return n_trunc, longest


# ============================================================================
# Retrieval (plug-in point for BM25 / hybrid)
# ============================================================================
class Retriever(ABC):
    @abstractmethod
    def build(self, chunks: list[Chunk], chunks_fingerprint: str) -> None:
        """Index the chunk list. Chunk order defines `Hit.chunk_idx`."""

    @abstractmethod
    def retrieve(self, query: str, k: int,
                 allowed: Optional[np.ndarray] = None) -> list[Hit]:
        """Top-k hits, best first. `allowed` optionally restricts to these chunk indices."""

    def describe(self) -> dict:
        return {}


class DenseRetriever(Retriever):
    def __init__(self, embedder: Embedder, cache_dir: Optional[str],
                 prepend_title: bool):
        self.embedder, self.prepend_title = embedder, prepend_title
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.emb: Optional[np.ndarray] = None
        self.n_truncated = self.longest = 0

    def _text(self, c: Chunk) -> str:
        return f"{c.title}\n{c.text}" if self.prepend_title else c.text

    def build(self, chunks: list[Chunk], chunks_fingerprint: str) -> None:
        texts = [self._text(c) for c in chunks]
        self.n_truncated, self.longest = self.embedder.count_truncated(texts)
        logger.info("Embedder max_seq_length=%d | longest chunk=%d wordpieces | truncated=%d/%d",
                    self.embedder.max_seq_length, self.longest, self.n_truncated, len(texts))
        cache_path = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha1(
                f"{self.embedder.key}|{self.prepend_title}|{chunks_fingerprint}".encode()
            ).hexdigest()[:16]
            cache_path = self.cache_dir / f"emb_{key}.npy"
            if cache_path.exists():
                emb = np.load(cache_path)
                if emb.shape[0] == len(chunks):
                    logger.info("Loaded cached embeddings: %s", cache_path)
                    self.emb = emb
                    return
        emb = self.embedder.encode_docs(texts)
        if cache_path:
            np.save(cache_path, emb)
        self.emb = emb

    def retrieve(self, query: str, k: int,
                 allowed: Optional[np.ndarray] = None) -> list[Hit]:
        q = self.embedder.encode_query(query)
        if allowed is None:
            idx, scores = None, self.emb @ q
        else:
            if len(allowed) == 0:
                return []
            idx, scores = allowed, self.emb[allowed] @ q
        order = np.argsort(-scores, kind="stable")[:k]      # ties -> lower index first
        return [Hit(int(order_i if idx is None else idx[order_i]), float(scores[order_i]))
                for order_i in order]

    def describe(self) -> dict:
        return {"embedder": self.embedder.key,
                "embedder_max_seq_length": self.embedder.max_seq_length,
                "longest_chunk_wordpieces": self.longest,
                "n_chunks_truncated_by_embedder": self.n_truncated}


def _build_dense(args: argparse.Namespace) -> Retriever:
    embedder = Embedder(args.embedder, device=args.device, batch_size=args.embed_batch_size)
    return DenseRetriever(embedder, args.cache_dir, args.prepend_title)


RETRIEVERS: dict[str, Callable[[argparse.Namespace], Retriever]] = {
    "dense": _build_dense,
    # "bm25": _build_bm25,      # later: implement Retriever, register here
    # "hybrid": _build_hybrid,
}


# ============================================================================
# Reranking (optional second stage)
# ============================================================================
RERANKERS = {
    "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], hits: list[Hit],
               k: int) -> tuple[list[Ranked], list[Ranked]]:
        """`chunks[i]` is the chunk for `hits[i]` (first-stage order, best first).
        Returns (all candidates in first-stage order with rerank scores/ranks,
                 top-k candidates in reranked order)."""

    def describe(self) -> dict:
        return {}


class CrossEncoderReranker(Reranker):
    def __init__(self, key: str, device: Optional[str], max_length: int,
                 batch_size: int, prepend_title: bool):
        from sentence_transformers import CrossEncoder
        self.key, self.max_length = key, max_length
        self.batch_size, self.prepend_title = batch_size, prepend_title
        self.model = CrossEncoder(RERANKERS[key], max_length=max_length, device=device)
        self.n_pairs = self.n_truncated_pairs = self.longest_pair = 0

    def _text(self, c: Chunk) -> str:
        return f"{c.title}\n{c.text}" if self.prepend_title else c.text

    def _track_truncation(self, query: str, texts: list[str]) -> None:
        ids = self.model.tokenizer([query] * len(texts), texts, add_special_tokens=True,
                                   truncation=False)["input_ids"]
        lens = [len(x) for x in ids]
        self.n_pairs += len(lens)
        self.n_truncated_pairs += sum(l > self.max_length for l in lens)
        self.longest_pair = max(self.longest_pair, max(lens, default=0))

    def rerank(self, query, chunks, hits, k):
        if not hits:
            return [], []
        texts = [self._text(c) for c in chunks]
        self._track_truncation(query, texts)
        raw = self.model.predict([(query, t) for t in texts], batch_size=self.batch_size,
                                 show_progress_bar=False)
        scores = [float(x) for x in np.asarray(raw).reshape(-1)]
        order = np.argsort(-np.asarray(scores), kind="stable")   # ties -> first-stage order
        rr_rank = {int(i): r for r, i in enumerate(order, 1)}
        cands = [Ranked(h.chunk_idx, i + 1, h.score, rr_rank[i], scores[i])
                 for i, h in enumerate(hits)]
        top = [cands[int(i)] for i in order[:k]]
        return cands, top

    def describe(self) -> dict:
        return {"reranker": self.key, "reranker_max_length": self.max_length,
                "rerank_pairs_scored": self.n_pairs,
                "rerank_pairs_truncated": self.n_truncated_pairs,
                "rerank_longest_pair_tokens": self.longest_pair}


def build_reranker(args: argparse.Namespace) -> Optional[Reranker]:
    if not args.reranker:
        return None
    return CrossEncoderReranker(args.reranker, args.device, args.rerank_max_length,
                                args.rerank_batch_size, args.rerank_prepend_title)


# ============================================================================
# Generation
# ============================================================================
def call_with_retries(fn, attempts: int = 5, base_delay: float = 2.0):
    """Returns (result, latency_of_successful_call_s, n_attempts)."""
    for a in range(1, attempts + 1):
        try:
            t0 = time.perf_counter()
            out = fn()
            return out, time.perf_counter() - t0, a
        except Exception as e:  # noqa: BLE001
            if a == attempts:
                raise
            delay = base_delay * 2 ** (a - 1)
            logger.warning("Generator call failed (%s); retry %d/%d in %.0fs",
                           e, a, attempts - 1, delay)
            time.sleep(delay)


class Generator(ABC):
    context_limit: Optional[int] = None

    def warmup(self) -> None:
        pass

    @abstractmethod
    def generate(self, system: str, user: str) -> GenResult: ...


class OllamaGenerator(Generator):
    def __init__(self, model: str, base_url: str, temperature: float, max_new_tokens: int,
                 seed: int, num_ctx: int, timeout: float, keep_alive: str = "30m"):
        import requests
        self.session = requests.Session()
        self.model, self.url = model, base_url.rstrip("/") + "/api/chat"
        self.temperature, self.max_new_tokens, self.seed = temperature, max_new_tokens, seed
        self.num_ctx, self.timeout, self.keep_alive = num_ctx, timeout, keep_alive
        self.context_limit = num_ctx

    def _payload(self, system: str, user: str, num_predict: int) -> dict:
        return {
            "model": self.model, "stream": False, "keep_alive": self.keep_alive,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "num_predict": num_predict, "num_ctx": self.num_ctx},
        }

    def _post(self, payload: dict) -> dict:
        r = self.session.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def warmup(self) -> None:
        logger.info("Warming up Ollama model %s (load time excluded from latency)", self.model)
        self._post(self._payload("You are helpful.", "Say OK.", 1))

    def generate(self, system: str, user: str) -> GenResult:
        data, lat, att = call_with_retries(
            lambda: self._post(self._payload(system, user, self.max_new_tokens)))
        ns = 1e9
        extra = {k: data[k] / ns for k in ("total_duration", "load_duration",
                                           "prompt_eval_duration", "eval_duration")
                 if isinstance(data.get(k), (int, float))}
        extra["done_reason"] = data.get("done_reason")
        return GenResult(data["message"]["content"], data.get("prompt_eval_count"),
                         data.get("eval_count"), lat, att, extra)


class OpenAIGenerator(Generator):
    def __init__(self, model: str, temperature: float, max_new_tokens: int,
                 seed: int, timeout: float):
        import os
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set. Put it in a .env file "
                             "(OPENAI_API_KEY=sk-...) or export it in your shell.")
        self.client = OpenAI(max_retries=0, timeout=timeout)   # retries handled by us; key from env
        self.model, self.temperature = model, temperature
        self.max_new_tokens, self.seed = max_new_tokens, seed

    def generate(self, system: str, user: str) -> GenResult:
        resp, lat, att = call_with_retries(lambda: self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_new_tokens,
            seed=self.seed,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}]))
        u = resp.usage
        return GenResult(resp.choices[0].message.content or "",
                         getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None),
                         lat, att,
                         {"system_fingerprint": getattr(resp, "system_fingerprint", None),
                          "finish_reason": resp.choices[0].finish_reason})


GENERATORS = {"qwen2.5:14b": "ollama", "gpt-4o-mini": "openai"}


def build_generator(args: argparse.Namespace) -> Generator:
    if GENERATORS[args.generator] == "ollama":
        return OllamaGenerator(args.generator, args.ollama_url, args.temperature,
                               args.max_new_tokens, args.seed, args.num_ctx, args.timeout)
    return OpenAIGenerator(args.generator, args.temperature, args.max_new_tokens,
                           args.seed, args.timeout)


def build_prompt(question: str, chunks: list[Chunk]) -> tuple[str, str]:
    ctx = "\n\n".join(f"[{i}] {c.title}\n{c.text}" for i, c in enumerate(chunks, 1))
    return SYSTEM_PROMPT, f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"


# ============================================================================
# Metrics (HotpotQA / SQuAD-style EM & F1; doc-level retrieval recall)
# ============================================================================
_PUNCT = set(string.punctuation)
_YN = {"yes", "no", "noanswer"}


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if (p in _YN or g in _YN) and p != g:          # as in the official HotpotQA script
        return 0.0
    pt, gt = p.split(), g.split()
    common = sum((Counter(pt) & Counter(gt)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(pt), common / len(gt)
    return 2 * prec * rec / (prec + rec)


def qa_metrics(pred: str, answers: list[str]) -> tuple[float, float]:
    em = max(float(normalize_answer(pred) == normalize_answer(a)) for a in answers)
    f1 = max(f1_score(pred, a) for a in answers)
    return em, f1


def retrieval_metrics(retrieved_doc_ids: list[str], gold: list[str]):
    g = set(gold)
    if not g:
        return None, None
    hit = len(g & set(retrieved_doc_ids))
    return hit / len(g), float(hit == len(g))


# ============================================================================
# Pipeline
# ============================================================================
def summarize(results_path: Path) -> dict:
    recs = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else None

    lat = lambda key: [r["latency"].get(key) for r in recs if r["error"] is None]  # noqa: E731
    return {
        "n_questions": len(recs),
        "n_generation_failures": sum(r["error"] is not None for r in recs),
        "EM": mean([r["metrics"]["em"] for r in recs]),
        "F1": mean([r["metrics"]["f1"] for r in recs]),
        "retrieval_doc_recall@k": mean([r["metrics"]["doc_recall"] for r in recs]),
        "retrieval_all_gold_hit@k": mean([r["metrics"]["all_gold_hit"] for r in recs]),
        "stage1_doc_recall@retrieve_k": mean([r["metrics"].get("stage1_doc_recall") for r in recs]),
        "stage1_all_gold_hit@retrieve_k": mean([r["metrics"].get("stage1_all_gold_hit") for r in recs]),
        "latency_s_mean": {k: mean(lat(k)) for k in ("retrieval_s", "rerank_s", "generation_s", "total_s")},
        "latency_s_median": {k: med(lat(k)) for k in ("retrieval_s", "rerank_s", "generation_s", "total_s")},
        "tokens_total": {
            "prompt": sum(r["usage"]["prompt_tokens"] or 0 for r in recs),
            "completion": sum(r["usage"]["completion_tokens"] or 0 for r in recs),
            "prompt_est_cl100k": sum(r["usage"]["prompt_tokens_est_cl100k"] for r in recs),
        },
    }


def _version(mod: str) -> Optional[str]:
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return None


def _load_env() -> None:
    """Load variables (e.g. OPENAI_API_KEY) from a .env file if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def run(args: argparse.Namespace) -> None:
    _load_env()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        rr = f"__rr-{args.reranker}-from{args.retrieve_k}" if args.reranker else ""
        tag = f"{args.dataset}__{args.embedder}{rr}__{args.generator}__k{args.top_k}__{ts}"
        out_dir = Path("runs") / re.sub(r"[^A-Za-z0-9._-]+", "_", tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path, config_path = out_dir / "results.jsonl", out_dir / "config.json"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stderr),
                                  logging.FileHandler(out_dir / "run.log")])
    if results_path.exists() and results_path.stat().st_size > 0 and not args.resume:
        raise SystemExit(f"{results_path} already has results; use --resume or a new --output-dir")

    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- corpus + chunks (embedder-independent) ---
    examples, docs = load_examples(args.data, args.dataset, args.limit, args.sample_seed)
    chunker = FixedSizeChunker(args.chunk_size, args.chunk_overlap, args.chunk_unit)
    chunks = chunker.chunk_corpus(docs.values())
    fingerprint = fingerprint_chunks(chunks)
    doc2chunks: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        doc2chunks[c.doc_id].append(i)
    logger.info("%d questions | %d unique docs | %d chunks | chunk fingerprint %s",
                len(examples), len(docs), len(chunks), fingerprint[:16])

    signature = {
        "dataset": args.dataset, "data": str(args.data), "limit": args.limit,
        "sample_seed": args.sample_seed, "corpus_scope": args.corpus_scope,
        "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap,
        "chunk_unit": args.chunk_unit, "chunks_fingerprint": fingerprint,
        "retriever": args.retriever, "embedder": args.embedder,
        "prepend_title": args.prepend_title, "top_k": args.top_k,
        "reranker": args.reranker,
        "retrieve_k": args.retrieve_k if args.reranker else None,
        "rerank_max_length": args.rerank_max_length if args.reranker else None,
        "rerank_prepend_title": args.rerank_prepend_title if args.reranker else None,
        "generator": args.generator, "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens, "seed": args.seed,
        "num_ctx": args.num_ctx if GENERATORS[args.generator] == "ollama" else None,
        "prompt_version": PROMPT_VERSION,
    }
    if config_path.exists():
        old = json.loads(config_path.read_text())["signature"]
        if old != signature:
            diff = {k: (old.get(k), signature.get(k)) for k in signature if old.get(k) != signature.get(k)}
            raise SystemExit(f"Config differs from existing run in {out_dir}: {diff}")

    # --- retriever / generator ---
    retriever = RETRIEVERS[args.retriever](args)
    retriever.build(chunks, fingerprint)
    reranker = build_reranker(args)
    generator = build_generator(args)

    config_path.write_text(json.dumps({
        "signature": signature, "args": vars(args), "system_prompt": SYSTEM_PROMPT,
        "n_docs": len(docs), "n_chunks": len(chunks), "retriever_info": retriever.describe(),
        "env": {"python": platform.python_version(), "numpy": _version("numpy"),
                "torch": _version("torch"), "sentence_transformers": _version("sentence_transformers"),
                "tiktoken": _version("tiktoken"), "openai": _version("openai")},
        "started_utc": ts,
    }, indent=2, default=str))

    done = set()
    if results_path.exists():
        done = {json.loads(l)["qid"] for l in results_path.read_text().splitlines() if l.strip()}
        logger.info("Resuming: %d questions already done", len(done))
    todo = [e for e in examples if e.qid not in done]
    if todo:
        generator.warmup()

    from tqdm import tqdm
    enc = get_enc()
    with open(results_path, "a", encoding="utf-8") as fout:
        for ex in tqdm(todo, desc="RAG", unit="q"):
            # ---- retrieval ----
            t0 = time.perf_counter()
            allowed = None
            if args.corpus_scope == "per_question":
                allowed = np.array(sorted(i for d in ex.doc_ids for i in doc2chunks[d]), dtype=np.int64)
            first_k = args.retrieve_k if reranker else args.top_k
            hits = retriever.retrieve(ex.question, first_k, allowed)
            t_ret = time.perf_counter() - t0

            # ---- optional reranking ----
            t_rr = None
            if reranker:
                t1 = time.perf_counter()
                cands, final = reranker.rerank(ex.question, [chunks[h.chunk_idx] for h in hits],
                                               hits, args.top_k)
                t_rr = time.perf_counter() - t1
            else:
                cands = [Ranked(h.chunk_idx, r, h.score) for r, h in enumerate(hits, 1)]
                final = cands[:args.top_k]
            got = [chunks[x.chunk_idx] for x in final]

            # ---- generation ----
            system, user = build_prompt(ex.question, got)
            prompt_est = len(enc.encode(system + "\n" + user, disallowed_special=()))
            if generator.context_limit and prompt_est > 0.85 * generator.context_limit:
                logger.warning("qid=%s prompt (~%d tok) is near/over num_ctx=%d; raise --num-ctx",
                               ex.qid, prompt_est, generator.context_limit)
            res, err = None, None
            try:
                res = generator.generate(system, user)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                logger.error("qid=%s generation failed: %s", ex.qid, err)
            pred = res.text.strip() if res else ""
            em, f1 = qa_metrics(pred, ex.answers) if res else (0.0, 0.0)
            recall, all_hit = retrieval_metrics([c.doc_id for c in got], ex.gold_doc_ids)
            s1_recall, s1_all = (retrieval_metrics([chunks[x.chunk_idx].doc_id for x in cands],
                                                   ex.gold_doc_ids) if reranker else (None, None))
            t_gen = res.latency_s if res else None
            t_total = t_ret + (t_rr or 0.0) + t_gen if t_gen is not None else None

            rec = {
                "qid": ex.qid, "dataset": args.dataset, "question": ex.question,
                "gold_answers": ex.answers, "prediction": pred, "error": err,
                # chunks actually given to the generator, in prompt order
                "retrieved": [{"rank": r, "chunk_id": chunks[x.chunk_idx].chunk_id,
                               "doc_id": chunks[x.chunk_idx].doc_id,
                               "title": chunks[x.chunk_idx].title,
                               "chunk_index": chunks[x.chunk_idx].chunk_index,
                               "dense_rank": x.dense_rank, "retrieval_score": x.dense_score,
                               "rerank_rank": x.rerank_rank, "rerank_score": x.rerank_score,
                               "is_gold_doc": chunks[x.chunk_idx].doc_id in ex.gold_doc_ids,
                               "text": chunks[x.chunk_idx].text}
                              for r, x in enumerate(final, 1)],
                # all first-stage candidates (compact, no text); only when reranking
                "candidates": ([{"chunk_id": chunks[x.chunk_idx].chunk_id,
                                 "doc_id": chunks[x.chunk_idx].doc_id,
                                 "dense_rank": x.dense_rank, "retrieval_score": x.dense_score,
                                 "rerank_rank": x.rerank_rank, "rerank_score": x.rerank_score,
                                 "is_gold_doc": chunks[x.chunk_idx].doc_id in ex.gold_doc_ids}
                                for x in cands] if reranker else None),
                "latency": {"retrieval_s": t_ret, "rerank_s": t_rr,
                            "generation_s": t_gen, "total_s": t_total},
                "usage": {"prompt_tokens": res.prompt_tokens if res else None,
                          "completion_tokens": res.completion_tokens if res else None,
                          "prompt_tokens_est_cl100k": prompt_est},
                "generation_extra": ({"attempts": res.attempts, **res.extra} if res else None),
                "metrics": {"em": em, "f1": f1, "doc_recall": recall, "all_gold_hit": all_hit,
                            "stage1_doc_recall": s1_recall, "stage1_all_gold_hit": s1_all},
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    summary = {"signature": signature, **summarize(results_path),
               "reranker_info": reranker.describe() if reranker else None}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Summary:\n%s", json.dumps({k: v for k, v in summary.items() if k != "signature"}, indent=2))
    logger.info("Outputs in %s", out_dir)


# ============================================================================
# CLI
# ============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vector RAG baseline",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--dataset", required=True, choices=["2wiki", "musique"])
    g.add_argument("--data", required=True, help="JSON array or JSONL file")
    g.add_argument("--limit", type=int, default=None, help="use only N questions")
    g.add_argument("--sample-seed", type=int, default=None,
                   help="if set with --limit, random sample with this seed; else first N")
    g.add_argument("--corpus-scope", choices=["pooled", "per_question"], default="pooled",
                   help="pooled: search all selected questions' paragraphs (open-domain); "
                        "per_question: search only that question's own paragraphs (distractor)")

    g = p.add_argument_group("chunking (keep fixed across experiments)")
    g.add_argument("--chunk-size", type=int, default=200)
    g.add_argument("--chunk-overlap", type=int, default=40)
    g.add_argument("--chunk-unit", choices=["tokens", "words"], default="tokens",
                   help="tokens = tiktoken cl100k_base (embedder-independent)")

    g = p.add_argument_group("retrieval")
    g.add_argument("--retriever", choices=sorted(RETRIEVERS), default="dense")
    g.add_argument("--embedder", choices=sorted(EMBEDDERS), default="all-MiniLM-L6-v2")
    g.add_argument("--top-k", type=int, default=5,
                   help="chunks passed to the generator (final context size)")
    g.add_argument("--prepend-title", action="store_true",
                   help="embed '<title>\\n<chunk>' instead of the chunk text alone")
    g.add_argument("--embed-batch-size", type=int, default=64)
    g.add_argument("--device", default=None, help="cuda / cpu / mps (default: auto)")
    g.add_argument("--cache-dir", default=".cache/embeddings",
                   help="embedding cache dir ('' to disable)")

    g = p.add_argument_group("reranking (optional cross-encoder second stage)")
    g.add_argument("--reranker", choices=sorted(RERANKERS), default=None,
                   help="enable cross-encoder reranking of the first-stage candidates")
    g.add_argument("--retrieve-k", type=int, default=50,
                   help="first-stage depth when reranking (ignored without --reranker)")
    g.add_argument("--rerank-max-length", type=int, default=512)
    g.add_argument("--rerank-batch-size", type=int, default=32)
    g.add_argument("--rerank-prepend-title", action="store_true",
                   help="score '<title>\\n<chunk>' instead of the chunk text alone")

    g = p.add_argument_group("generation")
    g.add_argument("--generator", choices=sorted(GENERATORS), default="qwen2.5:14b")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--max-new-tokens", type=int, default=64)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--ollama-url", default="http://localhost:11434")
    g.add_argument("--num-ctx", type=int, default=8192,
                   help="Ollama context window (its default is small and silently truncates)")
    g.add_argument("--timeout", type=float, default=300.0)

    g = p.add_argument_group("output")
    g.add_argument("--output-dir", default=None, help="default: runs/<auto-name>")
    g.add_argument("--resume", action="store_true", help="skip questions already in results.jsonl")

    args = p.parse_args(argv)
    if args.top_k <= 0:
        p.error("--top-k must be > 0")
    if args.reranker and args.retrieve_k < args.top_k:
        p.error("--retrieve-k must be >= --top-k when --reranker is set")
    return args


if __name__ == "__main__":
    run(parse_args())