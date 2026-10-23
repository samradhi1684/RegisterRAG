#!/usr/bin/env python3
"""
SentenceRAG baseline: deterministic sentence segmentation -> sentence-level dense retrieval
-> (optional) cross-encoder reranking -> generation.

This script only defines the *segmentation* step. Everything else -- corpus construction,
embedding, exact cosine retrieval, reranking, prompt, generators, logging, metrics, seeding --
is the shared driver in `vector_rag.py` (imported, not copied), so SentenceRAG and the
fixed-size-chunk Vector RAG baseline differ in exactly one component, and the two SentenceRAG
modes below differ in exactly one component (the reranker).

Two modes (select with --reranker):
  1. No reranking : the top --top-k sentences by dense similarity go to the generator.
  2. Reranking    : the top --retrieve-k (default 50) sentences by dense similarity are
                    rescored by a cross-encoder; the best --top-k go to the generator.

Reproducibility:
  * Sentence splitting is rule-based (pysbd), needs no model download and is a pure function
    of the text. The splitter name/version, a SHA-256 fingerprint of all sentences, the
    SHA-256 of the data file, the git commit, CLI command, library versions and every seed
    are written to config.json. `--resume` refuses to continue a run whose settings differ.
  * Sampling (--limit/--sample-seed), generation (--seed, temperature 0) and global RNGs
    (python / numpy / torch via --seed) are controlled from the CLI.
  * Retrieval is exact (no ANN) with deterministic tie-breaking. Floating-point results can
    still differ slightly across GPU/CPU types; log hardware when reporting.

Requirements:
  pip install numpy sentence-transformers tiktoken requests openai tqdm pysbd python-dotenv

Examples:
  # mode 1: SentenceRAG, no reranking
  python sentence_rag.py --dataset 2wiki --data 2wiki_dev.json --embedder BAAI/bge-base-en-v1.5 \
      --generator qwen2.5:14b --top-k 10 --limit 500 --sample-seed 42 --seed 42

  # mode 2: identical, plus cross-encoder reranking of the top-50 sentences
  python sentence_rag.py --dataset 2wiki --data 2wiki_dev.json --embedder BAAI/bge-base-en-v1.5 \
      --generator qwen2.5:14b --top-k 10 --limit 500 --sample-seed 42 --seed 42 \
      --reranker bge-reranker-base --retrieve-k 50

Using it from RegisterRAG later:
  from sentence_rag import SentenceChunker, SPLITTERS
  chunks = SentenceChunker(SPLITTERS["pysbd"]()).chunk_corpus(docs.values())   # list[Chunk]
  # then use vector_rag.DenseRetriever / CrossEncoderReranker / build_generator as-is.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Iterable

import vector_rag as vr
from vector_rag import Chunk, Doc

METHOD = "sentence_rag"
DEFAULT_TOP_K = 10      # sentences are short, so the default context holds more of them


# ============================================================================
# Deterministic sentence splitting
# ============================================================================
class SentenceSplitter(ABC):
    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Pure function: the same text always yields the same list of sentences."""

    @abstractmethod
    def describe(self) -> dict:
        """Everything that determines the output (goes into the run signature)."""


class PysbdSplitter(SentenceSplitter):
    """Rule-based segmenter (pysbd). Handles abbreviations such as 'St. Olav' or
    'George H. W. Bush' without any statistical model."""

    def __init__(self, language: str = "en"):
        import pysbd
        self.language, self.version = language, pysbd.__version__
        self._seg = pysbd.Segmenter(language=language, clean=False)

    def split(self, text: str) -> list[str]:
        """Each line is segmented on its own, so line breaks (e.g. markdown fields or list
        items) are always sentence boundaries. Text without line breaks is unaffected."""
        out: list[str] = []
        for line in text.split("\n"):
            if line.strip():
                out.extend(" ".join(s.split()) for s in self._seg.segment(line))
        return out

    def describe(self) -> dict:
        return {"splitter": "pysbd", "splitter_version": self.version, "language": self.language}


SPLITTERS = {"pysbd": PysbdSplitter}


class SentenceChunker:
    """Turns each document into one `Chunk` per sentence (chunk_id = '<doc_id>:s<i>').
    Same interface as vector_rag.FixedSizeChunker, so the driver treats it as a chunker."""

    def __init__(self, splitter: SentenceSplitter):
        self.splitter = splitter

    def sentences(self, text: str) -> list[str]:
        sents = [s for s in self.splitter.split(text) if any(ch.isalnum() for ch in s)]
        return sents or [text]          # never drop a document that has text

    def chunk_corpus(self, docs: Iterable[Doc]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for d in docs:
            for i, sent in enumerate(self.sentences(d.text)):
                chunks.append(Chunk(f"{d.doc_id}:s{i}", d.doc_id, d.title, i, sent))
        return chunks

    def describe(self) -> dict:
        return {"chunker": "sentence", **self.splitter.describe()}


# ============================================================================
# CLI (all non-segmentation options are inherited from vector_rag)
# ============================================================================
def add_sentence_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("sentence splitting (keep fixed across experiments)")
    g.add_argument("--splitter", choices=sorted(SPLITTERS), default="pysbd")
    g.add_argument("--language", default="en", help="splitter language")


def parse_args(argv=None) -> argparse.Namespace:
    p = vr.build_parser("SentenceRAG baseline (optional cross-encoder reranking)",
                        add_sentence_args)
    p.set_defaults(top_k=DEFAULT_TOP_K)
    return vr.finalize_args(p, p.parse_args(argv))


def main(argv=None) -> None:
    args = parse_args(argv)
    chunker = SentenceChunker(SPLITTERS[args.splitter](args.language))
    vr.run(args, chunker=chunker, method=METHOD)


if __name__ == "__main__":
    main()