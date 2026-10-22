from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import asdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .distillation import EvidenceDistillationConfig, EvidenceDistiller
from .llm import PROVIDERS
from .distilled_prior import (
    SCORING_METHODS,
    call_llm,
    read_api_key,
    write_jsonl,
    _method_rule,
    _context_pubmed_query,
)
from .evidence import EvidenceSelectionConfig, select_evidence_documents
from .models import EvidenceDocument
from .retrieval import JsonlRetrievalCache, PubMedClient, _gene_pubmed_query, parse_exclude_pmids



MODULE_SCORE_COLUMNS = [
    "candidate_id",
    "source_module",
    "target_module",
    "method",
    "fatereasoner_score",
    "evidence_level",
    "supporting_pmids",
    "rationale",
    "caveats",
    "module_evidence_scope",
    "context_match",
    "evidence_strength",
    "source_evidence_genes",
    "target_evidence_genes",
]


def add_score_grn_modules_parser(subparsers: argparse._SubParsersAction) -> None:
    module_parser = subparsers.add_parser(
        "score-grn-modules",
        help="Score GRN module -> module candidates with distilled evidence plus LLM prior.",
    )
    module_parser.add_argument("--network", required=True, help="Module-annotated GRN edge table (generic columns or IReNA output).")
    module_parser.add_argument("--species", required=True, help="Species label, e.g. mouse, zebrafish, human.")
    module_parser.add_argument("--context", required=True, help="Biological context text or path to a text file.")
    module_parser.add_argument("--out", required=True, help="Output directory.")
    module_parser.add_argument("--top-module-pairs", type=int, default=20)
    module_parser.add_argument(
        "--per-gene-retmax",
        type=int,
        default=8,
        help="Maximum PubMed records retrieved for each individual module gene.",
    )
    module_parser.add_argument(
        "--per-pair-retmax",
        type=int,
        default=3,
        help="Maximum PubMed records retrieved for every directed GRN gene pair in a module pair.",
    )
    module_parser.add_argument("--module-gene-block-size", type=int, default=8)
    module_parser.add_argument("--module-block-retmax", type=int, default=35)
    module_parser.add_argument("--module-evidence-min-docs", type=int, default=20)
    module_parser.add_argument("--module-evidence-max-docs", type=int, default=30)
    module_parser.add_argument("--method", choices=SCORING_METHODS, default="distilled_plus_prior")
    module_parser.add_argument("--retmax", type=int, default=30)
    module_parser.add_argument("--date-cutoff", default=None)
    module_parser.add_argument("--delay-seconds", type=float, default=0.34)
    module_parser.add_argument("--retrieval-timeout", type=int, default=12, help="Per-request PubMed timeout in seconds.")
    module_parser.add_argument(
        "--exclude-pmids",
        default="",
        help="Comma/semicolon separated PubMed IDs to exclude from retrieval and evidence packets.",
    )
    module_parser.add_argument("--evidence-threshold", type=float, default=0.25)
    module_parser.add_argument("--evidence-min-keep", type=int, default=5)
    module_parser.add_argument("--evidence-max-keep", type=int, default=20)
    module_parser.add_argument(
        "--distill-evidence",
        choices=["lite_weightfree", "lite_weighted", "lite", "medcpt_go", "biomedbert_rerank"],
        default="lite_weightfree",
    )
    module_parser.add_argument("--distilled-max-chars", type=int, default=3600)
    module_parser.add_argument("--distilled-packet-style", choices=["audit", "concise"], default="concise")
    module_parser.add_argument("--distill-max-windows", type=int, default=8)
    module_parser.add_argument("--distilled-anchor-docs", type=int, default=3)
    module_parser.add_argument("--distilled-anchor-chars", type=int, default=650)
    module_parser.add_argument("--anchor-strategy", choices=["balanced", "diverse_preserve"], default="diverse_preserve")
    module_parser.add_argument("--include-go", dest="include_go", action="store_true", default=False)
    module_parser.add_argument("--no-go", dest="include_go", action="store_false", help=argparse.SUPPRESS)
    module_parser.add_argument("--medcpt-model", default="ncbi/MedCPT-Cross-Encoder")
    module_parser.add_argument(
        "--biomedbert-model",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        help="HuggingFace model name or local path used when --distill-evidence biomedbert_rerank.",
    )
    module_parser.add_argument("--provider", choices=PROVIDERS, default=None, help="LLM provider (required unless --prepare-only).")
    module_parser.add_argument("--model", default=None, help="Provider model ID; defaults to the provider preset.")
    module_parser.add_argument("--api-key", default=None)
    module_parser.add_argument("--api-key-file", default=None)
    module_parser.add_argument("--base-url", default=None)
    module_parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    module_parser.add_argument("--temperature", type=float, default=0.0)
    module_parser.add_argument("--max-output-tokens", type=int, default=8000)
    module_parser.add_argument("--request-batch-size", type=int, default=1)
    module_parser.add_argument("--timeout", type=float, default=120.0)
    module_parser.add_argument("--max-retries", type=int, default=3)
    module_parser.add_argument("--prepare-only", action="store_true")
    module_parser.add_argument("--resume", action="store_true")
    module_parser.set_defaults(func=run_score_grn_modules)


def run_score_grn_modules(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = read_text_arg(args.context)
    edges = load_network(args.network, args.species)
    candidates = build_module_pair_candidates(edges).head(args.top_module_pairs).copy()
    candidates.to_csv(out_dir / "module_pair_candidates.csv", index=False)
    (out_dir / "context.txt").write_text(context + "\n", encoding="utf-8")
    packets = build_module_distilled_packets(candidates, context, args, out_dir)
    packets_path = out_dir / f"{args.method}_module_packets.jsonl"
    write_jsonl(packets_path, packets)
    packets_to_table(packets).to_csv(out_dir / "distilled_module_evidence_packets.csv", index=False)
    module_gene_pair_evidence_to_table(packets).to_csv(out_dir / "module_gene_pair_regulation_evidence.csv", index=False)
    if args.prepare_only:
        print(f"module_pairs={len(candidates)} packets={packets_path}")
        print(f"prepare_only=true out={out_dir}")
        return
    score_path = out_dir / f"{args.method}_module_scores.csv"
    scores = score_module_packets_with_llm(packets, args, score_path)
    merged = merge_module_scores(candidates, scores)
    merged.to_csv(out_dir / "module_pair_scores.csv", index=False)
    print(f"module_pairs={len(candidates)} scores={score_path}")
    print(f"result={out_dir / 'module_pair_scores.csv'}")


GENERIC_NETWORK_COLUMNS = ["regulator_symbol", "source_module", "target_symbol", "target_module"]


def load_network(path: str | Path, species: str) -> pd.DataFrame:
    """Load a module-annotated GRN edge table.

    Two layouts are accepted:

    * Generic: a table (TSV or CSV) with columns ``regulator_symbol``, ``source_module``,
      ``target_symbol`` and ``target_module``; ``expression_correlation``,
      ``regulation_type`` and ``motif_loci`` are optional. ``species`` is a free label
      (e.g. "mouse", "human") used to focus PubMed retrieval.
    * IReNA output for mouse or zebrafish, recognised by its original column names.
    """
    sep = "," if str(path).lower().endswith(".csv") else "\t"
    raw = pd.read_csv(path, sep=sep)
    if all(column in raw.columns for column in GENERIC_NETWORK_COLUMNS):
        out = raw.copy()
        out["regulator_id"] = out.get("regulator_id", out["regulator_symbol"])
        out["target_id"] = out.get("target_id", out["target_symbol"])
        for column, default in [("expression_correlation", 0.0), ("regulation_type", ""), ("motif_loci", "")]:
            if column not in out.columns:
                out[column] = default
        keep = ["regulator_id", "regulator_symbol", "source_module", "target_id", "target_symbol",
                "target_module", "expression_correlation", "regulation_type", "motif_loci"]
        return _finalize_network(out[keep].copy(), species)
    if species == "mouse":
        mapping = {
            "TF": "regulator_id",
            "TFSymbol": "regulator_symbol",
            "TFGroup": "source_module",
            "Target": "target_id",
            "TargetSymbol": "target_symbol",
            "TargetGroup": "target_module",
            "Expression.correlation": "expression_correlation",
            "Type.of.regulation": "regulation_type",
            "GRCm38.loci.of.regulation..Motif.ID.Type.of.locus.Chr.Start.End.FOS": "motif_loci",
        }
    elif species == "zebrafish":
        mapping = {
            "Regulator.ID": "regulator_id",
            "Regulator.symbol": "regulator_symbol",
            "Regulator.module": "source_module",
            "Target.ID": "target_id",
            "Target.symbol": "target_symbol",
            "Target.module": "target_module",
            "Expression.correlation": "expression_correlation",
            "Type.of.regulation": "regulation_type",
            "GRCz10.loci.of.regulation: Motif.ID,Type.of.locus,Chr,Start,End,FOS": "motif_loci",
        }
    else:
        raise ValueError(
            f"Network columns not recognised. Provide the generic columns {GENERIC_NETWORK_COLUMNS} "
            "or IReNA output with species='mouse' or 'zebrafish'."
        )
    missing = [column for column in mapping if column not in raw.columns]
    if missing:
        raise ValueError(f"{species} network missing columns: {missing}")
    out = raw[list(mapping)].rename(columns=mapping).copy()
    return _finalize_network(out, species)


def _finalize_network(out: pd.DataFrame, species: str) -> pd.DataFrame:
    out.insert(0, "species", species)
    out["expression_correlation"] = pd.to_numeric(out["expression_correlation"], errors="coerce").fillna(0.0)
    for column in [
        "regulator_id",
        "regulator_symbol",
        "source_module",
        "target_id",
        "target_symbol",
        "target_module",
        "regulation_type",
        "motif_loci",
    ]:
        out[column] = out[column].fillna("").astype(str).str.strip().str.strip('"')
    locus_stats = out["motif_loci"].map(parse_motif_loci)
    out["motif_locus_count"] = [item["motif_locus_count"] for item in locus_stats]
    out["promoter_locus_count"] = [item["promoter_locus_count"] for item in locus_stats]
    out["abs_correlation"] = out["expression_correlation"].abs()
    return out


def parse_motif_loci(value: str) -> dict[str, int]:
    records = [record.strip() for record in str(value).strip().strip('"').split(";") if record.strip()]
    promoter = 0
    for record in records:
        parts = [part.strip() for part in record.split(",")]
        if len(parts) > 1 and parts[1] == "promoter":
            promoter += 1
    return {"motif_locus_count": len(records), "promoter_locus_count": promoter}


def build_module_pair_candidates(edges: pd.DataFrame) -> pd.DataFrame:
    source_sizes = edges.groupby("source_module")["regulator_symbol"].nunique()
    target_sizes = edges.groupby("target_module")["target_symbol"].nunique()
    module_gene_sets: dict[str, set[str]] = {}
    for module, group in edges.groupby("source_module"):
        module_gene_sets.setdefault(str(module), set()).update(group["regulator_symbol"].dropna().astype(str))
    for module, group in edges.groupby("target_module"):
        module_gene_sets.setdefault(str(module), set()).update(group["target_symbol"].dropna().astype(str))
    pair_strength = edges.groupby(["source_module", "target_module"])["abs_correlation"].sum()
    source_total = pair_strength.groupby(level=0).sum()
    target_total = pair_strength.groupby(level=1).sum()
    rows = []
    for (species, source_module, target_module), group in edges.groupby(
        ["species", "source_module", "target_module"],
        sort=False,
    ):
        strength = float(group["abs_correlation"].sum())
        n_source = int(source_sizes.get(source_module, group["regulator_symbol"].nunique()))
        n_target = int(target_sizes.get(target_module, group["target_symbol"].nunique()))
        candidate_id = f"{species}:{source_module}->{target_module}"
        rows.append(
            {
                "species": species,
                "candidate_id": candidate_id,
                "source_module": source_module,
                "target_module": target_module,
                "source_module_genes": ";".join(sorted(module_gene_sets.get(str(source_module), set()))),
                "target_module_genes": ";".join(sorted(module_gene_sets.get(str(target_module), set()))),
                "source_regulators": ";".join(sorted(group["regulator_symbol"].unique())),
                "target_genes": ";".join(sorted(group["target_symbol"].unique())),
                "cross_module_edges": ";".join(
                    f"{source}->{target}"
                    for source, target in group[["regulator_symbol", "target_symbol"]]
                    .drop_duplicates()
                    .sort_values(["regulator_symbol", "target_symbol"])
                    .itertuples(index=False, name=None)
                ),
                "top_regulators": _top_symbols(group, "regulator_symbol", "abs_correlation", 12),
                "top_targets": _top_symbols(group, "target_symbol", "abs_correlation", 12),
                "top_edges": _top_edges(group, 10),
                "n_edges": int(len(group)),
                "n_regulators": int(group["regulator_symbol"].nunique()),
                "n_targets": int(group["target_symbol"].nunique()),
                "mean_abs_correlation": float(group["abs_correlation"].mean()),
                "sum_abs_correlation": strength,
                "mean_signed_correlation": float(group["expression_correlation"].mean()),
                "edge_density": _safe_div(len(group), n_source * n_target),
                "source_specificity": _safe_div(strength, float(source_total.get(source_module, 0.0))),
                "target_specificity": _safe_div(strength, float(target_total.get(target_module, 0.0))),
                "motif_locus_count": int(group["motif_locus_count"].sum()),
                "promoter_locus_fraction": _safe_div(group["promoter_locus_count"].sum(), group["motif_locus_count"].sum()),
                "regulation_type_summary": _value_summary(group["regulation_type"]),
            }
        )
    out = pd.DataFrame(rows)
    out["data_score"] = (
        0.30 * _minmax(out["sum_abs_correlation"])
        + 0.20 * _minmax(out["edge_density"])
        + 0.20 * _minmax(out["source_specificity"])
        + 0.15 * _minmax(out["mean_abs_correlation"])
        + 0.15 * _minmax(out["motif_locus_count"])
    )
    return out.sort_values(["data_score", "sum_abs_correlation", "candidate_id"], ascending=[False, False, True]).reset_index(drop=True)


def build_module_distilled_packets(
    candidates: pd.DataFrame,
    context: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict[str, Any]]:
    packet_cache_path = out_dir / f"{args.method}_module_packets.jsonl"
    cached_packets: dict[str, dict[str, Any]] = {}
    if args.resume and packet_cache_path.exists():
        for line in packet_cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                packet = json.loads(line)
                cached_packets[str(packet.get("candidate_id", "")).upper()] = packet
    pubmed = ModulePubMedClient(
        retmax=args.retmax,
        per_gene_retmax=args.per_gene_retmax,
        gene_cache=JsonlRetrievalCache(out_dir / "module_gene_retrieval_cache.jsonl"),
        block_cache=JsonlRetrievalCache(out_dir / "module_block_retrieval_cache.jsonl"),
        block_size=args.module_gene_block_size,
        block_retmax=args.module_block_retmax,
        timeout=args.retrieval_timeout,
        date_cutoff=args.date_cutoff,
        delay_seconds=args.delay_seconds,
        exclude_pmids=parse_exclude_pmids(getattr(args, "exclude_pmids", "")),
    )
    pubmed.context = context
    cache = JsonlRetrievalCache(out_dir / "module_retrieval_cache.jsonl")
    evidence_config = EvidenceSelectionConfig(
        threshold=args.evidence_threshold,
        min_keep=args.module_evidence_min_docs,
        max_keep=args.module_evidence_max_docs,
    )
    distiller = EvidenceDistiller(
        EvidenceDistillationConfig(
            method=args.distill_evidence,
            max_chars=args.distilled_max_chars,
            medcpt_model=args.medcpt_model,
            biomedbert_model=args.biomedbert_model,
            max_windows=args.distill_max_windows,
            raw_anchor_max_docs=args.distilled_anchor_docs,
            raw_anchor_max_chars=args.distilled_anchor_chars,
            anchor_strategy=getattr(args, "anchor_strategy", "balanced"),
            packet_style=args.distilled_packet_style,
            include_go_terms=args.include_go,
        )
    )
    packets = []
    for idx, row in enumerate(candidates.to_dict("records"), start=1):
        candidate_id = str(row["candidate_id"])
        cached_packet = cached_packets.get(candidate_id.upper())
        if cached_packet is not None:
            packets.append(cached_packet)
            print(f"prepared {idx}/{len(candidates)} module_pair={candidate_id} cached=true", flush=True)
            continue
        docs = cache.get(candidate_id)
        if docs is None:
            gene_docs = pubmed.search_and_fetch_module(row, context)
            block_docs = pubmed.search_and_fetch_module_blocks(row, context)
            docs_by_pmid = {doc.pmid: doc for doc in gene_docs + block_docs}
            docs = list(docs_by_pmid.values())
            cache.set(candidate_id, docs)
        else:
            docs = [doc for doc in docs if doc.pmid not in pubmed.exclude_pmids]
        relation_context = module_relation_context(row, context)
        source_module_genes = _split_semicolon(row.get("source_module_genes", ""))
        target_module_genes = _split_semicolon(row.get("target_module_genes", ""))
        relation_docs = _module_relation_documents(docs, source_module_genes, target_module_genes)
        selection = select_evidence_documents(candidate_id, relation_context, relation_docs, evidence_config)
        query_label = module_query_label(row)
        packet = distiller.distill(query_label, relation_context, selection.documents)
        gene_evidence = _module_gene_evidence_summary(docs, source_module_genes, target_module_genes)
        pair_evidence, pair_evidence_rows = _module_document_edge_evidence_summary(
            selection.documents, _parse_module_edges(row["cross_module_edges"])
        )
        network_summary = module_network_summary(row)
        distilled_text = (
            f"Module-pair relationship evidence (documents mentioning at least one source and one target module gene):\n"
            f"{packet.distilled_text}\n\n{gene_evidence}\n\n{pair_evidence}"
        )
        built_packet = {
                "candidate_id": candidate_id,
                "method": args.method,
                "species": row["species"],
                "source_module": row["source_module"],
                "target_module": row["target_module"],
                "source_module_genes": row["source_module_genes"],
                "target_module_genes": row["target_module_genes"],
                "source_regulators": row["source_regulators"],
                "target_genes": row["target_genes"],
                "cross_module_edges": row["cross_module_edges"],
                "context": context,
                "module_relation_context": relation_context,
                "retrieved_count": selection.retrieved_count,
                "evidence_count": selection.evidence_count,
                "evidence_filter_method": selection.filter_method,
                "distilled_text": distilled_text,
                "distilled_pmids": ";".join(packet.selected_pmids),
                "distilled_go_terms": ";".join(packet.go_terms),
                "distilled_token_estimate": packet.token_estimate,
                "distillation_method": packet.method,
                "network_summary": network_summary,
                "documents": [asdict(doc) for doc in selection.documents],
                "gene_pair_evidence": pair_evidence_rows,
        }
        packets.append(built_packet)
        with packet_cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(built_packet, ensure_ascii=False) + "\n")
        print(
            f"prepared {idx}/{len(candidates)} module_pair={candidate_id} retrieved={len(docs)} relation_docs={len(relation_docs)} evidence={selection.evidence_count}",
            flush=True,
        )
    return packets


class ModulePubMedClient(PubMedClient):
    def __init__(
        self,
        *args: Any,
        per_gene_retmax: int = 8,
        gene_cache: JsonlRetrievalCache | None = None,
        block_cache: JsonlRetrievalCache | None = None,
        block_size: int = 8,
        block_retmax: int = 35,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.per_gene_retmax = per_gene_retmax
        self.gene_cache = gene_cache
        self.block_cache = block_cache
        self.block_size = block_size
        self.block_retmax = block_retmax
        self._gene_document_cache: dict[tuple[str, str], list[EvidenceDocument]] = {}

    def search_and_fetch_module(self, candidate: dict[str, Any], context: str) -> list[EvidenceDocument]:
        source_genes = _split_semicolon(candidate.get("source_module_genes", ""))
        target_genes = _split_semicolon(candidate.get("target_module_genes", ""))
        unique_genes = list(dict.fromkeys(source_genes + target_genes))
        docs_by_pmid: dict[str, EvidenceDocument] = {}
        for gene in unique_genes:
            key = (str(candidate.get("species", "")), gene.upper())
            if key not in self._gene_document_cache:
                cache_key = f"{key[0]}:{key[1]}"
                cached_docs = self.gene_cache.get(cache_key) if self.gene_cache is not None else None
                if cached_docs is not None:
                    self._gene_document_cache[key] = [doc for doc in cached_docs if doc.pmid not in self.exclude_pmids]
                else:
                    original_retmax = self.retmax
                    try:
                        self.retmax = self.per_gene_retmax
                        ids = self.search(self.build_gene_query(gene, candidate.get("species", ""), context))
                        self._gene_document_cache[key] = self.fetch(ids)
                    finally:
                        self.retmax = original_retmax
                    if self.gene_cache is not None:
                        self.gene_cache.set(cache_key, self._gene_document_cache[key])
            for document in self._gene_document_cache[key]:
                docs_by_pmid.setdefault(document.pmid, document)
        docs = list(docs_by_pmid.values())
        for rank, doc in enumerate(docs, start=1):
            doc.pubmed_rank = rank
        return docs

    def search_and_fetch_module_blocks(self, candidate: dict[str, Any], context: str) -> list[EvidenceDocument]:
        source_blocks = _gene_blocks(_split_semicolon(candidate.get("source_module_genes", "")), self.block_size)
        target_blocks = _gene_blocks(_split_semicolon(candidate.get("target_module_genes", "")), self.block_size)
        documents: dict[str, EvidenceDocument] = {}
        species = str(candidate.get("species", ""))
        for source_index, source_block in enumerate(source_blocks):
            for target_index, target_block in enumerate(target_blocks):
                cache_key = f"{candidate['candidate_id']}:S{source_index}:T{target_index}"
                cached = self.block_cache.get(cache_key) if self.block_cache is not None else None
                if cached is None:
                    original_retmax = self.retmax
                    try:
                        self.retmax = self.block_retmax
                        ids = self.search(self.build_module_block_query(source_block, target_block, species, strict=True))
                        if not ids:
                            ids = self.search(self.build_module_block_query(source_block, target_block, species, strict=False))
                        cached = self.fetch(ids)
                    except RuntimeError:
                        cached = []
                    finally:
                        self.retmax = original_retmax
                    if self.block_cache is not None:
                        self.block_cache.set(cache_key, cached)
                for doc in cached:
                    if doc.pmid not in self.exclude_pmids:
                        documents.setdefault(doc.pmid, doc)
        out = list(documents.values())
        for rank, doc in enumerate(out, start=1):
            doc.pubmed_rank = rank
        return out

    def search_and_fetch_module_pairs(self, candidate: dict[str, Any], context: str) -> dict[str, list[EvidenceDocument]]:
        evidence: dict[str, list[EvidenceDocument]] = {}
        species = str(candidate.get("species", ""))
        unresolved: list[tuple[str, str]] = []
        for source, target in _parse_module_edges(candidate.get("cross_module_edges", "")):
            edge_id = f"{species}:{source}->{target}"
            cached_docs = self.pair_cache.get(edge_id) if self.pair_cache is not None else None
            if cached_docs is None:
                unresolved.append((source, target))
            else:
                evidence[f"{source}->{target}"] = [doc for doc in cached_docs if doc.pmid not in self.exclude_pmids]

        # Query every edge, but batch 20 edge clauses per Entrez request.  This
        # preserves edge-level attribution while avoiding thousands of serial
        # requests for a module-pair experiment.
        for start in range(0, len(unresolved), 20):
            chunk = unresolved[start : start + 20]
            original_retmax = self.retmax
            try:
                self.retmax = max(20, self.per_pair_retmax * len(chunk))
                try:
                    ids = self.search(self.build_pair_batch_query(chunk, species, context))
                    if not ids:
                        ids = self.search(self.build_pair_batch_query(chunk, "", ""))
                    documents = self.fetch(ids)
                except RuntimeError:
                    documents = []
            finally:
                self.retmax = original_retmax
            for source, target in chunk:
                matching = [
                    doc
                    for doc in documents
                    if _gene_hits(f"{doc.title} {doc.abstract}", [source])
                    and _gene_hits(f"{doc.title} {doc.abstract}", [target])
                ]
                edge_id = f"{species}:{source}->{target}"
                if self.pair_cache is not None:
                    self.pair_cache.set(edge_id, matching)
                evidence[f"{source}->{target}"] = matching

        for edge, docs in evidence.items():
            for rank, doc in enumerate(docs, start=1):
                doc.pubmed_rank = rank
        return evidence

    def build_gene_query(self, gene: str, species: str, context: str) -> str:
        context_terms = f'({_context_pubmed_query(context)}) OR "cell fate"[Title/Abstract]'
        clauses = [f"({_gene_pubmed_query(gene)})", f"({context_terms})"]
        species_terms = self._species_terms(str(species))
        if species_terms:
            clauses.append(f"({species_terms})")
        return f"({' AND '.join(clauses)})"

    def build_module_block_query(self, source_block: list[str], target_block: list[str], species: str, strict: bool) -> str:
        source_clause = " OR ".join(_gene_pubmed_query(gene) for gene in source_block)
        target_clause = " OR ".join(_gene_pubmed_query(gene) for gene in target_block)
        relation_terms = 'regulat*[Title/Abstract] OR activat*[Title/Abstract] OR repress*[Title/Abstract] OR target*[Title/Abstract] OR bind*[Title/Abstract] OR "gene regulatory network"[Title/Abstract]'
        clauses = [f"({source_clause})", f"({target_clause})", f"({relation_terms})"]
        if strict:
            context = getattr(self, "context", "")
            if context:
                clauses.append(f'({_context_pubmed_query(context)}) OR "cell fate"[Title/Abstract]')
            species_terms = self._species_terms(species)
            if species_terms:
                clauses.append(f"({species_terms})")
        return f"({' AND '.join(clauses)})"

    def build_pair_batch_query(self, pairs: list[tuple[str, str]], species: str, context: str) -> str:
        relation_terms = 'regulat*[Title/Abstract] OR activat*[Title/Abstract] OR repress*[Title/Abstract] OR target*[Title/Abstract] OR bind*[Title/Abstract] OR "transcription factor"[Title/Abstract]'
        pair_clause = " OR ".join(
            f"(({_gene_pubmed_query(source)}) AND ({_gene_pubmed_query(target)}))" for source, target in pairs
        )
        clauses = [f"({pair_clause})", f"({relation_terms})"]
        if context:
            clauses.append(f'({_context_pubmed_query(context)}) OR "cell fate"[Title/Abstract]')
        species_terms = self._species_terms(species) if species else ""
        if species_terms:
            clauses.append(f"({species_terms})")
        return f"({' AND '.join(clauses)})"

    @staticmethod
    def _species_terms(species: str) -> str:
        if species == "mouse":
            return '"mouse"[Title/Abstract] OR murine[Title/Abstract] OR "Mus musculus"[Title/Abstract] OR vertebrate[Title/Abstract]'
        if species == "zebrafish":
            return '"zebrafish"[Title/Abstract] OR "Danio rerio"[Title/Abstract] OR vertebrate[Title/Abstract]'
        if species == "human":
            return '"human"[Title/Abstract] OR "Homo sapiens"[Title/Abstract] OR patient*[Title/Abstract]'
        return ""


def _gene_hits(text: str, genes: list[str]) -> list[str]:
    normalized = str(text or "")
    return [gene for gene in genes if re.search(rf"(?<![A-Za-z0-9]){re.escape(gene)}(?![A-Za-z0-9])", normalized, flags=re.IGNORECASE)]


def _module_relation_documents(
    docs: list[EvidenceDocument],
    source_genes: list[str],
    target_genes: list[str],
) -> list[EvidenceDocument]:
    relation_words = re.compile(r"\b(regulat(?:e|es|ed|ing|or)|repress(?:es|ed|ion)?|activat(?:e|es|ed|ion)|target(?:s|ed|ing)?|bind(?:s|ing)?|control(?:s|led|ling)?)\b", re.IGNORECASE)
    ranked: list[tuple[int, EvidenceDocument]] = []
    for document in docs:
        text = f"{document.title} {document.abstract}"
        source_hits = _gene_hits(text, source_genes)
        target_hits = _gene_hits(text, target_genes)
        if not source_hits or not target_hits:
            continue
        score = 4 * min(len(source_hits), 3) + 4 * min(len(target_hits), 3) + int(bool(relation_words.search(text)))
        ranked.append((score, document))
    return [document for _, document in sorted(ranked, key=lambda item: (-item[0], int(item[1].pubmed_rank or 999999)))]


def _module_gene_evidence_summary(
    docs: list[EvidenceDocument],
    source_genes: list[str],
    target_genes: list[str],
) -> str:
    def summarize(side: str, genes: list[str]) -> list[str]:
        lines = [f"{side} complete-module gene evidence ({len(genes)} genes):"]
        for gene in genes:
            matches = [doc for doc in docs if gene in _gene_hits(f"{doc.title} {doc.abstract}", [gene])]
            if matches:
                citations = "; ".join(f"PMID {doc.pmid}: {doc.title[:110]}" for doc in matches[:2])
                lines.append(f"- {gene}: {citations}")
            else:
                lines.append(f"- {gene}: no retrieved context-matched record")
        return lines

    return "\n".join(summarize("Source module", source_genes) + [""] + summarize("Target module", target_genes))


def _parse_module_edges(value: Any) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for item in _split_semicolon(value):
        if "->" not in item:
            continue
        source, target = (part.strip() for part in item.split("->", 1))
        if source and target:
            edges.append((source, target))
    return edges


def _gene_blocks(genes: list[str], block_size: int) -> list[list[str]]:
    size = max(1, int(block_size))
    return [genes[start : start + size] for start in range(0, len(genes), size)]


def _module_document_edge_evidence_summary(
    documents: list[EvidenceDocument], edges: list[tuple[str, str]]
) -> tuple[str, list[dict[str, Any]]]:
    """Annotate selected module-level documents by side and original GRN edges."""
    rows: list[dict[str, Any]] = []
    for doc in documents:
        text = f"{doc.title} {doc.abstract}"
        matched_edges = [(source, target) for source, target in edges if _gene_hits(text, [source]) and _gene_hits(text, [target])]
        source_hits = sorted({source for source, _ in matched_edges})
        target_hits = sorted({target for _, target in matched_edges})
        rows.append(
            {
                "pmid": doc.pmid,
                "title": doc.title,
                "source_genes_hit": ";".join(source_hits),
                "target_genes_hit": ";".join(target_hits),
                "grn_edges_covered": ";".join(f"{source}->{target}" for source, target in matched_edges),
                "n_grn_edges_covered": len(matched_edges),
            }
        )
    lines = [
        f"Full-module document-level edge annotation: {len(documents)} selected documents; {len(edges)} original cross-module GRN edges supplied.",
        "Each row below lists module genes and original GRN edges co-mentioned in that document; co-mention is not proof of direction.",
    ]
    for row in rows:
        lines.append(
            f"- PMID {row['pmid']} | source hits: {row['source_genes_hit'] or 'none'} | target hits: {row['target_genes_hit'] or 'none'} | GRN edges covered: {row['grn_edges_covered'] or 'none'} | {row['title']}"
        )
    return "\n".join(lines), rows


def _module_gene_pair_evidence_summary(
    documents_by_edge: dict[str, list[EvidenceDocument]],
    max_llm_records: int = 30,
) -> tuple[str, list[dict[str, Any]]]:
    """Create an auditable all-edge package and a bounded LLM-facing synopsis.

    Every directed edge is queried and retained in ``rows``.  The LLM receives
    the highest-specificity records rather than thousands of unfiltered titles.
    """
    relation_words = re.compile(
        r"\b(regulat(?:e|es|ed|ing|or)|repress(?:es|ed|ion)?|activat(?:e|es|ed|ion)|target(?:s|ed|ing)?|bind(?:s|ing)?|control(?:s|led|ling)?)\b",
        re.IGNORECASE,
    )
    rows: list[dict[str, Any]] = []
    ranked: list[tuple[int, dict[str, Any]]] = []
    for edge, documents in documents_by_edge.items():
        source, target = edge.split("->", 1)
        for doc in documents:
            text = f"{doc.title} {doc.abstract}"
            source_hit = bool(_gene_hits(text, [source]))
            target_hit = bool(_gene_hits(text, [target]))
            relation_hit = bool(relation_words.search(text))
            record = {
                "gene_pair": edge,
                "source_gene": source,
                "target_gene": target,
                "pmid": doc.pmid,
                "title": doc.title,
                "abstract": doc.abstract,
                "source_mentioned": source_hit,
                "target_mentioned": target_hit,
                "relation_term_mentioned": relation_hit,
            }
            rows.append(record)
            if source_hit and target_hit and relation_hit:
                ranked.append((3, record))
            elif source_hit and target_hit:
                ranked.append((2, record))
    ranked.sort(key=lambda item: (item[0], item[1]["gene_pair"], item[1]["pmid"]), reverse=True)
    selected = [record for _, record in ranked[:max_llm_records]]
    queried = len(documents_by_edge)
    retrieved = len(rows)
    lines = [
        f"Directed GRN gene-pair regulation package: all {queried} cross-module edges were queried independently; {retrieved} PubMed records were retrieved.",
        "The following are the highest-specificity records (both edge genes visible in title/abstract; relation-term records ranked first).",
    ]
    if selected:
        for record in selected:
            abstract = " ".join(str(record["abstract"] or "").split())[:420]
            lines.append(f"- {record['gene_pair']} | PMID {record['pmid']} | {record['title']} | {abstract}")
    else:
        lines.append("- No retrieved record mentioned both genes of a directed GRN edge in its title/abstract.")
    lines.append("Do not infer direction from co-mention alone; assess direction only when the cited record explicitly supports it.")
    return "\n".join(lines), rows


def score_module_packets_with_llm(packets: list[dict[str, Any]], args: argparse.Namespace, out: Path) -> pd.DataFrame:
    api_key = read_api_key(args.api_key, args.api_key_file, getattr(args, "provider", None))
    response_path = out.with_name(f"{out.stem}_response.txt")
    api_response_path = out.with_name(f"{out.stem}_api_response.jsonl")
    manifest_path = out.with_name(f"{out.stem}_request_manifest.json")
    score_frames = []
    if args.resume and out.exists():
        existing = pd.read_csv(out)
        if not existing.empty:
            score_frames.append(existing)
    completed = {
        candidate.upper()
        for frame in score_frames
        for candidate in frame.get("candidate_id", pd.Series(dtype=str)).astype(str)
    }
    pending = [packet for packet in packets if str(packet["candidate_id"]).upper() not in completed]
    if not args.resume:
        response_path.write_text("", encoding="utf-8")
        api_response_path.write_text("", encoding="utf-8")
    for batch_idx, start in enumerate(range(0, len(pending), args.request_batch_size), start=1):
        batch = pending[start : start + args.request_batch_size]
        prompt = module_scoring_prompt(batch)
        response_text, response_data = call_llm(
            prompt=prompt,
            api_key=api_key,
            model=args.model,
            provider=getattr(args, "provider", None),
            base_url=args.base_url,
            thinking=args.thinking,
            temperature=args.temperature,
            max_tokens=args.max_output_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        with response_path.open("a", encoding="utf-8") as handle:
            handle.write(f"--- RESPONSE {batch_idx} ---\n{response_text}\n\n")
        with api_response_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(response_data, ensure_ascii=False) + "\n")
        batch_scores = parse_module_score_csv(response_text)
        try:
            validate_module_scores(batch_scores, [str(packet["candidate_id"]) for packet in batch], batch)
        except ValueError as exc:
            # Ask for one grounded correction rather than accepting an LLM
            # hallucinated module membership or aborting an otherwise resumable run.
            correction_prompt = (
                prompt
                + "\n\nThe previous response failed validation: "
                + str(exc)
                + "\nReturn a corrected CSV. Every evidence-gene field must contain only symbols from its corresponding allowed list; remove an invalid symbol rather than substituting one."
            )
            corrected_text, corrected_data = call_llm(
                prompt=correction_prompt,
                api_key=api_key,
                model=args.model,
                provider=getattr(args, "provider", None),
                base_url=args.base_url,
                thinking=args.thinking,
                temperature=args.temperature,
                max_tokens=args.max_output_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            with response_path.open("a", encoding="utf-8") as handle:
                handle.write(f"--- CORRECTION RESPONSE {batch_idx} ---\n{corrected_text}\n\n")
            with api_response_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(corrected_data, ensure_ascii=False) + "\n")
            batch_scores = parse_module_score_csv(corrected_text)
            validate_module_scores(batch_scores, [str(packet["candidate_id"]) for packet in batch], batch)
        score_frames.append(batch_scores)
        scores = order_module_scores(pd.concat(score_frames, ignore_index=True), [str(packet["candidate_id"]) for packet in packets])
        scores.to_csv(out, index=False)
        print(f"scored_batch={batch_idx} rows={len(batch_scores)} total={len(scores)}", flush=True)
    scores = (
        order_module_scores(pd.concat(score_frames, ignore_index=True), [str(packet["candidate_id"]) for packet in packets])
        if score_frames
        else pd.DataFrame(columns=MODULE_SCORE_COLUMNS)
    )
    scores.to_csv(out, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "provider": getattr(args, "provider", None),
                "model": args.model,
                "base_url": args.base_url,
                "thinking": args.thinking,
                "method": args.method,
                "n_packets": len(packets),
                "score_columns": MODULE_SCORE_COLUMNS,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return scores


def module_scoring_prompt(packets: list[dict[str, Any]]) -> str:
    method = str(packets[0]["method"]) if packets else "distilled_plus_prior"
    lines = [
        "You are the biomedical LLM judge for FateReasoner.",
        f"Score each GRN module-pair candidate in the supplied biological context using this method: {method}.",
        "",
        _method_rule(method),
        "Use only the distilled literature evidence plus allowed model prior knowledge.",
        "Score the specific source module -> target module relationship, not merely whether individual member genes are biologically relevant.",
        "Do not use network data metrics, data_score, edge counts, correlations, motif counts, FOS, or module density for this FateReasoner score.",
        "Those GRN data features are retained separately as an independent network axis.",
        "Reward module pairs that plausibly encode cell-fate regulatory programs relevant to the supplied context, based on the literature packet and prior knowledge.",
        "The packet has separate single-gene and directed GRN gene-pair evidence sections. Use single-gene evidence for module relevance and directed pair evidence for the source-to-target claim.",
        "Do not upgrade a directional claim from gene relevance, network membership, or gene-pair co-mention alone.",
        "Distinguish direct module/GRN evidence from evidence about several members and from evidence about only one component.",
        "Do not call a module pair directly supported when the cited paper only supports one TF, one target, or a generic pathway.",
        "For self-loops, describe module-internal connectivity; do not call it a validated autoregulatory circuit without explicit evidence.",
        "Penalize generic development, cancer-only evidence, unrelated tissue, weak species/context match, and symbol ambiguity.",
        "Cite only PMIDs visible in the packet in supporting_pmids. If prior knowledge contributes, mention it without inventing PMIDs.",
        "module_evidence_scope must be one of direct_module_evidence, convergent_member_evidence, component_only_evidence, or prior_only.",
        "context_match must be one of matched, related, unrelated, or unclear.",
        "evidence_strength must be one of strong, moderate, weak, or none.",
        "source_evidence_genes and target_evidence_genes must list, separated by semicolons, only the source and target genes explicitly used as evidence in the rationale; leave a field blank when no gene-specific evidence is available.",
        "Never attribute a gene to the source module when it is present only in the target module, or vice versa.",
        "Treat the two allowed gene lists as distinct even for self-loops: a source gene is not automatically a target gene. If membership is uncertain, leave the evidence-gene field blank rather than guessing.",
        "Calibrate fatereasoner_score continuously from 0.0 to 1.0 from the complete evidence packet; do not apply an artificial score cap or manufacture a direct regulatory claim.",
        "",
        "Return CSV only, with exactly these columns:",
        ",".join(MODULE_SCORE_COLUMNS),
        "",
        "fatereasoner_score is a single number from 0.0 to 1.0.",
        "method must exactly match the requested method.",
        "The annotation columns describe evidence scope and do not replace or rescale fatereasoner_score.",
        f"Return exactly {len(packets)} data rows, one row for each candidate.",
        "",
    ]
    for packet in packets:
        lines.extend(
            [
                f"## CANDIDATE: {packet['candidate_id']}",
                f"Species: {packet['species']}",
                f"Relationship: source module {packet['source_module']} regulates target module {packet['target_module']}",
                f"Complete source-module genes: {packet.get('source_module_genes', '')}",
                f"Complete target-module genes: {packet.get('target_module_genes', '')}",
                f"Pair-connected source genes: {packet.get('source_regulators', '')}",
                f"Pair-connected target genes: {packet.get('target_genes', '')}",
                f"All original cross-module GRN edges: {packet.get('cross_module_edges', '')}",
                f"Context: {packet['context']}",
                f"Method: {packet['method']}",
                f"Retrieved documents: {packet['retrieved_count']}; selected evidence documents: {packet['evidence_count']}",
                f"Distilled PMIDs: {packet['distilled_pmids']}",
                "Distilled literature evidence packet:",
                str(packet["distilled_text"]),
                "",
            ]
        )
    return "\n".join(lines)


def parse_module_score_csv(text: str) -> pd.DataFrame:
    cleaned = text.strip()
    fence = re.search(r"```(?:csv)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    header_idx = cleaned.find("candidate_id,")
    if header_idx > 0:
        cleaned = cleaned[header_idx:]
    rows = []
    structured_header = "source_module" in next((line for line in cleaned.splitlines() if line.strip()), "")
    for row in csv.reader(StringIO(cleaned)):
        if not row or row[0].strip() == "candidate_id":
            continue
        if len(row) < 6:
            continue
        if structured_header:
            legacy_structured_columns = len(MODULE_SCORE_COLUMNS) - 2
            if len(row) < legacy_structured_columns:
                continue
            fixed = [cell.strip() for cell in row[:7]]
            if len(row) >= len(MODULE_SCORE_COLUMNS):
                rationale = ",".join(row[7:-6]).strip()
                caveats = row[-6].strip()
                annotations = [cell.strip() for cell in row[-5:]]
            else:
                rationale = ",".join(row[7:-4]).strip()
                caveats = row[-4].strip()
                annotations = [cell.strip() for cell in row[-3:]] + ["", ""]
            rows.append(fixed + [rationale, caveats] + annotations)
        else:
            # Backward-compatible parser for the historical seven-column output.
            if len(row) < 7:
                continue
            rows.append(
                [
                    row[0].strip(),
                    "",
                    "",
                    row[1].strip(),
                    row[2].strip(),
                    row[3].strip(),
                    row[4].strip(),
                    ",".join(row[5:-1]).strip(),
                    row[-1].strip(),
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
    if not rows:
        raise ValueError("No module score rows found in LLM response")
    return pd.DataFrame(rows, columns=MODULE_SCORE_COLUMNS)


def validate_module_scores(
    df: pd.DataFrame,
    expected_ids: list[str],
    packets: list[dict[str, Any]] | None = None,
) -> None:
    optional_columns = {"source_evidence_genes", "target_evidence_genes"}
    for column in optional_columns - set(df.columns):
        df[column] = ""
    missing = [column for column in MODULE_SCORE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns: {missing}")
    observed = {candidate.upper() for candidate in df["candidate_id"].astype(str)}
    expected = {candidate.upper() for candidate in expected_ids}
    if observed != expected:
        raise ValueError(f"Module candidate mismatch: expected={sorted(expected)} observed={sorted(observed)}")
    if not df["method"].astype(str).isin(SCORING_METHODS).all():
        raise ValueError(f"method must be one of {SCORING_METHODS}")
    values = pd.to_numeric(df["fatereasoner_score"], errors="coerce")
    if values.isna().any() or ((values < 0) | (values > 1)).any():
        raise ValueError("fatereasoner_score must contain numeric values in [0, 1]")
    allowed_scopes = {"direct_module_evidence", "convergent_member_evidence", "component_only_evidence", "prior_only"}
    allowed_contexts = {"matched", "related", "unrelated", "unclear"}
    allowed_strengths = {"strong", "moderate", "weak", "none"}
    if packets is not None:
        packet_map = {str(packet["candidate_id"]).upper(): packet for packet in packets}
        for row_index, row in enumerate(df.to_dict("records")):
            key = str(row["candidate_id"]).upper()
            packet = packet_map.get(key)
            if packet is None:
                raise ValueError(f"Missing module packet for score: {row['candidate_id']}")
            expected_source = str(packet.get("source_module", "")).strip()
            expected_target = str(packet.get("target_module", "")).strip()
            observed_source = str(row.get("source_module", "")).strip()
            observed_target = str(row.get("target_module", "")).strip()
            # Some LLMs echo the species prefix from candidate_id (e.g. mouse:M7)
            # even though the module field itself is M7. Normalize that harmless
            # formatting variation before enforcing the packet-level invariant.
            species_prefix = f"{str(packet.get('species', '')).strip()}:"
            if observed_source == f"{species_prefix}{expected_source}":
                observed_source = expected_source
                df.at[row_index, "source_module"] = expected_source
            if observed_target == f"{species_prefix}{expected_target}":
                observed_target = expected_target
                df.at[row_index, "target_module"] = expected_target
            if observed_source != expected_source:
                raise ValueError(f"Source module mismatch for {row['candidate_id']}")
            if observed_target != expected_target:
                raise ValueError(f"Target module mismatch for {row['candidate_id']}")
            packet_pmids = {
                str(pmid).strip()
                for pmid in str(packet.get("distilled_pmids", "")).split(";")
                if str(pmid).strip()
            }
            packet_pmids.update(
                str(document.get("pmid", "")).strip()
                for document in packet.get("documents", [])
                if str(document.get("pmid", "")).strip()
            )
            packet_pmids.update(
                str(evidence.get("pmid", "")).strip()
                for evidence in packet.get("gene_pair_evidence", [])
                if str(evidence.get("pmid", "")).strip()
            )
            cited_pmids = {
                str(pmid).strip()
                for pmid in str(row.get("supporting_pmids", "")).replace(",", ";").split(";")
                if str(pmid).strip()
            }
            if not cited_pmids.issubset(packet_pmids):
                raise ValueError(f"Supporting PMID not present in packet for {row['candidate_id']}")
            if str(row.get("module_evidence_scope", "")) not in allowed_scopes:
                raise ValueError(f"Invalid module_evidence_scope for {row['candidate_id']}")
            if str(row.get("context_match", "")) not in allowed_contexts:
                raise ValueError(f"Invalid context_match for {row['candidate_id']}")
            if str(row.get("evidence_strength", "")) not in allowed_strengths:
                raise ValueError(f"Invalid evidence_strength for {row['candidate_id']}")
            _validate_evidence_genes(
                row.get("source_evidence_genes", ""),
                packet.get("source_module_genes", packet.get("source_regulators", "")),
                "source",
                row["candidate_id"],
            )
            _validate_evidence_genes(
                row.get("target_evidence_genes", ""),
                packet.get("target_module_genes", packet.get("target_genes", "")),
                "target",
                row["candidate_id"],
            )


def _validate_evidence_genes(value: Any, allowed_value: Any, side: str, candidate_id: str) -> None:
    cited = {gene.strip().upper() for gene in re.split(r"[;,]", str(value)) if gene.strip()}
    allowed = {gene.strip().upper() for gene in _split_semicolon(str(allowed_value)) if gene.strip()}
    if cited and (not allowed or not cited.issubset(allowed)):
        invalid = sorted(cited - allowed)
        raise ValueError(f"Invalid {side} evidence genes for {candidate_id}: {invalid}")


def order_module_scores(df: pd.DataFrame, candidate_ids: list[str]) -> pd.DataFrame:
    work = df.drop_duplicates("candidate_id", keep="last").copy()
    order = {candidate.upper(): idx for idx, candidate in enumerate(candidate_ids)}
    work["_order"] = work["candidate_id"].astype(str).str.upper().map(order)
    return work.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def merge_module_scores(candidates: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    merged = candidates.merge(scores, on="candidate_id", how="left")
    merged["fatereasoner_score"] = pd.to_numeric(merged["fatereasoner_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    merged["evidence_quadrant"] = merged.apply(_evidence_quadrant, axis=1)
    merged["fatereasoner_rank"] = merged["fatereasoner_score"].rank(method="first", ascending=False).astype(int)
    merged["data_rank"] = merged["data_score"].rank(method="first", ascending=False).astype(int)
    return merged.sort_values(
        ["fatereasoner_score", "data_score", "candidate_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def module_relation_context(row: dict[str, Any], context: str) -> str:
    return (
        f"{context}; {row['species']} GRN module-pair {row['source_module']} -> {row['target_module']}; "
        f"source regulators {row['top_regulators']}; target genes {row['top_targets']}"
    )


def module_query_label(row: dict[str, Any]) -> str:
    # The evidence distiller must see the complete module membership as well:
    # retrieval is deliberately performed per gene and no longer starts from a
    # hand-picked top-gene query.
    source_genes = _split_semicolon(row.get("source_module_genes", ""))
    target_genes = _split_semicolon(row.get("target_module_genes", ""))
    return " ".join(source_genes + target_genes) or str(row["candidate_id"])


def module_network_summary(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Network evidence summary:",
            f"candidate_id: {row['candidate_id']}",
            f"species: {row['species']}",
            f"relationship: source module {row['source_module']} regulates target module {row['target_module']}",
            f"top source regulators: {row['top_regulators']}",
            f"top target genes: {row['top_targets']}",
            f"top TF-target edges: {row['top_edges']}",
            f"n_edges: {row['n_edges']}; n_regulators: {row['n_regulators']}; n_targets: {row['n_targets']}",
            f"sum_abs_correlation: {row['sum_abs_correlation']:.4f}; mean_abs_correlation: {row['mean_abs_correlation']:.4f}; mean_signed_correlation: {row['mean_signed_correlation']:.4f}",
            f"edge_density: {row['edge_density']:.4f}; source_specificity: {row['source_specificity']:.4f}; target_specificity: {row['target_specificity']:.4f}",
            f"motif_locus_count: {row['motif_locus_count']}; promoter_locus_fraction: {row['promoter_locus_fraction']:.4f}",
            f"regulation_type_summary: {row['regulation_type_summary']}",
            f"data_score: {row['data_score']:.4f}",
        ]
    )


def read_text_arg(value: str) -> str:
    path = Path(value)
    if path.exists():
        return " ".join(path.read_text(encoding="utf-8").split())
    return " ".join(value.split())


def packets_to_table(packets: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{key: value for key, value in packet.items() if key not in {"documents", "gene_pair_evidence"}} for packet in packets]
    )


def module_gene_pair_evidence_to_table(packets: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for packet in packets:
        for evidence in packet.get("gene_pair_evidence", []):
            rows.append({"candidate_id": packet["candidate_id"], **evidence})
    return pd.DataFrame(rows)


def _evidence_quadrant(row: pd.Series) -> str:
    data = float(row.get("data_score", 0.0))
    score = float(row.get("fatereasoner_score", 0.0))
    if data >= 0.5 and score >= 0.7:
        return "concordant_high"
    if data >= 0.5:
        return "data_high_llm_lower"
    if score >= 0.7:
        return "llm_high_data_lower"
    return "lower_concordance"


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().sum() == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    lo = values.min()
    hi = values.max()
    if math.isclose(float(lo), float(hi)):
        return pd.Series(np.ones(len(series)), index=series.index)
    return ((values - lo) / (hi - lo)).fillna(0.0)


def _safe_div(numerator: Any, denominator: Any) -> float:
    denominator = float(denominator or 0.0)
    if math.isclose(denominator, 0.0):
        return 0.0
    return float(numerator or 0.0) / denominator


def _top_symbols(group: pd.DataFrame, symbol_col: str, weight_col: str, n: int) -> str:
    ranked = (
        group.groupby(symbol_col)[weight_col]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .index.astype(str)
        .tolist()
    )
    return ";".join(ranked)


def _top_edges(group: pd.DataFrame, n: int) -> str:
    work = group.sort_values("abs_correlation", ascending=False).head(n)
    return "; ".join(
        f"{row.regulator_symbol}->{row.target_symbol}({float(row.expression_correlation):.3f})"
        for row in work.itertuples(index=False)
    )


def _value_summary(series: pd.Series) -> str:
    counts = series.astype(str).value_counts()
    return ";".join(f"{key}:{value}" for key, value in counts.items())


def _split_semicolon(value: Any) -> list[str]:
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return ""
