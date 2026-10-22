from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .llm import PROVIDERS, call_llm_chat, read_api_key as resolve_api_key
from .distillation import EvidenceDistillationConfig, EvidenceDistiller
from .evidence import EvidenceSelectionConfig, select_evidence_documents
from .models import EvidenceDocument
from .retrieval import JsonlRetrievalCache, PubMedClient, _gene_pubmed_query, parse_exclude_pmids


SCORE_COLUMNS = [
    "GENE",
    "method",
    "score",
    "evidence_level",
    "supporting_pmids",
    "rationale",
    "caveats",
]

SCORING_METHODS = (
    "distilled_plus_prior",
    "biomedbert_plus_prior",
    "distilled_only_no_prior",
    "prior_only",
    "raw_abstract_plus_prior",
)


class ContextPubMedClient(PubMedClient):
    """PubMed retrieval that uses the user-supplied biological context."""

    def build_query(self, gene: str, context: str) -> str:
        return f"({_gene_pubmed_query(gene)}) AND ({_context_pubmed_query(context)})"

    def search_and_fetch(self, gene: str, context: str) -> list[EvidenceDocument]:
        ids = self.search(self.build_query(gene, context))
        if not ids:
            ids = self.search(_gene_pubmed_query(gene))
        docs = self.fetch(ids)
        if self.full_text_mode == "pmc-oa":
            self._add_pmc_full_text(docs)
        for rank, doc in enumerate(docs, start=1):
            doc.pubmed_rank = rank
        return docs


def add_score_genes_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "score-genes",
        help="Score a gene list with distilled evidence plus LLM prior.",
    )
    parser.add_argument("--genes", default=None, help="Gene list file, comma/newline string, or CSV/TSV table.")
    parser.add_argument("--context", default=None, help="Biological context, or a path to a text file containing it.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--precomputed-packets",
        "--distilled-packets",
        dest="precomputed_packets",
        default=None,
        help="Existing distilled packet JSONL/CSV. Skips PubMed retrieval, evidence selection, and distillation.",
    )
    parser.add_argument(
        "--method",
        choices=SCORING_METHODS,
        default="distilled_plus_prior",
        help="Scoring condition. The default is the FateReasoner product score.",
    )
    parser.add_argument("--gene-col", default=None, help="Gene column for CSV/TSV input. Auto-detected by default.")
    parser.add_argument("--retmax", type=int, default=30, help="PubMed records retrieved per gene before filtering.")
    parser.add_argument("--full-text", choices=["none", "pmc-oa"], default="none")
    parser.add_argument("--date-cutoff", default=None, help="Latest allowed PubMed publication date, YYYY-MM-DD.")
    parser.add_argument("--delay-seconds", type=float, default=0.34)
    parser.add_argument(
        "--exclude-pmids",
        default="",
        help="Comma/semicolon separated PubMed IDs to exclude from retrieval and evidence packets.",
    )
    parser.add_argument("--evidence-threshold", type=float, default=0.35)
    parser.add_argument("--evidence-min-keep", type=int, default=5)
    parser.add_argument("--evidence-max-keep", type=int, default=20)
    parser.add_argument(
        "--distill-evidence",
        choices=["lite_weightfree", "lite_weighted", "lite", "medcpt_go", "biomedbert_rerank"],
        default="lite_weightfree",
    )
    parser.add_argument("--distilled-max-chars", type=int, default=3000)
    parser.add_argument("--distilled-packet-style", choices=["audit", "concise"], default="concise")
    parser.add_argument("--distill-max-windows", type=int, default=6)
    parser.add_argument("--distilled-anchor-docs", type=int, default=2)
    parser.add_argument("--distilled-anchor-chars", type=int, default=520)
    parser.add_argument("--anchor-strategy", choices=["balanced", "diverse_preserve"], default="diverse_preserve")
    parser.add_argument("--include-go", dest="include_go", action="store_true", default=False)
    parser.add_argument("--no-go", dest="include_go", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--medcpt-model", default="ncbi/MedCPT-Cross-Encoder")
    parser.add_argument(
        "--biomedbert-model",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        help="HuggingFace model name or local path used when --distill-evidence biomedbert_rerank.",
    )
    parser.add_argument("--provider", choices=PROVIDERS, default=None, help="LLM provider (required unless --prepare-only).")
    parser.add_argument("--model", default=None, help="Provider model ID; defaults to the provider preset.")
    parser.add_argument("--api-key", default=None, help="API key; an environment variable or --api-key-file is also accepted.")
    parser.add_argument("--api-key-file", default=None, help="File containing the provider API key.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--request-batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true", help="Retrieve and distill evidence but do not call the LLM API.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing retrieval cache and partial score CSV.")
    parser.set_defaults(func=run_score_genes)


def run_score_genes(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.precomputed_packets:
        packets = load_precomputed_packets(Path(args.precomputed_packets), args.method)
        genes = [str(packet["GENE"]) for packet in packets]
        context_values = [str(packet.get("context", "")) for packet in packets if packet.get("context")]
        context = read_context(args.context) if args.context else (context_values[0] if context_values else "")
    else:
        if not args.genes:
            raise ValueError("--genes is required unless --precomputed-packets is provided.")
        if not args.context:
            raise ValueError("--context is required unless --precomputed-packets is provided.")
        genes = read_gene_list(args.genes, args.gene_col)
        context = read_context(args.context)
        if not genes:
            raise ValueError("No genes found in --genes input.")
        packets = build_distilled_packets(genes, context, args, out_dir)

    if not packets:
        raise ValueError("No packets available for scoring.")
    ensure_unique_genes(packets)
    (out_dir / "context.txt").write_text(context + "\n", encoding="utf-8")
    pd.DataFrame({"GENE": genes}).to_csv(out_dir / "input_genes.csv", index=False)
    packets_path = out_dir / f"{args.method}_packets.jsonl"
    write_jsonl(packets_path, packets)
    packet_table = packets_to_table(packets)
    packet_table.to_csv(out_dir / "distilled_evidence_packets.csv", index=False)
    if args.prepare_only:
        print(f"genes={len(genes)} packets={packets_path}")
        print(f"prepare_only=true out={out_dir}")
        return

    score_path = out_dir / f"{args.method}_scores.csv"
    scores = score_packets_with_llm(packets, args, score_path)
    merged = (
        packet_table[["GENE", "context", "retrieved_count", "evidence_count", "distilled_pmids"]]
        .merge(scores, on="GENE", how="left")
    )
    merged.to_csv(out_dir / "gene_scores.csv", index=False)
    print(f"genes={len(genes)} scores={score_path}")
    print(f"result={out_dir / 'gene_scores.csv'}")


def read_gene_list(value: str, gene_col: str | None = None) -> list[str]:
    path = Path(value)
    if path.exists():
        if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            if path.suffix.lower() in {".csv", ".tsv"}:
                sep = "\t" if path.suffix.lower() == ".tsv" else ","
                df = pd.read_csv(path, sep=sep)
                column = gene_col or _detect_gene_column(df)
                return _unique_genes(df[column].astype(str).tolist())
            text = path.read_text(encoding="utf-8")
            return _unique_genes(re.split(r"[\s,;]+", text))
    return _unique_genes(re.split(r"[\s,;]+", value))


def _detect_gene_column(df: pd.DataFrame) -> str:
    for column in ("GENE", "gene", "Gene", "symbol", "SYMBOL"):
        if column in df.columns:
            return column
    if len(df.columns) == 1:
        return str(df.columns[0])
    raise ValueError(f"Could not infer gene column. Use --gene-col. Available columns: {list(df.columns)}")


def _unique_genes(values: list[str]) -> list[str]:
    genes: list[str] = []
    seen: set[str] = set()
    for value in values:
        gene = str(value).strip()
        if not gene or gene.lower() == "nan":
            continue
        key = gene.upper()
        if key not in seen:
            genes.append(gene)
            seen.add(key)
    return genes


def read_context(value: str) -> str:
    path = Path(value)
    if path.exists():
        return " ".join(path.read_text(encoding="utf-8").split())
    return " ".join(value.split())


def load_precomputed_packets(path: Path, method: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Precomputed packet file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".csv":
        rows = pd.read_csv(path).to_dict("records")
    else:
        raise ValueError(f"Unsupported precomputed packet format: {path.suffix}. Use JSONL or CSV.")

    packets: list[dict[str, Any]] = []
    for row in rows:
        row_method = str(row.get("method") or row.get("condition") or method)
        if row_method != method:
            continue
        packets.append(normalize_precomputed_packet(row, method))
    return packets


def ensure_unique_genes(packets: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for packet in packets:
        gene = str(packet["GENE"]).upper()
        if gene in seen:
            duplicates.add(gene)
        seen.add(gene)
    if duplicates:
        raise ValueError(
            "score-genes requires one packet per gene because the output schema is keyed by GENE only. "
            f"Duplicate genes found: {sorted(duplicates)[:10]}. Score each context separately or deduplicate the gene list."
        )


def normalize_precomputed_packet(row: dict[str, Any], method: str) -> dict[str, Any]:
    gene = str(row.get("GENE") or row.get("gene") or "").strip()
    if not gene:
        raise ValueError(f"Precomputed packet is missing GENE/gene: {row}")
    context = str(row.get("context") or "")
    documents = row.get("documents") if isinstance(row.get("documents"), list) else []
    evidence_documents = row.get("evidence_documents") if isinstance(row.get("evidence_documents"), list) else []
    distilled_text = str(row.get("distilled_text") or "")
    if not distilled_text and evidence_documents:
        distilled_text = "\n\n".join(str(doc.get("evidence_text", "")) for doc in evidence_documents if doc.get("evidence_text"))
    if not distilled_text:
        distilled_text = f"Gene: {gene}\nContext: {context}"
    distilled_pmids = str(row.get("distilled_pmids") or "")
    if not distilled_pmids and evidence_documents:
        distilled_pmids = ";".join(str(doc.get("pmid", "")) for doc in evidence_documents if doc.get("pmid"))
    retrieved_count = _safe_int(row.get("retrieved_count"), len(evidence_documents) or len(documents))
    evidence_count = _safe_int(row.get("evidence_count") or row.get("scoring_doc_count"), len(evidence_documents) or len(documents))
    return {
        "GENE": gene,
        "context": context,
        "method": method,
        "retrieved_count": retrieved_count,
        "evidence_count": evidence_count,
        "evidence_filter_method": str(row.get("evidence_filter_method") or "precomputed"),
        "distilled_text": distilled_text,
        "distilled_pmids": distilled_pmids,
        "distilled_go_terms": str(row.get("distilled_go_terms") or ""),
        "distilled_token_estimate": _safe_int(row.get("distilled_token_estimate"), max(1, len(distilled_text) // 4)),
        "distillation_method": str(row.get("distillation_method") or "precomputed"),
        "documents": documents or evidence_documents,
    }


def _safe_int(value: Any, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def build_distilled_packets(genes: list[str], context: str, args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    pubmed = ContextPubMedClient(
        retmax=args.retmax,
        date_cutoff=args.date_cutoff,
        delay_seconds=args.delay_seconds,
        full_text_mode=args.full_text,
        exclude_pmids=parse_exclude_pmids(getattr(args, "exclude_pmids", "")),
    )
    cache = JsonlRetrievalCache(out_dir / "retrieval_cache.jsonl")
    evidence_config = EvidenceSelectionConfig(
        threshold=args.evidence_threshold,
        min_keep=args.evidence_min_keep,
        max_keep=args.evidence_max_keep,
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
    packets: list[dict[str, Any]] = []
    for idx, gene in enumerate(genes, start=1):
        docs = cache.get(gene)
        if docs is None:
            docs = pubmed.search_and_fetch(gene, context)
            cache.set(gene, docs)
        else:
            docs = [doc for doc in docs if doc.pmid not in pubmed.exclude_pmids]
        selection = select_evidence_documents(gene, context, docs, evidence_config)
        packet = distiller.distill(gene, context, selection.documents)
        packets.append(
            {
                "GENE": gene,
                "context": context,
                "method": args.method,
                "retrieved_count": selection.retrieved_count,
                "evidence_count": selection.evidence_count,
                "evidence_filter_method": selection.filter_method,
                "distilled_text": packet.distilled_text,
                "distilled_pmids": ";".join(packet.selected_pmids),
                "distilled_go_terms": ";".join(packet.go_terms),
                "distilled_token_estimate": packet.token_estimate,
                "distillation_method": packet.method,
                "documents": [asdict(doc) for doc in selection.documents],
            }
        )
        print(f"prepared {idx}/{len(genes)} gene={gene} retrieved={selection.retrieved_count} evidence={selection.evidence_count}", flush=True)
    return packets


def packets_to_table(packets: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for packet in packets:
        rows.append({key: value for key, value in packet.items() if key != "documents"})
    return pd.DataFrame(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_packets_with_llm(packets: list[dict[str, Any]], args: argparse.Namespace, out: Path) -> pd.DataFrame:
    api_key = read_api_key(args.api_key, args.api_key_file, getattr(args, "provider", None))
    response_path = out.with_name(f"{out.stem}_response.txt")
    api_response_path = out.with_name(f"{out.stem}_api_response.jsonl")
    manifest_path = out.with_name(f"{out.stem}_request_manifest.json")
    score_frames = []
    if args.resume and out.exists():
        existing = pd.read_csv(out)
        if not existing.empty:
            score_frames.append(existing)
    completed = {gene.upper() for frame in score_frames for gene in frame.get("GENE", pd.Series(dtype=str)).astype(str)}
    pending = [packet for packet in packets if str(packet["GENE"]).upper() not in completed]
    if not args.resume:
        response_path.write_text("", encoding="utf-8")
        api_response_path.write_text("", encoding="utf-8")
    for batch_idx, start in enumerate(range(0, len(pending), args.request_batch_size), start=1):
        batch = pending[start : start + args.request_batch_size]
        scored_frames = _score_batch_with_fallback(
            batch=batch,
            batch_label=batch_idx,
            api_key=api_key,
            args=args,
            response_path=response_path,
            api_response_path=api_response_path,
        )
        for batch_scores in scored_frames:
            score_frames.append(batch_scores)
            scores = order_scores(pd.concat(score_frames, ignore_index=True), [str(packet["GENE"]) for packet in packets])
            scores.to_csv(out, index=False)
            print(f"scored_batch={batch_idx} rows={len(batch_scores)} total={len(scores)}", flush=True)
    scores = order_scores(pd.concat(score_frames, ignore_index=True), [str(packet["GENE"]) for packet in packets]) if score_frames else pd.DataFrame(columns=SCORE_COLUMNS)
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
                "score_columns": SCORE_COLUMNS,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return scores


def _score_batch_with_fallback(
    *,
    batch: list[dict[str, Any]],
    batch_label: int,
    api_key: str,
    args: argparse.Namespace,
    response_path: Path,
    api_response_path: Path,
) -> list[pd.DataFrame]:
    prompt = scoring_prompt(batch)
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
        handle.write(f"--- RESPONSE {batch_label} ---\n{response_text}\n\n")
    with api_response_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(response_data, ensure_ascii=False) + "\n")

    expected_genes = [str(packet["GENE"]) for packet in batch]
    try:
        batch_scores = parse_score_csv(response_text)
        canonical_genes = {str(packet["GENE"]).upper(): str(packet["GENE"]) for packet in batch}
        batch_scores = filter_expected_scores(batch_scores, expected_genes)
        batch_scores["method"] = str(batch[0]["method"])
        validate_scores(batch_scores, expected_genes)
        batch_scores["GENE"] = batch_scores["GENE"].astype(str).map(lambda gene: canonical_genes.get(gene.upper(), gene))
        return [batch_scores]
    except Exception as exc:
        if len(batch) <= 1:
            raise RuntimeError(
                f"Failed to score single-gene batch for {expected_genes[0]} after response parse/validation: {exc}"
            ) from exc
        mid = max(1, len(batch) // 2)
        left = _score_batch_with_fallback(
            batch=batch[:mid],
            batch_label=batch_label * 10 + 1,
            api_key=api_key,
            args=args,
            response_path=response_path,
            api_response_path=api_response_path,
        )
        right = _score_batch_with_fallback(
            batch=batch[mid:],
            batch_label=batch_label * 10 + 2,
            api_key=api_key,
            args=args,
            response_path=response_path,
            api_response_path=api_response_path,
        )
        return left + right


def scoring_prompt(packets: list[dict[str, Any]]) -> str:
    method = str(packets[0]["method"]) if packets else "distilled_plus_prior"
    method_rule = _method_rule(method)
    lines = [
        "You are the biomedical LLM judge for FateReasoner.",
        f"Score each gene for the supplied cell fate context using this method for every row: {method}.",
        "",
        method_rule,
        "Cite only PMIDs visible in the packet in supporting_pmids. If model prior knowledge contributes, mention it without inventing PMIDs.",
        "",
        "Return CSV only, with exactly these columns:",
        ",".join(SCORE_COLUMNS),
        "",
        "score is a single number from 0.0 to 1.0.",
        "method must exactly match the requested method.",
        "Only score genes explicitly listed in the packet headers as '## GENE: ...'.",
        "Do not create score rows for genes mentioned inside Context, target marker genes, evidence text, or rationale.",
        f"Return exactly {len(packets)} data rows, one row for each '## GENE' packet below, in the same order.",
        "Penalize unrelated species, unrelated tissue, generic cancer/expression evidence, symbol ambiguity, and weak context match.",
        "",
    ]
    for packet in packets:
        lines.extend(
            [
                f"## GENE: {packet['GENE']}",
                f"Context: {packet['context']}",
                f"Method: {packet['method']}",
                f"Retrieved documents: {packet['retrieved_count']}; selected evidence documents: {packet['evidence_count']}",
                f"Distilled PMIDs: {packet['distilled_pmids']}",
                "Distilled evidence packet:",
                str(packet["distilled_text"]),
                "",
            ]
        )
    return "\n".join(lines)


def call_llm(**kwargs: Any) -> tuple[str, dict[str, Any]]:
    """Compatibility wrapper; dispatches to the selected provider."""
    return call_llm_chat(**kwargs)


def parse_score_csv(text: str) -> pd.DataFrame:
    cleaned = text.strip()
    fence = re.search(r"```(?:csv)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    header_idx = cleaned.find("GENE,")
    if header_idx > 0:
        cleaned = cleaned[header_idx:]
    rows = []
    for row in csv.reader(StringIO(cleaned)):
        if not row or row[0].strip() == "GENE":
            continue
        if len(row) < 6:
            continue
        fixed = [cell.strip() for cell in row[:5]]
        if len(row) == 6:
            rationale = row[5].strip()
            caveats = ""
        else:
            rationale = ",".join(row[5:-1]).strip()
            caveats = row[-1].strip()
        rows.append(fixed + [rationale, caveats])
    if not rows:
        raise ValueError("No score rows found in LLM response")
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def validate_scores(df: pd.DataFrame, expected_genes: list[str]) -> None:
    missing = [column for column in SCORE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns: {missing}")
    observed = {gene.upper() for gene in df["GENE"].astype(str)}
    expected = {gene.upper() for gene in expected_genes}
    if observed != expected:
        raise ValueError(f"Score gene mismatch: expected={sorted(expected)} observed={sorted(observed)}")
    if not df["method"].astype(str).isin(SCORING_METHODS).all():
        raise ValueError(f"method must be one of {SCORING_METHODS}")
    values = pd.to_numeric(df["score"], errors="coerce")
    if values.isna().any() or ((values < 0) | (values > 1)).any():
        raise ValueError("score must contain numeric values in [0, 1]")


def filter_expected_scores(df: pd.DataFrame, expected_genes: list[str]) -> pd.DataFrame:
    expected = {gene.upper() for gene in expected_genes}
    work = df.copy()
    work["_gene_upper"] = work["GENE"].astype(str).str.upper()
    missing = sorted(expected - set(work["_gene_upper"]))
    if missing:
        raise ValueError(f"Score gene mismatch: missing expected genes={missing}")
    work = work[work["_gene_upper"].isin(expected)].copy()
    return work.drop(columns=["_gene_upper"])


def order_scores(df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    work = df.drop_duplicates("GENE", keep="last").copy()
    order = {gene.upper(): idx for idx, gene in enumerate(genes)}
    work["_order"] = work["GENE"].astype(str).str.upper().map(order)
    return work.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def read_api_key(api_key: str | None, api_key_file: str | None, provider: str | None = None) -> str:
    return resolve_api_key(api_key, api_key_file, provider)


def _method_rule(method: str) -> str:
    if method == "distilled_plus_prior":
        return (
            "Use the supplied distilled evidence packet as the auditable literature source, "
            "and also use your own established biomedical prior knowledge. The score is the "
            "final combined distilled-evidence-plus-prior priority score."
        )
    if method == "biomedbert_plus_prior":
        return (
            "Use the supplied distilled evidence packet as the auditable literature source, "
            "and also use your own established biomedical prior knowledge. The packet was "
            "generated with a BiomedBERT-based reranking distiller, so compare it to the "
            "other distilled-evidence-plus-prior condition on equal scoring terms."
        )
    if method == "raw_abstract_plus_prior":
        return (
            "Use the supplied raw PubMed evidence text as the auditable literature source, "
            "and also use your own established biomedical prior knowledge. The score is the "
            "final raw-evidence-plus-prior priority score."
        )
    if method == "distilled_only_no_prior":
        return (
            "Use only the supplied distilled evidence packet. Do not use model prior knowledge "
            "that is not present in the packet. The score is an evidence-only support score."
        )
    if method == "prior_only":
        return (
            "Use only your own established biomedical prior knowledge and the supplied biological "
            "context. Ignore the evidence packet for scoring and leave supporting_pmids empty."
        )
    raise ValueError(f"Unknown method: {method}")


def _context_pubmed_query(context: str) -> str:
    terms = [term for term in re.split(r"[^A-Za-z0-9-]+", context) if len(term) >= 4]
    stop = {"cell", "cells", "gene", "genes", "with", "from", "this", "that", "using", "context"}
    useful = [term for term in terms if term.lower() not in stop][:12]
    if not useful:
        return context
    return " OR ".join(f'"{term}"[Title/Abstract]' for term in useful)
