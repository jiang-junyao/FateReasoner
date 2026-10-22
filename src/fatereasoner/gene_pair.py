from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from .distillation import EvidenceDistillationConfig, EvidenceDistiller
from .llm import PROVIDERS
from .distilled_prior import SCORING_METHODS, _method_rule, call_llm, read_api_key, write_jsonl
from .evidence import EvidenceSelectionConfig, select_evidence_documents
from .models import EvidenceDocument
from .retrieval import JsonlRetrievalCache, PubMedClient, _gene_pubmed_query, parse_exclude_pmids


GENE_PAIR_SCORE_COLUMNS = [
    "candidate_id",
    "source_gene",
    "target_gene",
    "method",
    "fatereasoner_score",
    "evidence_level",
    "supporting_pmids",
    "rationale",
    "caveats",
    "relationship_type",
    "context_match",
    "evidence_strength",
]

GENE_PAIR_LEGACY_SCORE_COLUMNS = GENE_PAIR_SCORE_COLUMNS[:9]


def add_score_gene_pairs_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "score-gene-pairs",
        help="Score directed gene pairs (TF -> target, ligand -> receptor) with distilled evidence plus LLM prior.",
    )
    parser.add_argument("--pairs", required=True, help="CSV/TSV table containing source and target genes.")
    parser.add_argument("--context", required=True, help="Biological context, or path to a text file.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--source-col", default="source_gene")
    parser.add_argument("--target-col", default="target_gene")
    parser.add_argument("--candidate-id-col", default="candidate_id")
    parser.add_argument("--top-n", type=int, default=0, help="0 keeps all input rows after sorting.")
    parser.add_argument("--sort-col", default="data_score", help="Column used before --top-n, if present.")
    parser.add_argument("--method", choices=SCORING_METHODS, default="distilled_plus_prior")
    parser.add_argument("--retmax", type=int, default=30)
    parser.add_argument(
        "--retrieval-policy",
        choices=["legacy", "pair_cascade"],
        default="legacy",
        help="PubMed retrieval policy; pair_cascade adds broader TF-target fallback queries.",
    )
    parser.add_argument("--date-cutoff", default=None)
    parser.add_argument("--delay-seconds", type=float, default=0.34)
    parser.add_argument(
        "--exclude-pmids",
        default="",
        help="Comma/semicolon separated PubMed IDs to exclude from retrieval and evidence packets.",
    )
    parser.add_argument("--evidence-threshold", type=float, default=0.25)
    parser.add_argument("--evidence-min-keep", type=int, default=5)
    parser.add_argument("--evidence-max-keep", type=int, default=20)
    parser.add_argument(
        "--distill-evidence",
        choices=["lite_weightfree", "lite_weighted", "lite", "medcpt_go", "biomedbert_rerank"],
        default="lite_weightfree",
    )
    parser.add_argument("--distilled-max-chars", type=int, default=3200)
    parser.add_argument("--distilled-packet-style", choices=["audit", "concise"], default="concise")
    parser.add_argument("--distill-max-windows", type=int, default=8)
    parser.add_argument("--distilled-anchor-docs", type=int, default=3)
    parser.add_argument("--distilled-anchor-chars", type=int, default=650)
    parser.add_argument("--anchor-strategy", choices=["balanced", "diverse_preserve"], default="diverse_preserve")
    parser.add_argument("--include-go", dest="include_go", action="store_true", default=False)
    parser.add_argument("--no-go", dest="include_go", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--medcpt-model", default="ncbi/MedCPT-Cross-Encoder")
    parser.add_argument("--biomedbert-model", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--provider", choices=PROVIDERS, default=None, help="LLM provider (required unless --prepare-only).")
    parser.add_argument("--model", default=None, help="Provider model ID; defaults to the provider preset.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--request-batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(func=run_score_gene_pairs)


def run_score_gene_pairs(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = read_text_arg(args.context)
    candidates = read_pair_candidates(args.pairs, args)
    candidates.to_csv(out_dir / "gene_pair_candidates.csv", index=False)
    (out_dir / "context.txt").write_text(context + "\n", encoding="utf-8")
    packets = build_gene_pair_distilled_packets(candidates, context, args, out_dir)
    packets_path = out_dir / f"{args.method}_gene_pair_packets.jsonl"
    write_jsonl(packets_path, packets)
    packet_table = gene_pair_packets_to_table(packets)
    packet_table.to_csv(out_dir / "distilled_gene_pair_packets.csv", index=False)
    if args.prepare_only:
        print(f"gene_pairs={len(candidates)} packets={packets_path}")
        print(f"prepare_only=true out={out_dir}")
        return
    score_path = out_dir / f"{args.method}_gene_pair_scores.csv"
    scores = score_gene_pair_packets_with_llm(packets, args, score_path)
    result = merge_gene_pair_scores(candidates, scores)
    result.to_csv(out_dir / "gene_pair_scores.csv", index=False)
    print(f"gene_pairs={len(candidates)} scores={score_path}")
    print(f"result={out_dir / 'gene_pair_scores.csv'}")


def read_pair_candidates(path: str | Path, args: argparse.Namespace) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    raw = pd.read_csv(path, sep=sep)
    return normalize_pair_candidates(
        raw,
        source_col=args.source_col,
        target_col=args.target_col,
        candidate_id_col=args.candidate_id_col,
        top_n=args.top_n,
        sort_col=args.sort_col,
    )


def normalize_pair_candidates(
    raw: pd.DataFrame,
    *,
    source_col: str = "source_gene",
    target_col: str = "target_gene",
    candidate_id_col: str = "candidate_id",
    top_n: int = 0,
    sort_col: str = "data_score",
) -> pd.DataFrame:
    missing = [column for column in (source_col, target_col) if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing gene-pair columns: {missing}")
    out = raw.copy()
    out["source_gene"] = out[source_col].astype(str).str.strip()
    out["target_gene"] = out[target_col].astype(str).str.strip()
    out["relation_type"] = out.apply(_pair_relation_type, axis=1)
    if candidate_id_col in out.columns:
        out["candidate_id"] = out[candidate_id_col].astype(str).str.strip()
    elif "candidate_id" not in out.columns:
        out["candidate_id"] = out["source_gene"] + "->" + out["target_gene"]
    out = out[(out["source_gene"] != "") & (out["target_gene"] != "")].copy()
    out = out[out["source_gene"].str.lower().ne("nan") & out["target_gene"].str.lower().ne("nan")].copy()
    out = out.drop_duplicates("candidate_id", keep="first")
    if "data_score" not in out.columns:
        out["data_score"] = _pair_data_score(out)
    out["data_score"] = pd.to_numeric(out["data_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)
    if top_n and top_n > 0:
        out = out.head(top_n)
    return out.reset_index(drop=True)


def _optional_text(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _pair_relation_type(row: dict[str, Any] | pd.Series) -> str:
    explicit = _optional_text(row, "relation_type", "candidate_type").lower().replace("-", "_")
    if explicit in {"ligand_receptor", "ligandreceptor", "lr", "cell_cell_communication"}:
        return "ligand_receptor"
    if explicit in {"tf_target", "tftarget", "grn", "regulatory", "gene_regulatory"}:
        return "tf_target"
    if any(_optional_text(row, name) for name in ("ligand", "receptor", "sender_celltype", "receiver_celltype")):
        return "ligand_receptor"
    return "tf_target"


def _pair_celltypes(row: dict[str, Any]) -> tuple[str, str]:
    sender = _optional_text(row, "sender_celltype", "source_celltype", "source_cell_type")
    receiver = _optional_text(row, "receiver_celltype", "target_celltype", "target_cell_type")
    return sender, receiver


def _pair_data_score(df: pd.DataFrame) -> pd.Series:
    if "abs_correlation" in df.columns:
        return _minmax(pd.to_numeric(df["abs_correlation"], errors="coerce"))
    if "expression_correlation" in df.columns:
        return _minmax(pd.to_numeric(df["expression_correlation"], errors="coerce").abs())
    return pd.Series([1.0] * len(df), index=df.index)


def build_gene_pair_distilled_packets(
    candidates: pd.DataFrame,
    context: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict[str, Any]]:
    pubmed = GenePairPubMedClient(
        retmax=args.retmax,
        date_cutoff=args.date_cutoff,
        delay_seconds=args.delay_seconds,
        exclude_pmids=parse_exclude_pmids(getattr(args, "exclude_pmids", "")),
        retrieval_policy=getattr(args, "retrieval_policy", "legacy"),
    )
    cache = JsonlRetrievalCache(out_dir / "gene_pair_retrieval_cache.jsonl")
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
    # Persist each completed packet so long retrieval jobs can resume after an
    # interrupted session without re-distilling earlier candidates.
    packet_cache_path = out_dir / f"{args.method}_gene_pair_packets.partial.jsonl"
    cached_packets: dict[str, dict[str, Any]] = {}
    if args.resume and packet_cache_path.exists():
        with packet_cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    packet = json.loads(line)
                    candidate_id = str(packet.get("candidate_id", ""))
                    if candidate_id:
                        cached_packets[candidate_id] = packet
                except json.JSONDecodeError:
                    continue
    packets: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates.to_dict("records"), start=1):
        candidate_id = str(row["candidate_id"])
        if candidate_id in cached_packets:
            packets.append(cached_packets[candidate_id])
            continue
        docs = cache.get(candidate_id)
        if docs is None:
            docs = pubmed.search_and_fetch_pair(row, context)
            cache.set(candidate_id, docs)
        else:
            docs = [doc for doc in docs if doc.pmid not in pubmed.exclude_pmids]
        relation_context = gene_pair_relation_context(row, context)
        sender_celltype, receiver_celltype = _pair_celltypes(row)
        selection = select_evidence_documents(candidate_id, relation_context, docs, evidence_config)
        packet = distiller.distill(gene_pair_query_label(row), relation_context, selection.documents)
        record = {
            "candidate_id": candidate_id,
            "source_gene": str(row["source_gene"]),
            "target_gene": str(row["target_gene"]),
            "relation_type": _pair_relation_type(row),
            "sender_celltype": sender_celltype,
            "receiver_celltype": receiver_celltype,
            "method": args.method,
            "context": context,
            "relation_context": relation_context,
            "retrieved_count": selection.retrieved_count,
            "evidence_count": selection.evidence_count,
            "evidence_filter_method": selection.filter_method,
            "distilled_text": packet.distilled_text,
            "distilled_pmids": ";".join(packet.selected_pmids),
            "distilled_go_terms": ";".join(packet.go_terms),
            "distilled_token_estimate": packet.token_estimate,
            "distillation_method": packet.method,
            "network_summary": gene_pair_network_summary(row),
            "documents": [asdict(doc) for doc in selection.documents],
        }
        with packet_cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        cached_packets[candidate_id] = record
        packets.append(record)
        print(
            f"prepared {idx}/{len(candidates)} gene_pair={candidate_id} retrieved={selection.retrieved_count} evidence={selection.evidence_count}",
            flush=True,
        )
    return packets


class GenePairPubMedClient(PubMedClient):
    def __init__(self, *args: Any, retrieval_policy: str = "legacy", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.retrieval_policy = retrieval_policy

    def search_and_fetch_pair(self, candidate: dict[str, Any], context: str) -> list[EvidenceDocument]:
        if self.retrieval_policy == "legacy":
            ids = self.search(self.build_pair_query(candidate, context))
            if not ids:
                ids = self.search(self.build_fallback_query(candidate, context))
        else:
            ids = []
            seen: set[str] = set()
            queries = (
                self.build_pair_query(candidate, context),
                self.build_fallback_query(candidate, context),
                self.build_pair_only_query(candidate),
            )
            for query in queries:
                for pmid in self.search(query):
                    if pmid not in seen:
                        ids.append(pmid)
                        seen.add(pmid)
                if len(ids) >= self.retmax:
                    break
        docs = self.fetch(ids)
        for rank, doc in enumerate(docs, start=1):
            doc.pubmed_rank = rank
        return docs

    def build_pair_query(self, candidate: dict[str, Any], context: str) -> str:
        source = _lr_gene_pubmed_query(str(candidate["source_gene"]))
        target = _lr_gene_pubmed_query(str(candidate["target_gene"]))
        if _pair_relation_type(candidate) == "ligand_receptor":
            relation_terms = (
                '"ligand-receptor"[Title/Abstract] OR "ligand receptor"[Title/Abstract] OR '
                'interact*[Title/Abstract] OR signal*[Title/Abstract] OR communica*[Title/Abstract]'
            )
        else:
            relation_terms = (
                '"transcription factor"[Title/Abstract] OR regulat*[Title/Abstract] OR target[Title/Abstract] OR '
                '"gene regulatory network"[Title/Abstract]'
            )
        disease_terms, cell_terms = _spatial_lr_retrieval_terms(candidate, context)
        relation_clause = f" AND ({relation_terms})"
        disease_clause = f" AND ({disease_terms})" if disease_terms else ""
        cell_clause = f" AND ({cell_terms})" if cell_terms else ""
        return f"(({source}) AND ({target}){relation_clause}{disease_clause}{cell_clause})"

    def build_fallback_query(self, candidate: dict[str, Any], context: str) -> str:
        source = _lr_gene_pubmed_query(str(candidate["source_gene"]))
        target = _lr_gene_pubmed_query(str(candidate["target_gene"]))
        if _pair_relation_type(candidate) == "ligand_receptor":
            relation_terms = '"ligand-receptor"[Title/Abstract] OR interact*[Title/Abstract] OR signal*[Title/Abstract]'
        else:
            relation_terms = 'regulat*[Title/Abstract] OR target[Title/Abstract] OR bind*[Title/Abstract]'
        disease_terms, _ = _spatial_lr_retrieval_terms(candidate, context)
        disease_clause = f" AND ({disease_terms})" if disease_terms else ""
        return f"(({source}) AND ({target}) AND ({relation_terms}){disease_clause})"

    def build_pair_only_query(self, candidate: dict[str, Any]) -> str:
        source = _lr_gene_pubmed_query(str(candidate["source_gene"]))
        target = _lr_gene_pubmed_query(str(candidate["target_gene"]))
        if _pair_relation_type(candidate) == "ligand_receptor":
            relation_terms = (
                '"ligand-receptor"[Title/Abstract] OR signal*[Title/Abstract] OR '
                'interact*[Title/Abstract] OR communica*[Title/Abstract]'
            )
        else:
            relation_terms = (
                'regulat*[Title/Abstract] OR target[Title/Abstract] OR '
                'bind*[Title/Abstract] OR express*[Title/Abstract] OR '
                'interact*[Title/Abstract]'
            )
        return f"(({source}) AND ({target}) AND ({relation_terms}))"


# These expansions are intentionally small and only used for ligand-receptor
# retrieval.  They address well-known protein-name aliases without altering the
# candidate identity or its final FateReasoner score.
LR_PUBMED_ALIASES = {
    "APOE": ("apolipoprotein E",),
    "LSR": ("angulin-1", "lipolysis-stimulated lipoprotein receptor"),
    "MIF": ("macrophage migration inhibitory factor",),
    "SELENOP": ("selenoprotein P",),
    "SPP1": ("osteopontin",),
}


def _lr_gene_pubmed_query(gene: str) -> str:
    gene = str(gene).strip()
    query = _gene_pubmed_query(gene)
    # CellChat-style receptor complexes are encoded with underscores (for
    # example ITGA3_ITGB1), whereas papers normally name their subunits
    # separately.  Search both representations, requiring all subunits in
    # the latter form so that a complex is not treated as either subunit alone.
    components = [part for part in gene.split("_") if part]
    if len(components) > 1:
        component_query = " AND ".join(f'"{part}"[Title/Abstract]' for part in components)
        query = f"(({query}) OR ({component_query}))"
    aliases = LR_PUBMED_ALIASES.get(gene.upper(), ())
    if not aliases:
        return query
    alias_query = " OR ".join(f'"{alias}"[Title/Abstract]' for alias in aliases)
    return f"(({query}) OR ({alias_query}))"


def _spatial_lr_retrieval_terms(candidate: dict[str, Any], context: str) -> tuple[str, str]:
    """Return literature-standard disease and broad cell terms for spatial LR retrieval.

    Dataset-local labels (for example, ``Tumor III`` and ``Macrophage-SPP1+``)
    are deliberately not queried verbatim.  They remain in the packet and LLM
    prompt, where their instance-level meaning can be evaluated.
    """
    if _pair_relation_type(candidate) != "ligand_receptor":
        return "", ""
    text = f"{context} {_optional_text(candidate, 'sender_celltype')} {_optional_text(candidate, 'receiver_celltype')}".lower()
    if "colorectal" in text or re.search(r"\bcrc\b", text):
        disease = '"colorectal cancer"[Title/Abstract] OR "colon cancer"[Title/Abstract] OR "rectal cancer"[Title/Abstract]'
    elif "oral squamous" in text or re.search(r"\boscc\b", text):
        disease = '"oral squamous cell carcinoma"[Title/Abstract] OR "head and neck squamous cell carcinoma"[Title/Abstract]'
    else:
        disease = ""

    broad_cells: list[str] = []
    if "macrophage" in text:
        broad_cells.append('macrophage[Title/Abstract]')
    if any(term in text for term in ("mycaf", "caf", "fibroblast")):
        broad_cells.extend(['"cancer-associated fibroblast"[Title/Abstract]', 'fibroblast[Title/Abstract]', 'myofibroblast[Title/Abstract]'])
    if any(term in text for term in ("tumor", "tumour", "edge")):
        broad_cells.extend(['tumor[Title/Abstract]', 'cancer[Title/Abstract]', 'carcinoma[Title/Abstract]'])
    if "t cell" in text:
        broad_cells.append('"T cell"[Title/Abstract]')
    return disease, " OR ".join(dict.fromkeys(broad_cells))


def score_gene_pair_packets_with_llm(packets: list[dict[str, Any]], args: argparse.Namespace, out: Path) -> pd.DataFrame:
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
        prompt = gene_pair_scoring_prompt(batch)
        expected_ids = [str(packet["candidate_id"]) for packet in batch]
        batch_scores = None
        for validation_attempt in range(max(1, args.max_retries)):
            request_prompt = prompt
            if validation_attempt:
                request_prompt += (
                    "\nYour previous response failed validation. Return exactly one CSV row and "
                    f"use this exact candidate_id: {expected_ids[0]}. Do not substitute a numeric id."
                )
            response_text, response_data = call_llm(
                prompt=request_prompt,
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
                handle.write(f"--- RESPONSE {batch_idx} ATTEMPT {validation_attempt + 1} ---\n{response_text}\n\n")
            with api_response_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(response_data, ensure_ascii=False) + "\n")
            try:
                batch_scores = parse_gene_pair_score_csv(response_text)
                validate_gene_pair_scores(batch_scores, expected_ids)
                break
            except ValueError:
                if validation_attempt == max(1, args.max_retries) - 1:
                    raise
        assert batch_scores is not None
        score_frames.append(batch_scores)
        scores = order_gene_pair_scores(pd.concat(score_frames, ignore_index=True), [str(packet["candidate_id"]) for packet in packets])
        scores.to_csv(out, index=False)
        print(f"scored_batch={batch_idx} rows={len(batch_scores)} total={len(scores)}", flush=True)
    scores = (
        order_gene_pair_scores(pd.concat(score_frames, ignore_index=True), [str(packet["candidate_id"]) for packet in packets])
        if score_frames
        else pd.DataFrame(columns=GENE_PAIR_SCORE_COLUMNS)
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
                "retrieval_policy": getattr(args, "retrieval_policy", "legacy"),
                "n_packets": len(packets),
                "score_columns": GENE_PAIR_SCORE_COLUMNS,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return scores


def gene_pair_scoring_prompt(packets: list[dict[str, Any]]) -> str:
    method = str(packets[0]["method"]) if packets else "distilled_plus_prior"
    relation_types = {_pair_relation_type(packet) for packet in packets}
    ligand_receptor_mode = relation_types == {"ligand_receptor"}
    if ligand_receptor_mode:
        relation_intro = "Score each ligand -> receptor candidate in the supplied biological context using this method: {method}."
        relation_eval = (
            "Evaluate the specific ligand-receptor relationship, including known receptor binding/signaling, "
            "biological plausibility, and—when provided—sender-to-receiver cell-type direction."
        )
        relation_caveat = (
            "Sender and receiver cell types are optional. If absent, do not infer a cellular source, target, "
            "or communication direction; evaluate the ligand-receptor pair in the supplied context only."
        )
        direct_rule = "Do not label a pair direct_ligand_receptor unless the packet supports a ligand-receptor relationship."
    else:
        relation_intro = "Score each source TF -> target gene candidate in the supplied biological context using this method: {method}."
        relation_eval = (
            "Evaluate the specific source TF -> target gene relationship, not merely whether the two genes are individually relevant."
        )
        relation_caveat = "Source and target cell types are optional metadata; do not infer them when absent."
        direct_rule = "Do not label a pair direct_regulation unless the packet supports regulation of the target by the source TF."
    biological_eval = (
        "Evaluate the biological plausibility of the TF-target candidate."
        if not ligand_receptor_mode
        else "Evaluate the biological plausibility of the ligand-receptor candidate."
    )
    lines = [
        "You are the FateReasoner biomedical scoring judge.",
        relation_intro.format(method=method),
        "",
        _method_rule(method),
        "Use both the distilled literature evidence packet and established biomedical prior knowledge.",
        "A missing PMID does not force the score to zero, because prior knowledge may support the candidate; do not invent PMIDs.",
        biological_eval,
        relation_eval,
        relation_caveat,
        "Consider direct interaction, functional interaction, relevant co-expression or pathway biology, and established prior knowledge.",
        "Do not use network data metrics such as data_score, expression correlation, motif counts, footprint scores, FOS, centrality, or other GRN features for fatereasoner_score.",
        "Penalize unrelated species or tissues, generic cancer/expression evidence, weak target relevance, and symbol ambiguity.",
        direct_rule,
        "Cite only PMIDs visible in the packet in supporting_pmids. Mention prior knowledge in the rationale without inventing PMIDs.",
        "Add evidence annotations without changing the numeric score:",
        "relationship_type: direct_regulation, direct_ligand_receptor, binding_only, functional_indirect, coexpression_only, pathway_only, or no_pair_evidence.",
        "context_match must be one of strong_context, relevant_context, indirect_context, unrelated, or unclear.",
        "evidence_strength must be one of strong, moderate, weak, or none.",
        "",
        "Return CSV only, with exactly these columns:",
        ",".join(GENE_PAIR_SCORE_COLUMNS),
        "",
        "fatereasoner_score is a single number from 0.0 to 1.0.",
        "method must exactly match the requested method.",
        "The three annotation columns describe evidence and do not replace or rescale fatereasoner_score.",
        f"Return exactly {len(packets)} data rows, one row for each candidate.",
        "",
    ]
    for packet in packets:
        lines.extend(
            [
                f"## CANDIDATE: {packet['candidate_id']}",
                f"Relation type: {_pair_relation_type(packet)}",
                f"Source/ligand gene: {packet['source_gene']}",
                f"Target/receptor gene: {packet['target_gene']}",
                f"Sender/source cell type: {packet.get('sender_celltype', '') or 'not provided'}",
                f"Receiver/target cell type: {packet.get('receiver_celltype', '') or 'not provided'}",
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


def parse_gene_pair_score_csv(text: str) -> pd.DataFrame:
    cleaned = text.strip()
    fence = re.search(r"```(?:csv)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    header_idx = cleaned.find("candidate_id,")
    if header_idx > 0:
        cleaned = cleaned[header_idx:]
    header_line = next((line for line in cleaned.splitlines() if line.strip()), "")
    structured_header = "relationship_type" in header_line
    rows = []
    for row in csv.reader(StringIO(cleaned)):
        if not row or row[0].strip() == "candidate_id":
            continue
        if len(row) < 8:
            continue
        fixed = [cell.strip() for cell in row[:7]]
        if structured_header and len(row) >= len(GENE_PAIR_SCORE_COLUMNS):
            relationship_type = row[-3].strip()
            context_match = row[-2].strip()
            evidence_strength = row[-1].strip()
            rationale = ",".join(row[7:-4]).strip()
            caveats = row[-4].strip()
        elif len(row) == 8:
            rationale = row[7].strip()
            caveats = ""
            relationship_type = ""
            context_match = ""
            evidence_strength = ""
        else:
            rationale = ",".join(row[7:-1]).strip()
            caveats = row[-1].strip()
            relationship_type = ""
            context_match = ""
            evidence_strength = ""
        rows.append(fixed + [rationale, caveats, relationship_type, context_match, evidence_strength])
    if not rows:
        raise ValueError("No gene-pair score rows found in LLM response")
    return pd.DataFrame(rows, columns=GENE_PAIR_SCORE_COLUMNS)


def validate_gene_pair_scores(df: pd.DataFrame, expected_ids: list[str]) -> None:
    missing = [column for column in GENE_PAIR_SCORE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns: {missing}")
    observed = {candidate.upper() for candidate in df["candidate_id"].astype(str)}
    expected = {candidate.upper() for candidate in expected_ids}
    if observed != expected:
        raise ValueError(f"Gene-pair candidate mismatch: expected={sorted(expected)} observed={sorted(observed)}")
    if not df["method"].astype(str).isin(SCORING_METHODS).all():
        raise ValueError(f"method must be one of {SCORING_METHODS}")
    values = pd.to_numeric(df["fatereasoner_score"], errors="coerce")
    if values.isna().any() or ((values < 0) | (values > 1)).any():
        raise ValueError("fatereasoner_score must contain numeric values in [0, 1]")


def order_gene_pair_scores(df: pd.DataFrame, candidate_ids: list[str]) -> pd.DataFrame:
    work = df.drop_duplicates("candidate_id", keep="last").copy()
    order = {candidate.upper(): idx for idx, candidate in enumerate(candidate_ids)}
    work["_order"] = work["candidate_id"].astype(str).str.upper().map(order)
    return work.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def merge_gene_pair_scores(candidates: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    candidate_frame = candidates.copy()
    score_frame = scores.copy()
    candidate_frame["_candidate_key"] = candidate_frame["candidate_id"].astype(str).str.upper()
    score_frame["_candidate_key"] = score_frame["candidate_id"].astype(str).str.upper()
    merged = candidate_frame.merge(score_frame, on="_candidate_key", how="left", suffixes=("", "_score"))
    merged = merged.drop(columns=["_candidate_key"])
    merged["fatereasoner_score"] = pd.to_numeric(merged["fatereasoner_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    merged["evidence_quadrant"] = merged.apply(_evidence_quadrant, axis=1)
    merged["fatereasoner_rank"] = merged["fatereasoner_score"].rank(method="first", ascending=False).astype(int)
    merged["data_rank"] = merged["data_score"].rank(method="first", ascending=False).astype(int)
    return merged.sort_values(["fatereasoner_score", "data_score", "candidate_id"], ascending=[False, False, True]).reset_index(drop=True)


def gene_pair_relation_context(row: dict[str, Any], context: str) -> str:
    relation_type = _pair_relation_type(row)
    sender_celltype, receiver_celltype = _pair_celltypes(row)
    if relation_type == "ligand_receptor":
        relation = f"candidate ligand-receptor relation {row['source_gene']} -> {row['target_gene']}"
    else:
        relation = f"candidate TF-target relation {row['source_gene']} -> {row['target_gene']}"
    parts = [
        context,
        relation,
    ]
    if sender_celltype:
        parts.append(f"sender/source cell type {sender_celltype}")
    if receiver_celltype:
        parts.append(f"receiver/target cell type {receiver_celltype}")
    if row.get("species"):
        parts.append(f"species {row['species']}")
    if row.get("source_module") and row.get("target_module"):
        parts.append(f"GRN modules {row['source_module']} -> {row['target_module']}")
    return "; ".join(parts)


def gene_pair_query_label(row: dict[str, Any]) -> str:
    sender_celltype, receiver_celltype = _pair_celltypes(row)
    return " ".join(
        value
        for value in [row["source_gene"], row["target_gene"], sender_celltype, receiver_celltype]
        if value
    )


def gene_pair_network_summary(row: dict[str, Any]) -> str:
    fields = [
        "candidate_id",
        "species",
        "source_gene",
        "target_gene",
        "relation_type",
        "sender_celltype",
        "receiver_celltype",
        "source_module",
        "target_module",
        "expression_correlation",
        "abs_correlation",
        "motif_locus_count",
        "promoter_locus_fraction",
        "regulation_type",
        "data_score",
    ]
    return "\n".join(f"{field}: {row.get(field, '')}" for field in fields if field in row)


def gene_pair_packets_to_table(packets: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{key: value for key, value in packet.items() if key != "documents"} for packet in packets])


def read_text_arg(value: str) -> str:
    path = Path(value)
    if path.exists():
        return " ".join(path.read_text(encoding="utf-8").split())
    return " ".join(value.split())


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
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    lo = values.min()
    hi = values.max()
    if float(lo) == float(hi):
        return pd.Series([1.0] * len(series), index=series.index)
    return ((values - lo) / (hi - lo)).fillna(0.0)
