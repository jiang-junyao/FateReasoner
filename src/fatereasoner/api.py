from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Literal

import pandas as pd

from .distilled_prior import (
    SCORING_METHODS,
    build_distilled_packets,
    ensure_unique_genes,
    load_precomputed_packets,
    packets_to_table as gene_packets_to_table,
    read_context,
    read_gene_list,
    score_packets_with_llm,
    write_jsonl,
    _unique_genes,
)
from .grn import (
    build_module_distilled_packets,
    build_module_pair_candidates,
    load_network,
    merge_module_scores,
    packets_to_table as module_packets_to_table,
    read_text_arg,
    score_module_packets_with_llm,
)
from .gene_pair import (
    build_gene_pair_distilled_packets,
    gene_pair_packets_to_table,
    merge_gene_pair_scores,
    normalize_pair_candidates,
    read_text_arg as read_gene_pair_context,
    score_gene_pair_packets_with_llm,
)


ScoringMethod = Literal[
    "distilled_plus_prior",
    "biomedbert_plus_prior",
    "distilled_only_no_prior",
    "prior_only",
    "raw_abstract_plus_prior",
]
DistillationMethod = Literal["lite_weightfree", "lite_weighted", "lite", "medcpt_go", "biomedbert_rerank"]
GRNSpecies = str  # "mouse", "zebrafish" (IReNA formats) or any label for the generic format


@dataclass(frozen=True)
class GeneScoringConfig:
    precomputed_packets: str | None = None
    method: ScoringMethod = "distilled_plus_prior"
    retmax: int = 30
    full_text: Literal["none", "pmc-oa"] = "pmc-oa"
    date_cutoff: str | None = None
    delay_seconds: float = 0.34
    exclude_pmids: str | tuple[str, ...] = ""
    evidence_threshold: float = 0.35
    evidence_min_keep: int = 5
    evidence_max_keep: int = 20
    distill_evidence: DistillationMethod = "lite_weightfree"
    distilled_max_chars: int = 3000
    distilled_packet_style: Literal["audit", "concise"] = "concise"
    distill_max_windows: int = 6
    distilled_anchor_docs: int = 2
    distilled_anchor_chars: int = 520
    anchor_strategy: str = "diverse_preserve"
    include_go: bool = False
    medcpt_model: str = "ncbi/MedCPT-Cross-Encoder"
    biomedbert_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    provider: Literal["deepseek", "gpt", "claude", "gemini", "glm", "qwen"] | None = None
    model: str | None = None
    api_key: str | None = None
    api_key_file: str | None = None
    base_url: str | None = None
    thinking: Literal["enabled", "disabled"] = "disabled"
    temperature: float = 0.0
    max_output_tokens: int = 8000
    request_batch_size: int = 1
    timeout: float = 120.0
    max_retries: int = 3
    prepare_only: bool = False
    resume: bool = False


@dataclass(frozen=True)
class GRNModuleScoringConfig:
    top_module_pairs: int = 20
    per_gene_retmax: int = 8
    per_pair_retmax: int = 3
    module_gene_block_size: int = 8
    module_block_retmax: int = 35
    module_evidence_min_docs: int = 20
    module_evidence_max_docs: int = 30
    method: ScoringMethod = "distilled_plus_prior"
    retmax: int = 30
    full_text: Literal["none", "pmc-oa"] = "pmc-oa"
    date_cutoff: str | None = None
    delay_seconds: float = 0.34
    retrieval_timeout: int = 12
    exclude_pmids: str | tuple[str, ...] = ""
    evidence_threshold: float = 0.25
    evidence_min_keep: int = 5
    evidence_max_keep: int = 20
    distill_evidence: DistillationMethod = "lite_weightfree"
    distilled_max_chars: int = 3600
    distilled_packet_style: Literal["audit", "concise"] = "concise"
    distill_max_windows: int = 8
    distilled_anchor_docs: int = 3
    distilled_anchor_chars: int = 650
    anchor_strategy: str = "diverse_preserve"
    include_go: bool = False
    medcpt_model: str = "ncbi/MedCPT-Cross-Encoder"
    biomedbert_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    provider: Literal["deepseek", "gpt", "claude", "gemini", "glm", "qwen"] | None = None
    model: str | None = None
    api_key: str | None = None
    api_key_file: str | None = None
    base_url: str | None = None
    thinking: Literal["enabled", "disabled"] = "disabled"
    temperature: float = 0.0
    max_output_tokens: int = 8000
    request_batch_size: int = 1
    timeout: float = 120.0
    max_retries: int = 3
    prepare_only: bool = False
    resume: bool = False


@dataclass(frozen=True)
class GenePairScoringConfig:
    top_n: int = 0
    sort_col: str = "data_score"
    method: ScoringMethod = "distilled_plus_prior"
    retmax: int = 30
    full_text: Literal["none", "pmc-oa"] = "pmc-oa"
    date_cutoff: str | None = None
    delay_seconds: float = 0.34
    exclude_pmids: str | tuple[str, ...] = ""
    evidence_threshold: float = 0.25
    evidence_min_keep: int = 5
    evidence_max_keep: int = 20
    distill_evidence: DistillationMethod = "lite_weightfree"
    distilled_max_chars: int = 3200
    distilled_packet_style: Literal["audit", "concise"] = "concise"
    distill_max_windows: int = 8
    distilled_anchor_docs: int = 3
    distilled_anchor_chars: int = 650
    anchor_strategy: str = "diverse_preserve"
    include_go: bool = False
    medcpt_model: str = "ncbi/MedCPT-Cross-Encoder"
    biomedbert_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    provider: Literal["deepseek", "gpt", "claude", "gemini", "glm", "qwen"] | None = None
    model: str | None = None
    api_key: str | None = None
    api_key_file: str | None = None
    base_url: str | None = None
    thinking: Literal["enabled", "disabled"] = "disabled"
    temperature: float = 0.0
    max_output_tokens: int = 8000
    request_batch_size: int = 1
    timeout: float = 120.0
    max_retries: int = 3
    prepare_only: bool = False
    resume: bool = False


@dataclass(frozen=True)
class ScoringResult:
    out_dir: Path
    candidates_path: Path | None
    packets_path: Path
    packet_table_path: Path
    scores_path: Path | None
    result_path: Path | None
    candidates: pd.DataFrame | None
    packets: list[dict]
    packet_table: pd.DataFrame
    scores: pd.DataFrame | None
    result: pd.DataFrame | None


def score_genes(
    genes: str | Path | Iterable[str] | None,
    context: str | Path | None,
    out: str | Path,
    *,
    gene_col: str | None = None,
    config: GeneScoringConfig | None = None,
) -> ScoringResult:
    """Score gene candidates with FateReasoner's distilled-evidence LLM judge.

    Writes the same artifacts as the ``fatereasoner score-genes`` CLI and returns
    the in-memory tables for programmatic use.
    """
    config = config or GeneScoringConfig()
    _validate_method(config.method)
    out_dir = _ensure_dir(out)
    args = _namespace(config, out=str(out_dir), genes=genes, gene_col=gene_col)

    if config.precomputed_packets:
        packets = load_precomputed_packets(Path(config.precomputed_packets), config.method)
        gene_list = [str(packet["GENE"]) for packet in packets]
        context_values = [str(packet.get("context", "")) for packet in packets if packet.get("context")]
        context_text = read_context(str(context)) if context is not None else (context_values[0] if context_values else "")
    else:
        if genes is None:
            raise ValueError("genes is required unless config.precomputed_packets is provided.")
        if context is None:
            raise ValueError("context is required unless config.precomputed_packets is provided.")
        gene_list = _coerce_genes(genes, gene_col)
        if not gene_list:
            raise ValueError("No genes found in input.")
        context_text = read_context(str(context))
        packets = build_distilled_packets(gene_list, context_text, args, out_dir)
    if not packets:
        raise ValueError("No packets available for scoring.")
    ensure_unique_genes(packets)

    (out_dir / "context.txt").write_text(context_text + "\n", encoding="utf-8")
    pd.DataFrame({"GENE": gene_list}).to_csv(out_dir / "input_genes.csv", index=False)

    packets_path = out_dir / f"{config.method}_packets.jsonl"
    write_jsonl(packets_path, packets)
    packet_table = gene_packets_to_table(packets)
    packet_table_path = out_dir / "distilled_evidence_packets.csv"
    packet_table.to_csv(packet_table_path, index=False)

    if config.prepare_only:
        return ScoringResult(out_dir, None, packets_path, packet_table_path, None, None, None, packets, packet_table, None, None)

    scores_path = out_dir / f"{config.method}_scores.csv"
    scores = score_packets_with_llm(packets, args, scores_path)
    result = packet_table[["GENE", "context", "retrieved_count", "evidence_count", "distilled_pmids"]].merge(scores, on="GENE", how="left")
    result_path = out_dir / "gene_scores.csv"
    result.to_csv(result_path, index=False)
    return ScoringResult(out_dir, None, packets_path, packet_table_path, scores_path, result_path, None, packets, packet_table, scores, result)


def score_grn_modules(
    network: str | Path,
    species: GRNSpecies,
    context: str | Path,
    out: str | Path,
    *,
    config: GRNModuleScoringConfig | None = None,
) -> ScoringResult:
    """Score module -> module candidates from a gene regulatory network.

    The FateReasoner score is generated from distilled literature evidence and
    allowed prior knowledge only. GRN metrics remain in the output table as an
    independent data axis.
    """
    config = config or GRNModuleScoringConfig()
    _validate_method(config.method)
    out_dir = _ensure_dir(out)
    context_text = read_text_arg(str(context))
    args = _namespace(config, network=str(network), species=species, context=str(context), out=str(out_dir))

    edges = load_network(network, species)
    candidates = build_module_pair_candidates(edges).head(config.top_module_pairs).copy()
    candidates_path = out_dir / "module_pair_candidates.csv"
    candidates.to_csv(candidates_path, index=False)
    (out_dir / "context.txt").write_text(context_text + "\n", encoding="utf-8")

    packets = build_module_distilled_packets(candidates, context_text, args, out_dir)
    packets_path = out_dir / f"{config.method}_module_packets.jsonl"
    write_jsonl(packets_path, packets)
    packet_table = module_packets_to_table(packets)
    packet_table_path = out_dir / "distilled_module_evidence_packets.csv"
    packet_table.to_csv(packet_table_path, index=False)

    if config.prepare_only:
        return ScoringResult(out_dir, candidates_path, packets_path, packet_table_path, None, None, candidates, packets, packet_table, None, None)

    scores_path = out_dir / f"{config.method}_module_scores.csv"
    scores = score_module_packets_with_llm(packets, args, scores_path)
    result = merge_module_scores(candidates, scores)
    result_path = out_dir / "module_pair_scores.csv"
    result.to_csv(result_path, index=False)
    return ScoringResult(out_dir, candidates_path, packets_path, packet_table_path, scores_path, result_path, candidates, packets, packet_table, scores, result)


def score_gene_pairs(
    pairs: str | Path | pd.DataFrame,
    context: str | Path,
    out: str | Path,
    *,
    source_col: str = "source_gene",
    target_col: str = "target_gene",
    candidate_id_col: str = "candidate_id",
    config: GenePairScoringConfig | None = None,
) -> ScoringResult:
    """Score directed gene-pair candidates (TF -> target or ligand -> receptor).

    The FateReasoner score is generated from distilled literature evidence and
    allowed prior knowledge only. GRN edge metrics remain in the output table
    as an independent data axis.
    """
    config = config or GenePairScoringConfig()
    _validate_method(config.method)
    out_dir = _ensure_dir(out)
    context_text = read_gene_pair_context(str(context))
    args = _namespace(
        config,
        pairs=str(pairs),
        context=str(context),
        out=str(out_dir),
        source_col=source_col,
        target_col=target_col,
        candidate_id_col=candidate_id_col,
    )

    raw = _coerce_candidate_table(pairs)
    candidates = normalize_pair_candidates(
        raw,
        source_col=source_col,
        target_col=target_col,
        candidate_id_col=candidate_id_col,
        top_n=config.top_n,
        sort_col=config.sort_col,
    )
    candidates_path = out_dir / "gene_pair_candidates.csv"
    candidates.to_csv(candidates_path, index=False)
    (out_dir / "context.txt").write_text(context_text + "\n", encoding="utf-8")

    packets = build_gene_pair_distilled_packets(candidates, context_text, args, out_dir)
    packets_path = out_dir / f"{config.method}_gene_pair_packets.jsonl"
    write_jsonl(packets_path, packets)
    packet_table = gene_pair_packets_to_table(packets)
    packet_table_path = out_dir / "distilled_gene_pair_packets.csv"
    packet_table.to_csv(packet_table_path, index=False)

    if config.prepare_only:
        return ScoringResult(out_dir, candidates_path, packets_path, packet_table_path, None, None, candidates, packets, packet_table, None, None)

    scores_path = out_dir / f"{config.method}_gene_pair_scores.csv"
    scores = score_gene_pair_packets_with_llm(packets, args, scores_path)
    result = merge_gene_pair_scores(candidates, scores)
    result_path = out_dir / "gene_pair_scores.csv"
    result.to_csv(result_path, index=False)
    return ScoringResult(out_dir, candidates_path, packets_path, packet_table_path, scores_path, result_path, candidates, packets, packet_table, scores, result)


def _namespace(config: object, **extra: object) -> SimpleNamespace:
    values = asdict(config)
    values.update(extra)
    return SimpleNamespace(**values)


def _ensure_dir(value: str | Path) -> Path:
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_genes(genes: str | Path | Iterable[str], gene_col: str | None) -> list[str]:
    if isinstance(genes, (str, Path)):
        return read_gene_list(str(genes), gene_col)
    return _unique_genes([str(gene) for gene in genes])


def _coerce_candidate_table(candidates: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(candidates, pd.DataFrame):
        return candidates.copy()
    return pd.read_csv(candidates)


def _validate_method(method: str) -> None:
    if method not in SCORING_METHODS:
        raise ValueError(f"Unknown scoring method: {method}. Expected one of {SCORING_METHODS}.")
