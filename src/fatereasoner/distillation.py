from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from .models import DistilledEvidencePacket, EvidenceDocument
from .scoring import _bm25_scores, _fetch_go_annotations, _gene_query_text, _normalize_positive, _split_terms


@dataclass
class EvidenceDistillationConfig:
    method: str = "none"
    max_chars: int = 1500
    medcpt_model: str = "ncbi/MedCPT-Cross-Encoder"
    biomedbert_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    max_windows: int = 6
    go_max_terms: int = 6
    generif_max: int = 3
    raw_anchor_max_docs: int = 2
    raw_anchor_max_chars: int = 520
    anchor_strategy: str = "diverse_preserve"
    packet_style: str = "audit"
    include_go_terms: bool = False
    domain: str = "auto"


class EvidenceDistiller:
    def __init__(self, config: EvidenceDistillationConfig | None = None) -> None:
        self.config = config or EvidenceDistillationConfig()
        self._medcpt = None
        self._medcpt_error = ""
        self._biomedbert = None
        self._biomedbert_error = ""

    def enabled(self) -> bool:
        return self.config.method.lower() not in {"", "none"}

    def distill(self, gene: str, context: str, documents: list[EvidenceDocument]) -> DistilledEvidencePacket:
        method = self.config.method.lower()
        if method in {"", "none"}:
            text = "\n\n".join(doc.compact_text() for doc in documents)
            return DistilledEvidencePacket(
                gene=gene,
                context=context,
                distilled_text=_cap_text(text, self.config.max_chars),
                selected_pmids=_unique_pmids(doc.pmid for doc in documents),
                method="none",
                token_estimate=_estimate_tokens(text),
            )

        go_terms = _go_terms_for_gene(gene, self.config.go_max_terms) if self.config.include_go_terms else []
        windows = _candidate_windows(gene, context, documents)
        if not windows:
            text = _assemble_packet(
                gene,
                context,
                [],
                go_terms,
                [],
                [],
                self.config.max_chars,
                style=self.config.packet_style,
            )
            return DistilledEvidencePacket(
                gene=gene,
                context=context,
                distilled_text=text,
                selected_pmids=[],
                go_terms=go_terms,
                method=method,
                token_estimate=_estimate_tokens(text),
            )

        if method == "biomedbert_rerank":
            candidates = _rank_windows_weightfree(gene, context, windows, self.config.domain)
            max_candidates = max(self.config.max_windows, min(len(candidates), self.config.max_windows * 4))
            candidates = candidates[:max_candidates]
            scores = self._biomedbert_scores(f"{_gene_query_text(gene)} {context}", candidates)
            if scores:
                top = [
                    window
                    for window, _ in sorted(
                        zip(candidates, scores),
                        key=lambda item: (item[1], _weightfree_rank_key(gene, context, item[0], _resolve_domain(context, self.config.domain))),
                        reverse=True,
                    )[: self.config.max_windows]
                ]
            else:
                top = candidates[: self.config.max_windows]
            anchors = _raw_anchor_windows_weightfree(
                gene,
                context,
                documents,
                max_docs=self.config.raw_anchor_max_docs,
                max_chars=self.config.raw_anchor_max_chars,
                domain=self.config.domain,
                strategy=self.config.anchor_strategy,
            )
        elif method in {"lite_weightfree", "weightfree"}:
            top = _rank_windows_weightfree(gene, context, windows, self.config.domain)[: self.config.max_windows]
            anchors = _raw_anchor_windows_weightfree(
                gene,
                context,
                documents,
                max_docs=self.config.raw_anchor_max_docs,
                max_chars=self.config.raw_anchor_max_chars,
                domain=self.config.domain,
                strategy=self.config.anchor_strategy,
            )
        else:
            scored = self._score_windows(gene, context, windows, go_terms)
            top = [window for window, _ in sorted(scored, key=lambda item: item[1], reverse=True)[: self.config.max_windows]]
            anchors = _raw_anchor_windows(
                gene,
                context,
                documents,
                max_docs=self.config.raw_anchor_max_docs,
                max_chars=self.config.raw_anchor_max_chars,
                strategy=self.config.anchor_strategy,
            )
        generifs = _generif_like_statements(gene, top, self.config.generif_max)
        entities = _simple_entities(top)
        text = _assemble_packet(
            gene,
            context,
            top,
            go_terms,
            generifs,
            entities,
            self.config.max_chars,
            anchors,
            style=self.config.packet_style,
        )
        selected_pmids = _unique_pmids([window.pmid for window in anchors] + [window.pmid for window in top])
        used_method = method
        if method == "medcpt_go" and self._medcpt_error:
            used_method = f"lite_medcpt_unavailable:{self._medcpt_error}"
        if method == "biomedbert_rerank" and self._biomedbert_error:
            used_method = f"lite_biomedbert_unavailable:{self._biomedbert_error}"
        return DistilledEvidencePacket(
            gene=gene,
            context=context,
            distilled_text=text,
            selected_pmids=selected_pmids,
            go_terms=go_terms,
            generif_statements=generifs,
            pubtator_entities=entities,
            method=used_method,
            token_estimate=_estimate_tokens(text),
        )

    def _score_windows(
        self,
        gene: str,
        context: str,
        windows: list["_Window"],
        go_terms: list[str],
    ) -> list[tuple["_Window", float]]:
        query = f"{_gene_query_text(gene)} {context} {' '.join(go_terms)}"
        query_terms = _split_terms(query)
        doc_terms = [_split_terms(window.text) for window in windows]
        bm25 = _bm25_scores(query_terms, doc_terms)
        medcpt_scores = self._medcpt_scores(query, windows) if self.config.method.lower() == "medcpt_go" else None
        scored = []
        for idx, window in enumerate(windows):
            lexical = _normalize_positive(bm25[idx], bm25)
            keyword = _keyword_score(gene, context, window.text, go_terms)
            marker_match = _marker_match_score(context, window.text, gene)
            context_specific = _context_specificity_score(context, window.text)
            fate = _cell_fate_score(window.text)
            experimental = _experimental_score(window.text)
            rank_prior = 1.0 / math.sqrt(max(1, window.pubmed_rank or 1))
            score = (
                0.24 * lexical
                + 0.20 * keyword
                + 0.22 * marker_match
                + 0.14 * context_specific
                + 0.08 * fate
                + 0.08 * experimental
                + 0.04 * rank_prior
            )
            if medcpt_scores:
                score = 0.55 * medcpt_scores[idx] + 0.45 * score
            score -= _mismatch_penalty(gene, context, window.text)
            score -= _off_context_penalty(window.text)
            score -= _generic_development_penalty(context, window.text, gene)
            scored.append((window, max(0.0, min(1.0, score))))
        return scored

    def _medcpt_scores(self, query: str, windows: list["_Window"]) -> list[float] | None:
        if self._medcpt_error:
            return None
        try:
            if self._medcpt is None:
                self._medcpt = _MedCPTCrossEncoder(self.config.medcpt_model)
            return self._medcpt.score(query, [window.text for window in windows])
        except Exception as exc:
            self._medcpt_error = exc.__class__.__name__
            return None

    def _biomedbert_scores(self, query: str, windows: list["_Window"]) -> list[float] | None:
        if self._biomedbert_error:
            return None
        try:
            if self._biomedbert is None:
                self._biomedbert = _BiomedBERTReranker(self.config.biomedbert_model)
            return self._biomedbert.score(query, [window.text for window in windows])
        except Exception as exc:
            self._biomedbert_error = exc.__class__.__name__
            return None


class _MedCPTCrossEncoder:
    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [[query, text] for text in texts]
        with self.torch.no_grad():
            encoded = self.tokenizer(pairs, truncation=True, padding=True, return_tensors="pt", max_length=512)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits.squeeze(dim=-1).detach().cpu().tolist()
        if not isinstance(logits, list):
            logits = [float(logits)]
        lo = min(logits)
        hi = max(logits)
        if math.isclose(lo, hi):
            return [0.5 for _ in logits]
        return [max(0.0, min(1.0, (value - lo) / (hi - lo))) for value in logits]


class _BiomedBERTReranker:
    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import modeling_utils as transformers_modeling_utils
        from transformers.utils import import_utils as transformers_import_utils

        if not model_name:
            raise ValueError("biomedbert_model is required for biomedbert_rerank")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # The local BiomedBERT checkpoint in this workflow is stored as a trusted
        # PyTorch .bin artifact. Transformers >=4.45 blocks loading such files on
        # torch<2.6 unless the safety check is bypassed explicitly.
        original_import_check = transformers_import_utils.check_torch_load_is_safe
        original_modeling_check = transformers_modeling_utils.check_torch_load_is_safe
        transformers_import_utils.check_torch_load_is_safe = lambda: None  # type: ignore[assignment]
        transformers_modeling_utils.check_torch_load_is_safe = lambda: None  # type: ignore[assignment]
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                use_safetensors=False,
            ).to(self.device)
        finally:
            transformers_import_utils.check_torch_load_is_safe = original_import_check  # type: ignore[assignment]
            transformers_modeling_utils.check_torch_load_is_safe = original_modeling_check  # type: ignore[assignment]
        self.model.eval()

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [[query, text] for text in texts]
        with self.torch.no_grad():
            encoded = self.tokenizer(pairs, truncation=True, padding=True, return_tensors="pt", max_length=512)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits.detach().cpu()
        if len(logits.shape) == 1:
            values = logits.tolist()
        elif logits.shape[-1] == 1:
            values = logits.squeeze(dim=-1).tolist()
        else:
            values = logits[:, -1].tolist()
        if not isinstance(values, list):
            values = [float(values)]
        values = [float(value) for value in values]
        lo = min(values)
        hi = max(values)
        if math.isclose(lo, hi):
            return [0.5 for _ in values]
        return [max(0.0, min(1.0, (value - lo) / (hi - lo))) for value in values]


@dataclass
class _Window:
    pmid: str
    title: str
    text: str
    year: str | None = None
    pubmed_rank: int | None = None


def _candidate_windows(gene: str, context: str, documents: list[EvidenceDocument]) -> list[_Window]:
    context_terms = {term for term in _split_terms(context) if len(term) >= 4}
    context_terms.update(term for term in _CELL_FATE_TERMS if len(term) >= 4)
    windows: list[_Window] = []
    for doc in documents:
        source = " ".join(part for part in [doc.title, doc.abstract, doc.full_text] if part)
        sentences = _sentences(source)
        for idx, sentence in enumerate(sentences):
            if not _looks_relevant(gene, context_terms, sentence):
                continue
            left = sentences[idx - 1] if idx > 0 and len(sentence) < 180 else ""
            right = sentences[idx + 1] if idx + 1 < len(sentences) and len(sentence) < 180 else ""
            text = " ".join(part for part in [left, sentence, right] if part)
            windows.append(
                _Window(
                    pmid=doc.pmid,
                    title=doc.title,
                    text=_cap_text(text, 650),
                    year=doc.year,
                    pubmed_rank=doc.pubmed_rank,
                )
            )
    if windows:
        return _dedupe_windows(windows)
    for doc in documents:
        text = " ".join(part for part in [doc.title, doc.abstract or doc.full_text] if part)
        if text:
            windows.append(_Window(doc.pmid, doc.title, _cap_text(text, 650), doc.year, doc.pubmed_rank))
    return _dedupe_windows(windows)


def _sentences(text: str) -> list[str]:
    clean = " ".join(str(text).split())
    if not clean:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean)
    return [part.strip() for part in parts if len(part.strip()) >= 20]


def _looks_relevant(gene: str, context_terms: set[str], sentence: str) -> bool:
    lower = sentence.lower()
    if gene.lower() in lower:
        return True
    return any(term in lower for term in context_terms)


def _keyword_score(gene: str, context: str, text: str, go_terms: list[str]) -> float:
    lower = text.lower()
    score = 0.0
    aliases = [gene.lower()]
    if gene.upper() == "T":
        aliases.extend(["tbxt", "brachyury"])
    if any(alias in lower for alias in aliases):
        score += 0.30
    context_terms = [term for term in _split_terms(context) if len(term) >= 4]
    if context_terms:
        hits = sum(1 for term in context_terms if term in lower)
        score += min(0.35, hits / max(1, min(10, len(context_terms))) * 0.35)
    go_text = " ".join(go_terms).lower()
    go_hits = sum(1 for term in _split_terms(go_text) if len(term) >= 5 and term in lower)
    score += min(0.20, go_hits / 4.0 * 0.20)
    if any(term in lower for term in ("mouse", "murine", "human", "k562", "hesc", "stem cell", "embryo")):
        score += 0.15
    return min(1.0, score)


_CELL_FATE_TERMS = (
    "hesc",
    "human embryonic stem",
    "embryonic stem",
    "pluripotent",
    "pluripotency",
    "differentiation",
    "lineage",
    "cell fate",
    "fate specification",
    "gastrulation",
    "embryo",
    "embryonic",
    "development",
    "developmental",
    "progenitor",
    "stem cell",
    "induced pluripotent",
    "organoid",
    "endoderm",
    "mesoderm",
    "ectoderm",
    "neural",
    "cardiac",
    "hematopoietic",
    "trophoblast",
    "epiblast",
)


_OFF_CONTEXT_TERMS = (
    "cancer",
    "tumor",
    "tumour",
    "carcinoma",
    "prognosis",
    "survival",
    "methylation",
    "database",
    "genome-wide association",
    "gwas",
    "immune infiltration",
)


_DOMAIN_PROFILES = {
    "cell_fate": {
        "positive_terms": _CELL_FATE_TERMS,
        "off_context_terms": ("cancer", "tumor", "tumour", "carcinoma", "prognosis", "survival", "methylation"),
    },
    "cancer": {
        "positive_terms": (
            "cancer",
            "tumor",
            "tumour",
            "carcinoma",
            "metastasis",
            "oncogene",
            "tumor suppressor",
            "proliferation",
            "apoptosis",
            "drug resistance",
            "xenograft",
            "organoid",
            "survival",
            "mutation",
            "copy number",
        ),
        "off_context_terms": ("embryonic development", "gastrulation", "paraxial mesoderm", "somitogenesis"),
    },
    "regeneration": {
        "positive_terms": (
            "regeneration",
            "regenerative",
            "repair",
            "injury",
            "wound healing",
            "dedifferentiation",
            "stem cell activation",
            "progenitor activation",
            "tissue repair",
            "fibrosis",
        ),
        "off_context_terms": ("cancer prognosis", "tumor survival", "database", "gwas"),
    },
    "immunology": {
        "positive_terms": (
            "immune",
            "immunity",
            "inflammation",
            "cytokine",
            "t cell",
            "b cell",
            "macrophage",
            "monocyte",
            "dendritic",
            "nk cell",
            "antigen",
        ),
        "off_context_terms": ("embryonic development", "gastrulation", "methylation database"),
    },
    "general": {
        "positive_terms": (),
        "off_context_terms": ("database", "genome-wide association", "gwas"),
    },
}


def _rank_windows_weightfree(gene: str, context: str, windows: list[_Window], domain: str = "auto") -> list[_Window]:
    resolved_domain = _resolve_domain(context, domain)
    return sorted(
        windows,
        key=lambda window: _weightfree_rank_key(gene, context, window, resolved_domain),
        reverse=True,
    )


def _weightfree_rank_key(gene: str, context: str, window: _Window, domain: str) -> tuple:
    text = window.text
    context_terms = [term for term in _context_theme_terms(context) if len(term) >= 4]
    context_hits = _term_hits(text, context_terms)
    marker_hits = int(round(_marker_match_score(context, text, gene) * 3))
    functional_hits = _functional_term_hits(text)
    domain_hits = _term_hits(text, _DOMAIN_PROFILES.get(domain, _DOMAIN_PROFILES["general"])["positive_terms"])
    cell_fate_hits = _term_hits(text, _CELL_FATE_TERMS) if domain == "cell_fate" else 0
    off_context = _weightfree_off_context_flag(context, text, domain)
    species_mismatch = _species_mismatch_flag(context, text)
    rank = int(window.pubmed_rank or 999999)
    return (
        int(_contains_gene(gene, text)),
        context_hits,
        marker_hits,
        functional_hits,
        domain_hits,
        cell_fate_hits,
        int(not off_context),
        int(not species_mismatch),
        -rank,
        str(window.pmid),
    )


def _resolve_domain(context: str, domain: str) -> str:
    requested = str(domain or "auto").lower()
    if requested != "auto":
        return requested if requested in _DOMAIN_PROFILES else "general"
    lower = context.lower()
    if any(term in lower for term in ("cancer", "tumor", "tumour", "carcinoma", "metastasis", "oncogene")):
        return "cancer"
    if any(term in lower for term in ("regeneration", "regenerative", "repair", "injury", "wound healing")):
        return "regeneration"
    if any(term in lower for term in ("immune", "immun", "inflammation", "t cell", "b cell", "macrophage")):
        return "immunology"
    if any(term in lower for term in _CELL_FATE_TERMS):
        return "cell_fate"
    return "general"


def _contains_gene(gene: str, text: str) -> bool:
    lower = text.lower()
    aliases = [gene.lower()]
    if gene.upper() == "T":
        aliases.extend(["tbxt", "brachyury"])
    return any(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lower) for alias in aliases)


def _term_hits(text: str, terms: Iterable[str]) -> int:
    lower = text.lower()
    return sum(1 for term in terms if str(term).lower() in lower)


def _functional_term_hits(text: str) -> int:
    terms = (
        "knockout",
        "knockdown",
        "mutant",
        "loss-of-function",
        "gain-of-function",
        "overexpression",
        "crispr",
        "crispra",
        "perturb-seq",
        "single-cell",
        "in vivo",
        "in vitro",
        "functional",
        "differentiation",
        "lineage",
        "proliferation",
        "apoptosis",
        "migration",
        "invasion",
        "regeneration",
        "repair",
    )
    return _term_hits(text, terms)


def _weightfree_off_context_flag(context: str, text: str, domain: str) -> bool:
    lower = text.lower()
    domain_terms = _DOMAIN_PROFILES.get(domain, _DOMAIN_PROFILES["general"])["positive_terms"]
    if _term_hits(text, domain_terms) > 0:
        return False
    off_terms = _DOMAIN_PROFILES.get(domain, _DOMAIN_PROFILES["general"])["off_context_terms"]
    if any(term in lower for term in off_terms):
        return True
    if domain != "cancer" and any(term in lower for term in ("cancer", "tumor", "tumour", "carcinoma")):
        return True
    if "database" in lower and _context_specificity_score(context, text) <= 0:
        return True
    return False


def _species_mismatch_flag(context: str, text: str) -> bool:
    lower = text.lower()
    context_lower = context.lower()
    if "mouse" in context_lower and "human" in lower and not any(term in lower for term in ("mouse", "murine")):
        return True
    if "human" in context_lower and any(term in lower for term in ("mouse", "murine")) and "human" not in lower:
        return True
    return False


def _context_specificity_score(context: str, text: str) -> float:
    lower = text.lower()
    terms = [term for term in _context_theme_terms(context) if len(term) >= 4]
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if term in lower)
    return min(1.0, hits / max(1, min(6, len(terms))))


def _marker_terms(context: str) -> list[str]:
    lower = context.lower()
    marker_text = ""
    marker_prefix = "target marker genes:"
    if marker_prefix in lower:
        marker_text = context[lower.index(marker_prefix) + len(marker_prefix) :]
    terms = []
    for raw in re.split(r"[;,]\s*", marker_text):
        item = raw.strip()
        if not item:
            continue
        gene = re.sub(r"[^A-Za-z0-9_.-]", "", item).upper()
        if len(gene) >= 3:
            terms.append(gene)
    return terms


def _context_theme_terms(context: str) -> list[str]:
    lower = context.lower()
    before_markers = lower.split("target marker genes:", 1)[0]
    stop = {"human", "cell", "target", "cluster", "genes", "gene"}
    return [term for term in _split_terms(before_markers) if len(term) >= 4 and term not in stop]


def _marker_match_score(context: str, text: str, gene: str | None = None) -> float:
    gene_upper = str(gene or "").upper()
    markers = [term for term in _marker_terms(context) if len(term) >= 3 and term != gene_upper]
    if not markers:
        return 0.0
    upper = text.upper()
    hits = 0
    for marker in markers:
        if re.search(rf"(?<![A-Z0-9]){re.escape(marker)}(?![A-Z0-9])", upper):
            hits += 1
    return min(1.0, hits / 3.0)


def _cell_fate_score(text: str) -> float:
    lower = text.lower()
    hits = sum(1 for term in _CELL_FATE_TERMS if term in lower)
    return min(1.0, hits / 5.0)


def _experimental_score(text: str) -> float:
    lower = text.lower()
    terms = (
        "knockout",
        "knockdown",
        "mutant",
        "loss-of-function",
        "gain-of-function",
        "overexpression",
        "crispr",
        "crispra",
        "perturb-seq",
        "single-cell",
        "lineage",
        "differentiation",
        "cell fate",
        "in vivo",
        "in vitro",
        "functional",
    )
    hits = sum(1 for term in terms if term in lower)
    return min(1.0, hits / 4.0)


def _off_context_penalty(text: str) -> float:
    lower = text.lower()
    generic = sum(1 for term in _OFF_CONTEXT_TERMS if term in lower)
    if generic <= 0:
        return 0.0
    useful = sum(1 for term in _CELL_FATE_TERMS if term in lower) + sum(
        1
        for term in (
            "knockout",
            "knockdown",
            "mutant",
            "overexpression",
            "crispr",
            "in vivo",
            "in vitro",
            "functional",
        )
        if term in lower
    )
    return min(0.25, max(0.0, (generic - 0.5 * useful) * 0.08))


def _generic_development_penalty(context: str, text: str, gene: str | None = None) -> float:
    fate = _cell_fate_score(text)
    if fate < 0.4:
        return 0.0
    marker = _marker_match_score(context, text, gene)
    theme = _context_specificity_score(context, text)
    if marker >= 0.34 or theme >= 0.5:
        return 0.0
    return min(0.22, 0.12 + 0.10 * fate)


def _mismatch_penalty(gene: str, context: str, text: str) -> float:
    lower = text.lower()
    context_lower = context.lower()
    penalty = 0.0
    if any(term in context_lower for term in ("retina", "retinal", "muller", "müller", "regeneration")):
        retina_terms = (
            "retina",
            "retinal",
            "muller glia",
            "müller glia",
            "photoreceptor",
            "optic",
            "vision",
            "eye",
            "regeneration",
        )
        nmp_terms = (
            "neuromesodermal",
            "nmp",
            "paraxial mesoderm",
            "presomitic",
            "tailbud",
            "somitogenesis",
        )
        if any(term in lower for term in nmp_terms) and not any(term in lower for term in retina_terms):
            penalty += 0.35
    if gene.upper() == "T" and ("t cell" in lower or "t-cell" in lower) and "tbxt" not in lower and "brachyury" not in lower:
        penalty += 0.35
    if "mouse" in context_lower and "human" in lower and not any(term in lower for term in ("mouse", "murine")):
        penalty += 0.10
    if "human" in context_lower and any(term in lower for term in ("mouse", "murine")) and "human" not in lower:
        penalty += 0.10
    if "k562" in context_lower and "k562" not in lower and any(term in lower for term in ("neuron", "heart", "kidney", "liver")):
        penalty += 0.10
    return min(0.5, penalty)


def _go_terms_for_gene(gene: str, max_terms: int) -> list[str]:
    annotations = _fetch_go_annotations(gene)
    ranked = sorted(
        annotations,
        key=lambda item: (
            0 if str(item.get("namespace", "")).lower() == "biological_process" else 1,
            str(item.get("evidence", "")),
            str(item.get("term", "")),
        ),
    )
    terms = []
    for annotation in ranked:
        term = str(annotation.get("term", "")).strip()
        evidence = str(annotation.get("evidence", "")).strip()
        namespace = str(annotation.get("namespace", "")).strip()
        if term:
            label = term
            if evidence:
                label = f"{label} [{evidence}]"
            if namespace:
                label = f"{label} ({namespace})"
            terms.append(label)
        if len(terms) >= max_terms:
            break
    return terms


def _generif_like_statements(gene: str, windows: list[_Window], max_items: int) -> list[str]:
    statements = []
    for window in windows:
        if _experimental_score(window.text) <= 0:
            continue
        sentence = _sentences(window.text)
        text = sentence[0] if sentence else window.text
        statements.append(f"{gene}: {_cap_text(text, 220)} (PMID:{window.pmid})")
        if len(statements) >= max_items:
            break
    return statements


def _simple_entities(windows: list[_Window]) -> list[str]:
    text = " ".join(window.text for window in windows).lower()
    entities = []
    for label, terms in {
        "species:human": ("human",),
        "species:mouse": ("mouse", "murine", "mus musculus"),
        "cell_line:K562": ("k562",),
        "cell_type:hESC": ("hesc", "human embryonic stem"),
        "assay:Perturb-seq": ("perturb-seq",),
        "assay:single-cell RNA-seq": ("single-cell rna-seq", "scrna-seq", "single-cell transcript"),
    }.items():
        if any(term in text for term in terms):
            entities.append(label)
    return entities


def _raw_anchor_windows(
    gene: str,
    context: str,
    documents: list[EvidenceDocument],
    max_docs: int,
    max_chars: int,
    strategy: str = "balanced",
) -> list[_Window]:
    scored: list[tuple[_Window, float]] = []
    for doc in documents:
        text = " ".join(part for part in [doc.title, doc.abstract or doc.full_text] if part)
        if not text:
            continue
        window = _Window(
            pmid=doc.pmid,
            title=doc.title,
            text=_cap_text(text, max_chars),
            year=doc.year,
            pubmed_rank=doc.pubmed_rank,
        )
        score = (
            0.25 * _keyword_score(gene, context, text, [])
            + 0.30 * _marker_match_score(context, text, gene)
            + 0.18 * _context_specificity_score(context, text)
            + 0.15 * _cell_fate_score(text)
            + 0.12 * _experimental_score(text)
            - _off_context_penalty(text)
            - _generic_development_penalty(context, text, gene)
        )
        if gene.lower() in text.lower():
            score += 0.10
        scored.append((window, max(0.0, score)))
    ranked = [window for window, _ in sorted(scored, key=lambda item: item[1], reverse=True)]
    return _select_anchor_windows(gene, context, ranked, max_docs=max_docs, strategy=strategy)


def _raw_anchor_windows_weightfree(
    gene: str,
    context: str,
    documents: list[EvidenceDocument],
    max_docs: int,
    max_chars: int,
    domain: str = "auto",
    strategy: str = "balanced",
) -> list[_Window]:
    windows: list[_Window] = []
    for doc in documents:
        text = " ".join(part for part in [doc.title, doc.abstract or doc.full_text] if part)
        if text:
            windows.append(
                _Window(
                    pmid=doc.pmid,
                    title=doc.title,
                    text=_cap_text(text, max_chars),
                    year=doc.year,
                    pubmed_rank=doc.pubmed_rank,
                )
            )
    ranked = _rank_windows_weightfree(gene, context, windows, domain)
    return _select_anchor_windows(gene, context, ranked, max_docs=max_docs, strategy=strategy)


def _select_anchor_windows(
    gene: str,
    context: str,
    ranked_windows: list[_Window],
    *,
    max_docs: int,
    strategy: str = "balanced",
) -> list[_Window]:
    ranked_windows = _dedupe_windows(ranked_windows)
    if max_docs <= 0 or not ranked_windows:
        return []
    if str(strategy or "balanced").lower() != "diverse_preserve":
        return ranked_windows[:max_docs]

    selected: list[_Window] = []
    seen_pmids: set[str] = set()

    def add(window: _Window) -> bool:
        if window.pmid in seen_pmids:
            return False
        seen_pmids.add(window.pmid)
        selected.append(window)
        return True

    def choose_first(predicate) -> bool:
        for window in ranked_windows:
            if window.pmid in seen_pmids:
                continue
            if predicate(window):
                return add(window)
        return False

    # Keep a representative from each evidence mode before filling from the
    # general rank order. This preserves mismatch-heavy but still informative
    # anchors instead of collapsing them into a single cautionary fragment.
    choose_first(lambda window: _contains_gene(gene, window.text))
    choose_first(lambda window: _marker_match_score(context, window.text, gene) >= 0.34 or _context_specificity_score(context, window.text) >= 0.35)
    choose_first(lambda window: _experimental_score(window.text) >= 0.25)
    choose_first(lambda window: _cell_fate_score(window.text) >= 0.35)

    for window in ranked_windows:
        if len(selected) >= max_docs:
            break
        add(window)

    return _dedupe_windows(selected[:max_docs])


def _assemble_packet(
    gene: str,
    context: str,
    windows: list[_Window],
    go_terms: list[str],
    generifs: list[str],
    entities: list[str],
    max_chars: int,
    raw_anchors: list[_Window] | None = None,
    style: str = "audit",
) -> str:
    raw_anchors = raw_anchors or []
    concise = style.lower() == "concise"
    sections = [f"Gene: {gene}", f"Context: {context}"]
    if go_terms:
        sections.append("GO prior: " + "; ".join(go_terms[:6]))
    if generifs:
        sections.append("Mechanistic/function statements: " + " | ".join(generifs))
    if entities:
        sections.append("Detected entities: " + "; ".join(entities))
    marker_hits = _packet_marker_hits(context, raw_anchors + windows, gene)
    if marker_hits:
        sections.append("Cluster marker/context overlap: " + "; ".join(marker_hits[:12]))
    elif not concise:
        sections.append("Cluster marker/context overlap: none detected in selected evidence; treat broad developmental evidence as indirect.")
    if not concise and gene.upper() in set(_marker_terms(context)):
        sections.append(f"Candidate-marker note: {gene} appears in the cluster marker list; expression-marker status alone is not mechanistic evidence and should only be a weak tie-breaker.")
    off_context = [window for window in (raw_anchors + windows) if _off_context_penalty(window.text) > 0]
    if off_context and not concise:
        pmids = ";".join(_unique_pmids(window.pmid for window in off_context))
        sections.append(f"Off-context/generic caveat: PMID(s) {pmids} include disease, cancer, prognosis, methylation, or database-heavy evidence; use only if connected to hESC/development/cell fate.")
    generic_development = [window for window in (raw_anchors + windows) if _generic_development_penalty(context, window.text, gene) > 0]
    if generic_development and not concise:
        pmids = ";".join(_unique_pmids(window.pmid for window in generic_development))
        sections.append(f"Broad-development caveat: PMID(s) {pmids} support general development/differentiation but do not clearly match this cluster marker context.")
    if raw_anchors:
        anchors = []
        for window in raw_anchors:
            year = f", {window.year}" if window.year else ""
            anchors.append(f"PMID {window.pmid}{year}: {window.title}. {window.text}")
        prefix = "Selected context-linked evidence anchors: " if concise else "Context-matched raw evidence anchors: "
        sections.append(prefix + " || ".join(anchors))
    if windows:
        evidence = []
        for window in windows:
            year = f", {window.year}" if window.year else ""
            evidence.append(f"PMID {window.pmid}{year}: {window.title}. {window.text}")
        prefix = "Selected functional evidence windows: " if concise else "Distilled PubMed evidence windows: "
        sections.append(prefix + " || ".join(evidence))
    else:
        if not concise:
            sections.append("Distilled PubMed evidence windows: no relevant sentence windows after distillation.")
    return _cap_packet_text("\n".join(sections), max_chars)


def _packet_marker_hits(context: str, windows: list[_Window], gene: str | None = None) -> list[str]:
    markers = _marker_terms(context)
    gene_upper = str(gene or "").upper()
    markers = [marker for marker in markers if marker != gene_upper]
    if not markers:
        return []
    text = " ".join(window.text for window in windows).upper()
    hits = []
    for marker in markers:
        if re.search(rf"(?<![A-Z0-9]){re.escape(marker)}(?![A-Z0-9])", text):
            hits.append(marker)
    return hits


def _dedupe_windows(windows: list[_Window]) -> list[_Window]:
    seen = set()
    out = []
    for window in windows:
        key = (window.pmid, window.text[:120].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(window)
    return out


def _unique_pmids(pmids: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for pmid in pmids:
        value = str(pmid)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _cap_text(text: str, max_chars: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def _cap_packet_text(text: str, max_chars: int) -> str:
    lines = [" ".join(line.split()) for line in str(text).splitlines()]
    clean = "\n".join(line for line in lines if line)
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(str(text)) / 4))
