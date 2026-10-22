from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import EvidenceDocument


@dataclass
class EvidenceSelectionConfig:
    threshold: float = 0.35
    min_keep: int = 5
    max_keep: int = 20
    method: str = "tfidf-context"


@dataclass
class EvidenceSelectionResult:
    documents: list[EvidenceDocument]
    retrieved_count: int
    evidence_count: int
    filter_method: str


def select_evidence_documents(
    gene: str,
    context: str,
    documents: list[EvidenceDocument],
    config: EvidenceSelectionConfig | None = None,
) -> EvidenceSelectionResult:
    config = config or EvidenceSelectionConfig()
    if not documents:
        return EvidenceSelectionResult([], 0, 0, config.method)

    scored = _score_documents(gene, context, documents)
    ranked = sorted(zip(documents, scored), key=lambda item: item[1], reverse=True)
    above_threshold = [doc for doc, score in ranked if score >= config.threshold]
    min_keep = max(0, min(config.min_keep, len(ranked)))
    max_keep = max(min_keep, min(config.max_keep, len(ranked)))
    selected = above_threshold[:max_keep]
    if len(selected) < min_keep:
        selected = [doc for doc, _ in ranked[:min_keep]]
    selected = selected[:max_keep]
    return EvidenceSelectionResult(selected, len(documents), len(selected), config.method)


def _score_documents(gene: str, context: str, documents: list[EvidenceDocument]) -> list[float]:
    texts = [_document_text(doc, context) for doc in documents]
    try:
        return _tfidf_scores(gene, context, texts)
    except Exception:
        return [_keyword_score(gene, context, text) for text in texts]


def _tfidf_scores(gene: str, context: str, texts: list[str]) -> list[float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    query = f"{gene} {context}"
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([query] + texts)
    similarities = cosine_similarity(matrix[0], matrix[1:]).ravel()
    keyword_scores = np.array([_keyword_score(gene, context, text) for text in texts], dtype=float)
    scores = 0.75 * similarities + 0.25 * keyword_scores
    return np.clip(scores, 0.0, 1.0).tolist()


def _keyword_score(gene: str, context: str, text: str) -> float:
    lower = text.lower()
    score = 0.0
    if gene.lower() in lower:
        score += 0.25
    context_terms = [term for term in _split_terms(context) if len(term) >= 4]
    if context_terms:
        hits = sum(1 for term in context_terms if term in lower)
        score += min(0.35, hits / max(1, len(context_terms)) * 0.35)
    if any(term in lower for term in _direct_context_terms(context)):
        score += 0.2
    if any(term in lower for term in _species_terms(context)):
        score += 0.1
    if any(term in lower for term in _experimental_terms()):
        score += 0.1
    if _is_retina_context(context) and any(term in lower for term in _nmp_terms()) and not any(
        term in lower for term in _direct_context_terms(context)
    ):
        score -= 0.35
    return max(0.0, min(1.0, score))


def _document_text(doc: EvidenceDocument, context: str) -> str:
    return f"{doc.title} {doc.abstract} {_focused_full_text(doc.full_text, context)}".strip()


def _focused_full_text(text: str, context: str, max_chars: int = 8000) -> str:
    if not text:
        return ""
    terms = set(_direct_context_terms(context)) | set(_experimental_terms())
    lower = text.lower()
    snippets: list[str] = []
    for term in sorted(terms, key=len, reverse=True):
        start = 0
        while True:
            idx = lower.find(term, start)
            if idx == -1:
                break
            left = max(0, idx - 450)
            right = min(len(text), idx + 650)
            snippet = " ".join(text[left:right].split())
            if snippet and snippet not in snippets:
                snippets.append(snippet)
            start = idx + len(term)
            if sum(len(item) for item in snippets) >= max_chars:
                return " ".join(snippets)[:max_chars]
    if snippets:
        return " ".join(snippets)[:max_chars]
    return " ".join(text.split())[:2500]


def _split_terms(text: str) -> list[str]:
    clean = "".join(char.lower() if char.isalnum() else " " for char in text)
    return [term for term in clean.split() if term]


def _species_terms(context: str = "") -> tuple[str, ...]:
    lower = context.lower()
    if any(term in lower for term in ("retina", "retinal", "regeneration", "zebrafish", "danio")):
        return ("mouse", "murine", "mus musculus", "zebrafish", "danio rerio", "vertebrate")
    if any(term in lower for term in ("human", "cancer", "carcinoma", "tumor", "tumour", "crc", "oscc")):
        return ("human", "homo sapiens")
    return ("mouse", "murine", "mus musculus")


def _direct_context_terms(context: str = "") -> tuple[str, ...]:
    lower = context.lower()
    if _is_retina_context(context):
        return (
            "retinal regeneration",
            "retina",
            "retinal",
            "muller glia",
            "müller glia",
            "photoreceptor",
            "ganglion cell",
            "bipolar cell",
            "amacrine cell",
            "retinal pigment epithelium",
            "regenerative neurogenesis",
            "cell fate",
            "gene regulatory network",
        )
    if any(term in lower for term in ("oral squamous", "oscc", "head and neck")):
        return (
            "oral squamous cell carcinoma",
            "head and neck squamous cell carcinoma",
            "cancer-associated fibroblast",
            "fibroblast",
            "myofibroblast",
            "leading edge",
            "invasive edge",
            "invasion",
            "extracellular matrix",
            "collagen",
            "cell-cell communication",
        )
    if any(term in lower for term in ("colorectal", "colon cancer", "rectal cancer", "crc")):
        return (
            "colorectal cancer",
            "colon cancer",
            "rectal cancer",
            "macrophage",
            "tumor microenvironment",
            "cell-cell communication",
            "ligand-receptor",
        )
    return (
        "neuromesodermal",
        "nmp",
        "paraxial mesoderm",
        "spinal cord",
        "somite",
        "presomitic",
        "tailbud",
        "embryonic development",
    )


def _is_retina_context(context: str) -> bool:
    lower = context.lower()
    return any(term in lower for term in ("retina", "retinal", "muller", "müller", "regeneration"))


def _nmp_terms() -> tuple[str, ...]:
    return (
        "neuromesodermal",
        "nmp",
        "paraxial mesoderm",
        "presomitic",
        "tailbud",
        "somitogenesis",
    )


def _experimental_terms() -> tuple[str, ...]:
    return (
        "knockout",
        "mutant",
        "loss-of-function",
        "gain-of-function",
        "lineage tracing",
        "in vivo",
        "in vitro",
        "functional",
    )
