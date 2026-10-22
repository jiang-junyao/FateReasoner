from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Protocol

from .models import EvidenceDocument, KnowledgeScore


class KnowledgeScorer(Protocol):
    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        ...


class MockScorer:
    """Deterministic scorer for dry runs and tests."""

    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        pmids = [doc.pmid for doc in documents[:3]]
        if not documents:
            return KnowledgeScore(
                gene=gene,
                knowledge_score=0.0,
                evidence_level="none",
                evidence_summary="No PubMed evidence retrieved for the requested context.",
                caveats="Mock score; no LLM evidence adjudication was performed.",
                supporting_pmids=[],
                species_context_match="none",
            )
        score = min(0.85, 0.2 + 0.12 * len(documents))
        return KnowledgeScore(
            gene=gene,
            knowledge_score=score,
            evidence_level="association",
            evidence_summary=f"Retrieved {len(documents)} PubMed records mentioning {gene}; mock scorer did not judge experimental strength.",
            caveats="Mock score for pipeline validation only.",
            supporting_pmids=pmids,
            species_context_match="unknown",
        )


class LiteratureCountScorer:
    """Simple literature-density baseline for benchmark comparisons."""

    def __init__(self, cap: int = 20) -> None:
        self.cap = cap

    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        count = len(documents)
        score = min(1.0, math.log1p(count) / math.log1p(max(1, self.cap)))
        return KnowledgeScore(
            gene=gene,
            knowledge_score=score,
            evidence_level="association" if count else "none",
            evidence_summary=f"Literature-count baseline found {count} filtered PubMed records for {gene}.",
            caveats="Count baseline measures literature density, not functional evidence strength.",
            supporting_pmids=[doc.pmid for doc in documents[:5]],
            species_context_match="unknown" if count else "none",
        )


class HeuristicKnowledgeScorer:
    """Rule-based scorer for offline pilot runs and interpretable baselines."""

    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        if not documents:
            return _empty_score(gene, "No PubMed evidence retrieved for the requested context.")

        records = [_classify_document(gene, context, doc) for doc in documents]
        best = max(records, key=lambda item: item["score"])
        pmids = [record["pmid"] for record in sorted(records, key=lambda item: item["score"], reverse=True)[:5]]
        return KnowledgeScore(
            gene=gene,
            knowledge_score=float(best["score"]),
            evidence_level=str(best["level"]),
            evidence_summary=(
                f"Heuristic evidence scan found {len(documents)} filtered records; "
                f"strongest signal was {best['level']} based on keywords: {best['matched']}."
            ),
            caveats="Rule-based score; keyword matches can miss evidence or overcount incidental mentions.",
            supporting_pmids=pmids,
            species_context_match=str(best["species_context_match"]),
        )


class EmbeddingKnowledgeScorer:
    """Local TF-IDF similarity scorer for no-API evidence scoring."""

    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        if not documents:
            return _empty_score(gene, "No PubMed evidence retrieved for the requested context.")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except Exception as exc:  # pragma: no cover - dependency provided by package metadata
            raise RuntimeError("Install scikit-learn to use EmbeddingKnowledgeScorer") from exc

        query = f"{_gene_query_text(gene)} {context}"
        texts = [_doc_text(doc, gene) for doc in documents]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform([query] + texts)
        similarities = cosine_similarity(matrix[0], matrix[1:]).ravel()
        best_idx = int(similarities.argmax())
        best_similarity = float(similarities[best_idx])
        keyword_boost = _context_keyword_strength(gene, context, texts[best_idx])
        count_boost = min(0.2, math.log1p(len(documents)) / math.log1p(20) * 0.2)
        score = min(1.0, 0.65 * best_similarity + 0.25 * keyword_boost + count_boost)
        level = _level_from_score(score)
        ranked = sorted(zip(documents, similarities), key=lambda item: item[1], reverse=True)
        return KnowledgeScore(
            gene=gene,
            knowledge_score=score,
            evidence_level=level,
            evidence_summary=(
                f"Local TF-IDF scorer found max context similarity {best_similarity:.3f} "
                f"across {len(documents)} filtered records."
            ),
            caveats="Embedding baseline estimates textual relevance, not direct functional evidence.",
            supporting_pmids=[doc.pmid for doc, _ in ranked[:5]],
            species_context_match=_species_context_match(texts[best_idx]),
        )


class BM25KnowledgeScorer:
    """BM25 retrieval baseline over filtered PubMed title/abstract text."""

    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        if not documents:
            return _empty_score(gene, "No PubMed evidence retrieved for the requested context.")
        query_terms = _split_terms(f"{_gene_query_text(gene)} {context}")
        doc_terms = [_split_terms(_doc_text(doc, gene)) for doc in documents]
        scores = _bm25_scores(query_terms, doc_terms)
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        normalized = _normalize_positive(scores[best_idx], scores)
        keyword_boost = _context_keyword_strength(gene, context, _doc_text(documents[best_idx], gene))
        score = min(1.0, 0.75 * normalized + 0.25 * keyword_boost)
        ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
        return KnowledgeScore(
            gene=gene,
            knowledge_score=score,
            evidence_level=_level_from_score(score),
            evidence_summary=f"BM25 baseline found max relevance score {scores[best_idx]:.3f} across {len(documents)} records.",
            caveats="BM25 measures lexical relevance, not direct functional evidence.",
            supporting_pmids=[doc.pmid for doc, _ in ranked[:5]],
            species_context_match=_species_context_match(_doc_text(documents[best_idx], gene)),
        )


class PubMedRankScorer:
    """Baseline that trusts PubMed relevance order from E-utilities."""

    def __init__(self, cap_rank: int = 20) -> None:
        self.cap_rank = cap_rank

    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        if not documents:
            return _empty_score(gene, "No PubMed evidence retrieved for the requested context.")
        ranks = [doc.pubmed_rank for doc in documents if doc.pubmed_rank is not None]
        best_rank = min(ranks) if ranks else 1
        rank_score = max(0.0, 1.0 - (best_rank - 1) / max(1, self.cap_rank))
        count_boost = min(0.2, math.log1p(len(documents)) / math.log1p(20) * 0.2)
        score = min(1.0, 0.8 * rank_score + count_boost)
        ranked = sorted(documents, key=lambda doc: doc.pubmed_rank if doc.pubmed_rank is not None else 9999)
        return KnowledgeScore(
            gene=gene,
            knowledge_score=score,
            evidence_level="association",
            evidence_summary=(
                f"PubMed relevance-rank baseline found {len(documents)} filtered records; "
                f"best original PubMed rank was {best_rank}."
            ),
            caveats="PubMed rank baseline measures search-engine relevance, not functional evidence strength.",
            supporting_pmids=[doc.pmid for doc in ranked[:5]],
            species_context_match=_species_context_match(_doc_text(ranked[0], gene)),
        )


class GOAnnotationScorer:
    """No-training GO annotation overlap scorer backed by MyGene.info."""

    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        annotations = _fetch_go_annotations(gene)
        if not annotations:
            return KnowledgeScore(
                gene=gene,
                knowledge_score=0.0,
                evidence_level="none",
                evidence_summary=f"No GO annotations found for {gene} through MyGene.info.",
                caveats="GO baseline depends on structured annotations and may miss recent literature.",
                supporting_pmids=[],
                species_context_match="none",
            )
        context_terms = set(term for term in _split_terms(context) if len(term) >= 4)
        scored = []
        for annotation in annotations:
            text = f"{annotation.get('term', '')} {annotation.get('evidence', '')}"
            terms = set(term for term in _split_terms(text) if len(term) >= 4)
            overlap = len(context_terms & terms)
            scored.append((annotation, overlap))
        best, best_overlap = max(scored, key=lambda item: item[1])
        overlap_score = min(0.75, best_overlap / max(1, min(len(context_terms), 8)) * 0.75)
        count_boost = min(0.25, math.log1p(len(annotations)) / math.log1p(30) * 0.25)
        score = min(1.0, overlap_score + count_boost)
        return KnowledgeScore(
            gene=gene,
            knowledge_score=score,
            evidence_level="association" if score else "none",
            evidence_summary=(
                f"GO baseline found {len(annotations)} annotations; best matched term was "
                f"'{best.get('term', '')}' with {best_overlap} context-term overlaps."
            ),
            caveats="GO overlap is structured prior knowledge, not context-specific experimental evidence.",
            supporting_pmids=[],
            species_context_match="partial" if best_overlap else "none",
        )


class PubMedBERTKnowledgeScorer:
    """PubMedBERT embedding baseline for no-training biomedical text relevance."""

    def __init__(self, model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext") -> None:
        self.model_name = model
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install transformers and torch to use PubMedBERTKnowledgeScorer") from exc
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        model_kwargs = {}
        if os.path.isdir(model) and os.path.exists(os.path.join(model, "pytorch_model.bin")):
            model_kwargs["weights_only"] = False
        self.model = AutoModel.from_pretrained(model, **model_kwargs).to(self.device)
        self.model.eval()

    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        if not documents:
            return _empty_score(gene, "No PubMed evidence retrieved for the requested context.")
        query = f"{_gene_query_text(gene)} {context}"
        texts = [_doc_text(doc, gene) for doc in documents]
        embeddings = self._encode([query] + texts)
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        similarities = (doc_embeddings @ query_embedding).tolist()
        best_idx = max(range(len(similarities)), key=lambda idx: similarities[idx])
        best_similarity = float(similarities[best_idx])
        normalized = max(0.0, min(1.0, (best_similarity + 1.0) / 2.0))
        keyword_boost = _context_keyword_strength(gene, context, texts[best_idx])
        count_boost = min(0.15, math.log1p(len(documents)) / math.log1p(20) * 0.15)
        score = min(1.0, 0.7 * normalized + 0.15 * keyword_boost + count_boost)
        ranked = sorted(zip(documents, similarities), key=lambda item: item[1], reverse=True)
        return KnowledgeScore(
            gene=gene,
            knowledge_score=score,
            evidence_level=_level_from_score(score),
            evidence_summary=(
                f"PubMedBERT embedding baseline found max cosine similarity {best_similarity:.3f} "
                f"using {self.model_name} on {self.device}."
            ),
            caveats="Original PubMedBERT is not a sentence-transformer; this is a no-training embedding baseline.",
            supporting_pmids=[doc.pmid for doc, _ in ranked[:5]],
            species_context_match=_species_context_match(texts[best_idx]),
        )

    def _encode(self, texts: list[str]):
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            output = self.model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = summed / counts
        return self.torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu()


class GOPubMedHybridScorer:
    """Hybrid no-training scorer that combines GO annotations and PubMed evidence text."""

    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        go_score, go_summary, best_go_term = _go_overlap_score(gene, context)
        pubmed_score, pubmed_level, pubmed_summary, ranked_docs, species_match = _pubmed_text_evidence_score(
            gene, context, documents
        )
        count_score = min(1.0, math.log1p(len(documents)) / math.log1p(20))
        if documents and go_score > 0:
            score = 0.50 * pubmed_score + 0.35 * go_score + 0.15 * count_score
        elif documents:
            score = 0.75 * pubmed_score + 0.25 * count_score
        elif go_score > 0:
            score = go_score
        else:
            score = 0.0
        score = min(1.0, score)
        level = _level_from_score(score) if documents else ("association" if go_score else "none")
        return KnowledgeScore(
            gene=gene,
            knowledge_score=score,
            evidence_level=level if level != "none" else pubmed_level,
            evidence_summary=(
                f"GO+PubMed hybrid: {go_summary}; {pubmed_summary}; "
                f"best_go_term='{best_go_term}'."
            ),
            caveats=(
                "Hybrid no-training score combines structured GO annotations with PubMed text relevance; "
                "it is not a curated functional-evidence judgment."
            ),
            supporting_pmids=[doc.pmid for doc, _ in ranked_docs[:5]],
            species_context_match=species_match,
        )


class OfflineEvidenceJudgeScorer:
    """No-LLM evidence judge that combines capped evidence levels and component scores."""

    def __init__(self, pubmedbert_model: str | None = None, use_pubmedbert: bool = True) -> None:
        self.pubmedbert: PubMedBERTKnowledgeScorer | None = None
        if use_pubmedbert and pubmedbert_model and os.path.isdir(pubmedbert_model):
            self.pubmedbert = PubMedBERTKnowledgeScorer(model=pubmedbert_model)

    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        annotations = _fetch_go_annotations(gene)
        go_score, go_specificity, generic_penalty, go_summary, best_go_term = _go_specificity_score(gene, context, annotations)
        pubmed_text_score, pubmed_level, pubmed_summary, ranked_docs, species_match = _pubmed_text_evidence_score(
            gene, context, documents
        )
        experiment_score, experiment_level = _experiment_strength_score(gene, context, documents)
        species_context_score = _species_context_score(documents)
        functional_context_score = _functional_context_score(gene, documents)
        mismatch_penalty = _context_mismatch_penalty(gene, documents)
        pubmed_count_score = min(1.0, math.log1p(len(documents)) / math.log1p(20))
        pubmedbert_score = self._pubmedbert_component(gene, context, documents)
        offline_level = _offline_evidence_level(
            has_go=bool(annotations),
            has_pubmed=bool(documents),
            pubmed_level=pubmed_level,
            experiment_level=experiment_level,
            species_context_score=species_context_score,
            go_specificity_score=go_specificity,
        )
        evidence_cap = _evidence_cap(offline_level)
        raw = (
            0.30 * pubmed_text_score
            + 0.20 * experiment_score
            + 0.20 * species_context_score
            + 0.15 * go_specificity
            + 0.10 * pubmed_count_score
            + 0.05 * pubmedbert_score
        )
        raw = max(0.0, raw - mismatch_penalty)
        calibrated, score_floor = _calibrate_offline_score(
            raw=raw,
            evidence_level=offline_level,
            evidence_cap=evidence_cap,
            pubmed_text_score=pubmed_text_score,
            experiment_score=experiment_score,
            species_context_score=species_context_score,
            go_specificity_score=go_specificity,
            functional_context_score=functional_context_score,
        )
        ranked_pmids = [doc.pmid for doc, _ in ranked_docs[:5]]
        best_doc = ranked_docs[0][0] if ranked_docs else None
        return KnowledgeScore(
            gene=gene,
            knowledge_score=calibrated,
            evidence_level=offline_level,
            evidence_summary=(
                f"Offline judge: level={offline_level}, cap={evidence_cap:.2f}; "
                f"{go_summary}; {pubmed_summary}."
            ),
            caveats="No-LLM evidence judge; component rules approximate but do not replace manual/LLM curation.",
            supporting_pmids=ranked_pmids,
            species_context_match=species_match,
            extra={
                "knowledge_score_raw": raw,
                "knowledge_score_calibrated": calibrated,
                "knowledge_score_floor": score_floor,
                "evidence_cap": evidence_cap,
                "offline_evidence_level": offline_level,
                "pubmed_count_score": pubmed_count_score,
                "pubmed_text_score": pubmed_text_score,
                "pubmedbert_rerank_score": pubmedbert_score,
                "go_score": go_score,
                "go_specificity_score": go_specificity,
                "experiment_keyword_score": experiment_score,
                "species_context_score": species_context_score,
                "functional_context_score": functional_context_score,
                "context_mismatch_penalty": mismatch_penalty,
                "generic_go_penalty": generic_penalty,
                "best_go_term": best_go_term,
                "best_pubmed_pmid": best_doc.pmid if best_doc else "",
                "best_pubmed_title": best_doc.title if best_doc else "",
            },
        )

    def _pubmedbert_component(self, gene: str, context: str, documents: list[EvidenceDocument]) -> float:
        if self.pubmedbert is None or not documents:
            return 0.0
        query = f"{_gene_query_text(gene)} {context}"
        texts = [_doc_text(doc, gene) for doc in documents]
        embeddings = self.pubmedbert._encode([query] + texts)
        similarities = (embeddings[1:] @ embeddings[0]).tolist()
        if not similarities:
            return 0.0
        # Raw BERT cosine is compressed; this component is intentionally weak.
        return _robust_similarity_component(max(similarities), similarities)


class LocalLLMKnowledgeScorer:
    """Local transformers text-generation scorer using the same JSON schema."""

    def __init__(self, model: str | None = None, max_new_tokens: int = 400) -> None:
        if not model:
            raise RuntimeError("--local-llm-model is required when --scorer local_llm")
        self.model = model
        self.max_new_tokens = max_new_tokens
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - optional runtime behavior
            raise RuntimeError("Install transformers and torch to use LocalLLMKnowledgeScorer") from exc
        self.generator = pipeline("text-generation", model=model)

    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        if not documents:
            return _empty_score(gene, "No PubMed evidence retrieved for the requested context.")
        prompt = _build_prompt(gene, context, documents)
        response = self.generator(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = response[0].get("generated_text", "")
        json_text = _extract_json_object(text[len(prompt) :] if text.startswith(prompt) else text)
        return _score_from_json(gene, _loads_json(json_text))


class OpenAIKnowledgeScorer:
    def __init__(self, model: str = "gpt-5.4-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIKnowledgeScorer")
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("Install the openai package to use OpenAIKnowledgeScorer") from exc
        self.client = OpenAI(api_key=self.api_key)

    def score(self, gene: str, context: str, documents: list[EvidenceDocument]) -> KnowledgeScore:
        if not documents:
            return KnowledgeScore(
                gene=gene,
                knowledge_score=0.0,
                evidence_level="none",
                evidence_summary="No PubMed evidence retrieved for the requested context.",
                caveats="Absence of retrieved abstracts is not proof of absence in all literature.",
                supporting_pmids=[],
                species_context_match="none",
            )
        prompt = _build_prompt(gene, context, documents)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious biomedical evidence curator. "
                        "Return only valid JSON matching the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = response.output_text
        data = _loads_json(text)
        return _score_from_json(gene, data)


def create_knowledge_scorer(
    name: str,
    model: str = "gpt-5.4-mini",
    local_llm_model: str | None = None,
    pubmedbert_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
) -> KnowledgeScorer:
    normalized = name.replace("-", "_").lower()
    if normalized == "openai":
        return OpenAIKnowledgeScorer(model=model)
    if normalized == "heuristic":
        return HeuristicKnowledgeScorer()
    if normalized == "embedding":
        return EmbeddingKnowledgeScorer()
    if normalized == "bm25":
        return BM25KnowledgeScorer()
    if normalized == "pubmed_rank":
        return PubMedRankScorer()
    if normalized == "go":
        return GOAnnotationScorer()
    if normalized == "pubmedbert":
        return PubMedBERTKnowledgeScorer(model=pubmedbert_model)
    if normalized in {"go_pubmed", "hybrid"}:
        return GOPubMedHybridScorer()
    if normalized == "offline_judge":
        return OfflineEvidenceJudgeScorer(pubmedbert_model=pubmedbert_model)
    if normalized == "local_llm":
        return LocalLLMKnowledgeScorer(model=local_llm_model)
    if normalized == "count":
        return LiteratureCountScorer()
    if normalized == "mock":
        return MockScorer()
    raise ValueError(f"Unsupported scorer: {name}")


def _build_prompt(gene: str, context: str, documents: list[EvidenceDocument]) -> str:
    docs_text = "\n\n".join(doc.compact_text() for doc in documents[:20])
    return f"""
Score how strongly gene {gene} should be prioritized for this biological context by combining:
1. The provided PubMed/GO evidence packet.
2. Your own biomedical knowledge about this gene, pathway, cell state, species, and experimental context.

Use the provided evidence as the main auditable source, but you may use well-established biomedical knowledge even when it is not explicitly quoted in the packet. Keep the two sources conceptually separate in your rationale.

Context:
{context}

Evidence:
{docs_text}

Return JSON with exactly these fields:
{{
  "knowledge_score": number from 0 to 1,
  "literature_evidence_score": number from 0 to 1,
  "model_prior_score": number from 0 to 1,
  "evidence_level": "none" | "association" | "in_vitro" | "in_vivo" | "direct_context",
  "evidence_summary": short string,
  "model_prior_summary": short string,
  "caveats": short string,
  "supporting_pmids": array of PMID strings,
  "species_context_match": "none" | "weak" | "partial" | "strong"
}}

Scoring rules:
- literature_evidence_score should reflect only the provided evidence packet.
- model_prior_score should reflect your broader biomedical knowledge, including canonical roles not shown in the packet.
- knowledge_score is the final combined priority score. Weight strong provided evidence highest, but let strong model prior knowledge rescue well-established genes when the evidence packet is incomplete.
- 0 means neither provided evidence nor model knowledge supports relevance in the biological context.
- 0.2-0.4 means broad association, generic expression, or weak model prior only.
- 0.4-0.7 means functional evidence or model knowledge in a related system.
- 0.7-0.9 means strong functional evidence or strong prior knowledge in a close lineage, cell type, perturbation, or developmental context.
- 0.9-1.0 means direct functional support or canonical prior knowledge in the same or very close fate-decision context.
- Penalize unrelated species, unrelated disease-only evidence, same gene symbol ambiguity, and evidence that only mentions the gene without functional support.
- Cite only PMIDs present in the evidence packet in supporting_pmids. If model prior knowledge contributes, describe it in model_prior_summary rather than inventing citations.
""".strip()


def _empty_score(gene: str, summary: str) -> KnowledgeScore:
    return KnowledgeScore(
        gene=gene,
        knowledge_score=0.0,
        evidence_level="none",
        evidence_summary=summary,
        caveats="Absence of retrieved abstracts is not proof of absence in all literature.",
        supporting_pmids=[],
        species_context_match="none",
    )


def _classify_document(gene: str, context: str, doc: EvidenceDocument) -> dict[str, object]:
    text = _doc_text(doc, gene)
    lower = text.lower()
    matched: list[str] = []
    score = 0.05 if _gene_mentioned(gene, text) else 0.0
    level = "association" if score else "none"
    if any(term in lower for term in _association_terms()):
        score = max(score, 0.25)
        level = "association"
        matched.append("association")
    if any(term in lower for term in _in_vitro_terms()):
        score = max(score, 0.5)
        level = "in_vitro"
        matched.append("in_vitro")
    if any(term in lower for term in _in_vivo_terms()):
        score = max(score, 0.72)
        level = "in_vivo"
        matched.append("in_vivo")
    context_strength = _context_keyword_strength(gene, context, text)
    has_direct_context = any(term in lower for term in _direct_context_terms())
    has_in_vivo_evidence = any(term in lower for term in _in_vivo_terms())
    has_same_lineage_evidence = "paraxial mesoderm" in lower and _gene_mentioned(gene, text) and has_in_vivo_evidence
    if (context_strength >= 0.55 and has_direct_context) or has_same_lineage_evidence:
        score = max(score, 0.9)
        level = "direct_context"
        matched.append("direct_context")
    elif context_strength >= 0.35:
        score += 0.08
        matched.append("context")
    if any(term in lower for term in _species_terms()):
        score += 0.05
        matched.append("mouse")
    score = min(1.0, score)
    if not matched:
        matched.append("gene mention")
    return {
        "pmid": doc.pmid,
        "score": score,
        "level": level,
        "matched": ",".join(matched),
        "species_context_match": _species_context_match(text),
    }


def _context_keyword_strength(gene: str, context: str, text: str) -> float:
    lower = text.lower()
    gene_terms = set(_split_terms(_gene_query_text(gene)))
    terms = [term for term in _split_terms(context) if len(term) >= 4]
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if term not in gene_terms and term in lower)
    return min(1.0, hits / max(1, min(len(terms), 10)))


def _go_overlap_score(gene: str, context: str) -> tuple[float, str, str]:
    annotations = _fetch_go_annotations(gene)
    if not annotations:
        return 0.0, f"No GO annotations found for {gene}", ""
    context_terms = set(term for term in _split_terms(context) if len(term) >= 4)
    scored = []
    for annotation in annotations:
        text = f"{annotation.get('term', '')} {annotation.get('evidence', '')}"
        terms = set(term for term in _split_terms(text) if len(term) >= 4)
        scored.append((annotation, len(context_terms & terms)))
    best, best_overlap = max(scored, key=lambda item: item[1])
    overlap_score = min(0.75, best_overlap / max(1, min(len(context_terms), 8)) * 0.75)
    count_boost = min(0.25, math.log1p(len(annotations)) / math.log1p(30) * 0.25)
    score = min(1.0, overlap_score + count_boost)
    term = str(best.get("term", ""))
    summary = f"GO found {len(annotations)} annotations with best overlap {best_overlap}"
    return score, summary, term


def _go_specificity_score(gene: str, context: str, annotations: tuple[dict, ...]) -> tuple[float, float, float, str, str]:
    if not annotations:
        return 0.0, 0.0, 0.0, f"No GO annotations found for {gene}", ""
    context_terms = set(term for term in _split_terms(context) if len(term) >= 4)
    best_record: dict | None = None
    best_score = 0.0
    best_overlap = 0
    generic_penalty = 0.0
    for annotation in annotations:
        term = str(annotation.get("term", ""))
        terms = set(term for term in _split_terms(term) if len(term) >= 4)
        overlap = len(context_terms & terms)
        namespace_weight = _go_namespace_weight(str(annotation.get("namespace", "")))
        evidence_weight = _go_evidence_weight(str(annotation.get("evidence", "")))
        generic = _generic_go_penalty(term)
        score = min(1.0, overlap / max(1, min(len(context_terms), 8))) * namespace_weight * evidence_weight
        score = max(0.0, score - generic)
        if score > best_score:
            best_score = score
            best_record = annotation
            best_overlap = overlap
            generic_penalty = generic
    annotation_count_boost = min(0.25, math.log1p(len(annotations)) / math.log1p(40) * 0.25)
    go_score = min(1.0, best_score + annotation_count_boost)
    term = str(best_record.get("term", "")) if best_record else ""
    summary = f"GO found {len(annotations)} annotations; best term='{term}' overlap={best_overlap}"
    return go_score, best_score, generic_penalty, summary, term


def _go_namespace_weight(namespace: str) -> float:
    return {"BP": 1.0, "MF": 0.75, "CC": 0.45}.get(namespace, 0.6)


def _go_evidence_weight(evidence: str) -> float:
    code = evidence.upper()
    if code in {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}:
        return 1.0
    if code in {"TAS", "IC"}:
        return 0.85
    if code in {"ISS", "ISO", "ISA", "ISM", "IGC", "RCA"}:
        return 0.65
    if code == "IEA":
        return 0.35
    return 0.55


def _generic_go_penalty(term: str) -> float:
    lower = term.lower()
    generic_terms = (
        "gene expression",
        "regulation of transcription",
        "dna-binding transcription factor activity",
        "rna polymerase ii transcription",
        "biological process",
        "molecular function",
        "cellular process",
        "nucleus",
    )
    return 0.25 if any(generic in lower for generic in generic_terms) else 0.0


def _pubmed_text_evidence_score(
    gene: str,
    context: str,
    documents: list[EvidenceDocument],
) -> tuple[float, str, str, list[tuple[EvidenceDocument, float]], str]:
    if not documents:
        return 0.0, "none", "No PubMed evidence retrieved", [], "none"
    query_terms = _split_terms(f"{_gene_query_text(gene)} {context}")
    doc_terms = [_split_terms(_doc_text(doc, gene)) for doc in documents]
    bm25 = _bm25_scores(query_terms, doc_terms)
    records = [_classify_document(gene, context, doc) for doc in documents]
    hybrid_scores = []
    for doc, lexical, record in zip(documents, bm25, records):
        lexical_score = _normalize_positive(lexical, bm25)
        keyword_score = float(record["score"])
        hybrid_scores.append((doc, 0.55 * lexical_score + 0.45 * keyword_score))
    ranked = sorted(hybrid_scores, key=lambda item: item[1], reverse=True)
    best_doc, best_score = ranked[0]
    best_record = max(records, key=lambda item: item["score"])
    full_text_count = sum(1 for doc in documents if doc.full_text)
    summary = (
        f"PubMed found {len(documents)} filtered records "
        f"({full_text_count} with PMC full text) with best hybrid text score {best_score:.3f}"
    )
    return (
        min(1.0, best_score),
        str(best_record["level"]),
        summary,
        ranked,
        _species_context_match(_doc_text(best_doc, gene)),
    )


def _experiment_strength_score(gene: str, context: str, documents: list[EvidenceDocument]) -> tuple[float, str]:
    if not documents:
        return 0.0, "none"
    records = [_classify_document(gene, context, doc) for doc in documents]
    best = max(records, key=lambda item: item["score"])
    return float(best["score"]), str(best["level"])


def _species_context_score(documents: list[EvidenceDocument]) -> float:
    if not documents:
        return 0.0
    values = []
    for doc in documents:
        match = _species_context_match(_doc_text(doc))
        values.append({"strong": 1.0, "partial": 0.7, "weak": 0.35, "none": 0.0}.get(match, 0.0))
    return max(values) if values else 0.0


def _offline_evidence_level(
    has_go: bool,
    has_pubmed: bool,
    pubmed_level: str,
    experiment_level: str,
    species_context_score: float,
    go_specificity_score: float,
) -> str:
    strongest = _strongest_level(pubmed_level, experiment_level)
    if strongest == "direct_context" and species_context_score >= 0.7:
        return "direct_context"
    if strongest == "in_vivo" and species_context_score >= 0.35:
        return "in_vivo"
    if strongest == "in_vitro":
        return "in_vitro"
    if strongest == "association":
        return "association"
    if has_pubmed:
        return "mention"
    if has_go and go_specificity_score > 0:
        return "go_only"
    if has_go:
        return "association"
    return "none"


def _strongest_level(*levels: str) -> str:
    order = {"none": 0, "mention": 1, "association": 2, "in_vitro": 3, "in_vivo": 4, "direct_context": 5}
    return max(levels, key=lambda level: order.get(level, 0))


def _evidence_cap(level: str) -> float:
    return {
        "none": 0.0,
        "mention": 0.25,
        "association": 0.40,
        "go_only": 0.45,
        "in_vitro": 0.65,
        "in_vivo": 0.85,
        "direct_context": 1.0,
    }.get(level, 0.4)


def _calibrate_offline_score(
    *,
    raw: float,
    evidence_level: str,
    evidence_cap: float,
    pubmed_text_score: float,
    experiment_score: float,
    species_context_score: float,
    go_specificity_score: float,
    functional_context_score: float,
) -> tuple[float, float]:
    """Apply evidence-level floors after component scoring.

    The weighted components are useful for ranking weak evidence, but they
    should not make direct same-context in vivo evidence look merely moderate
    just because one optional component, such as GO, is unavailable.
    """
    floor = 0.0
    if evidence_level == "direct_context":
        if functional_context_score >= 0.75:
            floor = 0.90
        if functional_context_score >= 0.90 and species_context_score >= 1.0 and experiment_score >= 0.90 and pubmed_text_score >= 0.85:
            floor = 0.95
        if go_specificity_score >= 0.50:
            floor = max(floor, 0.90)
    score = min(max(raw, floor), evidence_cap)
    return score, floor


def _robust_similarity_component(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _bm25_scores(query_terms: list[str], documents: list[list[str]]) -> list[float]:
    k1 = 1.5
    b = 0.75
    n_docs = len(documents)
    avgdl = sum(len(doc) for doc in documents) / max(1, n_docs)
    doc_freq: dict[str, int] = {}
    for doc in documents:
        for term in set(doc):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    scores = []
    for doc in documents:
        length = len(doc) or 1
        counts: dict[str, int] = {}
        for term in doc:
            counts[term] = counts.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * length / max(avgdl, 1e-9))
            score += idf * (freq * (k1 + 1) / denom)
        scores.append(score)
    return scores


def _normalize_positive(value: float, values: list[float]) -> float:
    hi = max(values) if values else 0.0
    if hi <= 0:
        return 0.0
    return max(0.0, min(1.0, value / hi))


@lru_cache(maxsize=4096)
def _fetch_go_annotations(gene: str) -> tuple[dict, ...]:
    try:
        import requests

        hits = []
        for symbol in _gene_lookup_symbols(gene):
            response = requests.get(
                "https://mygene.info/v3/query",
                params={
                    "q": f"symbol:{symbol}",
                    "species": "mouse",
                    "fields": "go",
                    "size": 1,
                },
                timeout=20,
            )
            response.raise_for_status()
            hits = response.json().get("hits", [])
            if hits and hits[0].get("go"):
                break
    except Exception:
        return tuple()
    if not hits:
        return tuple()
    go = hits[0].get("go", {})
    records: list[dict] = []
    for namespace in ("BP", "MF", "CC"):
        items = go.get(namespace, [])
        if isinstance(items, dict):
            items = [items]
        for item in items:
            if isinstance(item, dict):
                record = dict(item)
                record["namespace"] = namespace
                records.append(record)
    return tuple(records)


def _species_context_match(text: str) -> str:
    lower = text.lower()
    has_species = any(term in lower for term in _species_terms())
    has_context = any(term in lower for term in _direct_context_terms())
    if has_species and has_context:
        return "strong"
    if has_context:
        return "partial"
    if has_species:
        return "weak"
    return "none"


def _functional_context_score(gene: str, documents: list[EvidenceDocument]) -> float:
    scores = []
    for doc in documents:
        text = _doc_text(doc, gene)
        lower = text.lower()
        if not _gene_mentioned(gene, text):
            continue
        lineage = 0.0
        if any(term in lower for term in ("neuromesodermal", "nmp", "paraxial mesoderm", "presomitic", "somite", "somitogenesis")):
            lineage = 1.0
        elif any(term in lower for term in ("tailbud", "axial elongation", "gastrulation", "primitive streak")):
            lineage = 0.8
        elif any(term in lower for term in ("spinal cord", "neural tube")):
            lineage = 0.5
        if lineage == 0.0:
            scores.append(0.0)
            continue
        functional = any(term in lower for term in _functional_evidence_terms())
        expression_only = any(term in lower for term in ("expression of", "is expressed", "are expressed", "expression pattern"))
        if functional:
            scores.append(lineage)
        elif expression_only:
            scores.append(min(lineage, 0.45))
        else:
            scores.append(min(lineage, 0.65))
    return max(scores) if scores else 0.0


def _context_mismatch_penalty(gene: str, documents: list[EvidenceDocument]) -> float:
    if not documents:
        return 0.0
    penalties = []
    for doc in documents:
        text = _doc_text(doc, gene)
        lower = text.lower()
        if _functional_context_score(gene, [doc]) >= 0.75:
            penalties.append(0.0)
            continue
        penalty = 0.0
        if any(term in lower for term in _off_context_terms()):
            penalty += 0.12
        if any(term in lower for term in _injury_disease_terms()):
            penalty += 0.10
        expression_only = any(term in lower for term in ("expression pattern", "is expressed", "are expressed", "marker"))
        if expression_only and not any(term in lower for term in _functional_evidence_terms()):
            penalty += 0.05
        penalties.append(min(0.25, penalty))
    return min(penalties) if penalties else 0.0


def _level_from_score(score: float) -> str:
    if score >= 0.85:
        return "direct_context"
    if score >= 0.65:
        return "in_vivo"
    if score >= 0.4:
        return "in_vitro"
    if score > 0:
        return "association"
    return "none"


def _doc_text(doc: EvidenceDocument, gene: str | None = None) -> str:
    full_text = _focused_full_text(doc.full_text, gene) if doc.full_text else ""
    return f"{doc.title} {doc.abstract} {full_text}".strip()


def _focused_full_text(text: str, gene: str | None = None, max_chars: int = 8000) -> str:
    if not text:
        return ""
    terms = set(_direct_context_terms()) | set(_functional_evidence_terms())
    if gene:
        terms.update(alias.lower() for alias in _gene_aliases(gene))
    lower = text.lower()
    windows: list[str] = []
    for term in sorted(terms, key=len, reverse=True):
        start = 0
        while True:
            idx = lower.find(term, start)
            if idx == -1:
                break
            left = max(0, idx - 450)
            right = min(len(text), idx + 650)
            snippet = " ".join(text[left:right].split())
            if snippet and snippet not in windows:
                windows.append(snippet)
            start = idx + len(term)
            if sum(len(window) for window in windows) >= max_chars:
                return " ".join(windows)[:max_chars]
    if windows:
        return " ".join(windows)[:max_chars]
    return " ".join(text.split())[:2500]


def _gene_aliases(gene: str) -> tuple[str, ...]:
    aliases = {
        "T": ("T", "Tbxt", "Brachyury"),
    }
    return aliases.get(gene, (gene,))


def _gene_lookup_symbols(gene: str) -> tuple[str, ...]:
    if gene == "T":
        return ("Tbxt", "T")
    return (gene,)


def _gene_query_text(gene: str) -> str:
    return " ".join(_gene_aliases(gene))


def _gene_mentioned(gene: str, text: str) -> bool:
    lower = text.lower()
    tokens = set(_split_terms(text))
    for alias in _gene_aliases(gene):
        alias_lower = alias.lower()
        if len(alias_lower) <= 2:
            if alias_lower in tokens:
                return True
        elif alias_lower in lower:
            return True
    return False


def _split_terms(text: str) -> list[str]:
    clean = "".join(char.lower() if char.isalnum() else " " for char in text)
    return [term for term in clean.split() if term]


def _species_terms() -> tuple[str, ...]:
    return ("mouse", "murine", "mus musculus")


def _direct_context_terms() -> tuple[str, ...]:
    return (
        "neuromesodermal",
        "nmp",
        "paraxial mesoderm",
        "spinal cord",
        "somite",
        "presomitic",
        "tailbud",
    )


def _association_terms() -> tuple[str, ...]:
    return ("expression", "expressed", "marker", "associated", "upregulated", "downregulated", "signature")


def _in_vitro_terms() -> tuple[str, ...]:
    return ("in vitro", "cell culture", "cultured", "organoid", "differentiation assay", "reporter assay")


def _in_vivo_terms() -> tuple[str, ...]:
    return (
        "in vivo",
        "embryo",
        "embryonic",
        "knockout",
        "mutant",
        "conditional deletion",
        "lineage tracing",
        "loss-of-function",
        "gain-of-function",
    )


def _off_context_terms() -> tuple[str, ...]:
    return (
        "immune",
        "t cell",
        "b cell",
        "macrophage",
        "microglia",
        "tumor",
        "cancer",
        "carcinoma",
        "lymphoma",
        "leukemia",
        "gonad",
        "ovary",
        "testis",
        "kidney",
        "retina",
        "photoreceptor",
        "heart",
        "cardiac",
        "lung",
        "pancreas",
        "thyroid",
    )


def _injury_disease_terms() -> tuple[str, ...]:
    return (
        "injury",
        "disease",
        "inflammation",
        "neuroinflammation",
        "fibrosis",
        "pain",
        "autoimmune",
        "infection",
        "metastasis",
        "autism",
        "tumor",
        "cancer",
    )


def _functional_evidence_terms() -> tuple[str, ...]:
    return (
        "required",
        "requirement",
        "regulates",
        "regulate",
        "controls",
        "control",
        "disrupts",
        "disruption",
        "mutant",
        "knockout",
        "loss-of-function",
        "gain-of-function",
        "ectopic expression",
        "conditional deletion",
        "lineage tracing",
        "specification",
        "specifies",
    )


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return stripped
    return stripped[start : end + 1]


def _loads_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return json.loads(stripped)


def _score_from_json(gene: str, data: dict) -> KnowledgeScore:
    extra = {}
    for key in ("literature_evidence_score", "model_prior_score", "model_prior_summary"):
        if key in data:
            extra[key] = data[key]
    return KnowledgeScore(
        gene=gene,
        knowledge_score=float(data.get("knowledge_score", 0.0)),
        evidence_level=str(data.get("evidence_level", "none")),
        evidence_summary=str(data.get("evidence_summary", "")),
        caveats=str(data.get("caveats", "")),
        supporting_pmids=[str(pmid) for pmid in data.get("supporting_pmids", [])],
        species_context_match=str(data.get("species_context_match", "unknown")),
        extra=extra,
    )
