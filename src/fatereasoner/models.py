from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Candidate:
    gene: str
    data_score: float
    statistics: dict[str, Any] = field(default_factory=dict)
    group: str | None = None


@dataclass
class EvidenceDocument:
    pmid: str
    title: str
    abstract: str
    journal: str | None = None
    year: str | None = None
    pubmed_rank: int | None = None
    pmcid: str | None = None
    full_text: str = ""
    full_text_source: str = ""

    def compact_text(self, max_chars: int = 1400) -> str:
        evidence = self.full_text if self.full_text else self.abstract
        label = "PMC Full Text" if self.full_text else "Abstract"
        pmc = f"\nPMCID: {self.pmcid}" if self.pmcid else ""
        text = f"PMID: {self.pmid}{pmc}\nTitle: {self.title}\n{label}: {evidence}"
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."


@dataclass
class KnowledgeScore:
    gene: str
    knowledge_score: float
    evidence_level: str
    evidence_summary: str
    caveats: str = ""
    supporting_pmids: list[str] = field(default_factory=list)
    species_context_match: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {}) or {}
        data["GENE"] = data["gene"]
        data["knowledge_score"] = float(max(0.0, min(1.0, data["knowledge_score"])))
        data["supporting_pmids"] = ";".join(data["supporting_pmids"])
        data.update(extra)
        return data


@dataclass
class DistilledEvidencePacket:
    gene: str
    context: str
    distilled_text: str
    selected_pmids: list[str] = field(default_factory=list)
    go_terms: list[str] = field(default_factory=list)
    generif_statements: list[str] = field(default_factory=list)
    pubtator_entities: list[str] = field(default_factory=list)
    method: str = "lite"
    token_estimate: int = 0

    def to_evidence_document(self) -> EvidenceDocument:
        pmids = ";".join(self.selected_pmids)
        title = f"Distilled evidence packet for {self.gene}"
        abstract = "\n".join(
            line
            for line in self.distilled_text.splitlines()
            if not line.startswith("Gene: ") and not line.startswith("Context: ")
        ).strip()
        if pmids:
            abstract = f"{abstract}\nSelected PMIDs: {pmids}"
        return EvidenceDocument(
            pmid="distilled",
            title=title,
            abstract=abstract,
            journal="FateReasoner evidence distillation",
            year=None,
        )

    def to_columns(self) -> dict[str, Any]:
        return {
            "distilled_text": self.distilled_text,
            "distilled_pmids": ";".join(self.selected_pmids),
            "distilled_go_terms": ";".join(self.go_terms),
            "distilled_generif_statements": " | ".join(self.generif_statements),
            "distilled_pubtator_entities": ";".join(self.pubtator_entities),
            "distilled_token_estimate": self.token_estimate,
            "distillation_method": self.method,
        }
