"""Offline fixtures: a tiny fake PubMed corpus and a fake LLM, so tests need no network or API key."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from fatereasoner.models import EvidenceDocument
from fatereasoner.retrieval import PubMedClient

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"

CORPUS = [
    EvidenceDocument("1001", "TBXT and MIXL1 mark primitive streak in human embryonic stem cell differentiation",
                     "During human embryonic stem cell differentiation, TBXT and MIXL1 are induced in primitive streak-like cells, "
                     "and EOMES directs definitive endoderm specification together with SOX17 and FOXA2.", "Dev Cell", "2020"),
    EvidenceDocument("1002", "GSC and LHX1 in anterior endoderm", "GSC and LHX1 regulate anterior definitive endoderm in human pluripotent stem cells.",
                     "Development", "2019"),
    EvidenceDocument("1003", "NFIA represses Hes1 in neural progenitors", "Nfia regulates Hes1 and controls progenitor differentiation; "
                     "Nfix and Nfib act redundantly. In Muller glia, Nfia, Nfib and Nfix maintain quiescence. Sox9 activates Nfia.",
                     "J Neurosci", "2010"),
    EvidenceDocument("1004", "Pax6 and Lhx2 in retinal progenitors", "Lhx2 and Pax6 regulate retinal gliogenesis; Klf6 and Stat3 drive reactive "
                     "Muller glia after injury; Myc and Sox9 are induced in reactive glia.", "Development", "2016"),
    EvidenceDocument("1005", "THBS1 promotes HNSCC migration through SDC1", "Thrombospondin-1 (THBS1) secreted by fibroblasts binds SDC1 "
                     "to promote migration and invasion of head and neck squamous cell carcinoma.", "Oncogene", "2025"),
    EvidenceDocument("1006", "SPP1-CD44 axis in tumor-associated macrophages", "SPP1+ macrophages signal through CD44 on tumor cells; "
                     "COL1A1 from cancer-associated fibroblasts also engages CD44. APOE and SORL1 are expressed in macrophages.",
                     "Nat Commun", "2023"),
]


@pytest.fixture(autouse=True)
def offline_pubmed(monkeypatch):
    by_id = {doc.pmid: doc for doc in CORPUS}

    def fake_search(self, query: str) -> list[str]:
        symbols = {s.upper() for s in re.findall(r'"?([A-Za-z0-9]+)"?\[(?:Title/Abstract|tiab|Gene Name)\]', query)}
        hits = [doc.pmid for doc in CORPUS if symbols & {w.upper() for w in re.findall(r"[A-Za-z0-9]+", doc.title + " " + doc.abstract)}]
        return hits or [doc.pmid for doc in CORPUS]

    def fake_fetch(self, pmids):
        return [EvidenceDocument(**{**by_id[p].__dict__}) for p in pmids if p in by_id]

    monkeypatch.setattr(PubMedClient, "search", fake_search)
    monkeypatch.setattr(PubMedClient, "fetch", fake_fetch)


@pytest.fixture
def data_dir() -> Path:
    return DATA
