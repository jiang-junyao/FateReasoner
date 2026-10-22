from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import requests

from .models import EvidenceDocument


class PubMedClient:
    """Small Entrez client for PubMed search and abstract fetch."""

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        retmax: int = 8,
        date_cutoff: str | None = None,
        delay_seconds: float = 0.34,
        timeout: int = 30,
        full_text_mode: str = "none",
        exclude_pmids: Iterable[str] | None = None,
    ) -> None:
        self.email = email or os.getenv("NCBI_EMAIL")
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        self.retmax = retmax
        self.date_cutoff = date_cutoff
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.full_text_mode = full_text_mode
        self.exclude_pmids = set(parse_exclude_pmids(exclude_pmids))

    def build_query(self, gene: str, context: str) -> str:
        gene_query = _gene_pubmed_query(gene)
        context_terms = (
            '"mouse"[Title/Abstract] OR "murine"[Title/Abstract] OR "Mus musculus"[Title/Abstract]'
        )
        biology_terms = (
            '"neuromesodermal progenitor"[Title/Abstract] OR NMP[Title/Abstract] OR '
            '"paraxial mesoderm"[Title/Abstract] OR "spinal cord"[Title/Abstract] OR '
            '"embryonic development"[Title/Abstract]'
        )
        return f'({gene_query}) AND ({context_terms}) AND ({biology_terms})'

    def search_and_fetch(self, gene: str, context: str) -> list[EvidenceDocument]:
        query = self.build_query(gene, context)
        ids = self.search(query)
        if not ids:
            broad_query = (
                f'({_gene_pubmed_query(gene)}) AND '
                '("mouse"[Title/Abstract] OR "murine"[Title/Abstract] OR "Mus musculus"[Title/Abstract])'
            )
            ids = self.search(broad_query)
        docs = self.fetch(ids)
        if self.full_text_mode == "pmc-oa":
            self._add_pmc_full_text(docs)
        for rank, doc in enumerate(docs, start=1):
            doc.pubmed_rank = rank
        return docs

    def search(self, query: str) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(self.retmax),
            "sort": "relevance",
        }
        if self.date_cutoff:
            params["datetype"] = "pdat"
            params["maxdate"] = self.date_cutoff.replace("-", "/")
        self._add_common_params(params)
        response = self._get_with_retries("esearch.fcgi", params, f"PubMed esearch failed for query: {query}")
        time.sleep(self.delay_seconds)
        data = response.json()
        ids = [str(pmid) for pmid in data.get("esearchresult", {}).get("idlist", [])]
        return [pmid for pmid in ids if pmid not in self.exclude_pmids]

    def fetch(self, pmids: Iterable[str]) -> list[EvidenceDocument]:
        pmid_list = [str(pmid) for pmid in pmids if str(pmid) and str(pmid) not in self.exclude_pmids]
        if not pmid_list:
            return []
        params = {"db": "pubmed", "id": ",".join(pmid_list), "retmode": "xml"}
        self._add_common_params(params)
        response = self._get_with_retries("efetch.fcgi", params, f"PubMed efetch failed for {len(pmid_list)} PMIDs")
        time.sleep(self.delay_seconds)
        docs = _parse_pubmed_xml(response.text)
        return [doc for doc in docs if doc.pmid not in self.exclude_pmids]

    def _add_pmc_full_text(self, docs: list[EvidenceDocument]) -> None:
        pmcids = [doc.pmcid for doc in docs if doc.pmcid]
        if not pmcids:
            return
        full_texts = self.fetch_pmc_full_texts(pmcids)
        for doc in docs:
            if doc.pmcid and doc.pmcid in full_texts:
                doc.full_text = full_texts[doc.pmcid]
                doc.full_text_source = "pmc-oa"

    def fetch_pmc_full_texts(self, pmcids: Iterable[str]) -> dict[str, str]:
        pmc_list = [str(pmcid).replace("PMC", "") for pmcid in pmcids if str(pmcid)]
        if not pmc_list:
            return {}
        params = {"db": "pmc", "id": ",".join(pmc_list), "retmode": "xml"}
        self._add_common_params(params)
        response = self._get_with_retries("efetch.fcgi", params, f"PMC efetch failed for {len(pmc_list)} PMCID records")
        time.sleep(self.delay_seconds)
        return _parse_pmc_full_text_xml(response.text)

    def _add_common_params(self, params: dict[str, str]) -> None:
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

    def _get_with_retries(self, endpoint: str, params: dict[str, str], error_message: str) -> requests.Response:
        last_error = ""
        for attempt in range(1, 5):
            try:
                response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc.__class__.__name__
                time.sleep(min(8.0, self.delay_seconds * (2**attempt)))
        raise RuntimeError(f"{error_message}; last_error={last_error}") from None


class JsonlRetrievalCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, list[dict[str, str]]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    self._data[record["gene"]] = record.get("documents", [])

    def get(self, gene: str) -> list[EvidenceDocument] | None:
        if gene not in self._data:
            return None
        fields = set(EvidenceDocument.__dataclass_fields__)
        return [EvidenceDocument(**{key: value for key, value in doc.items() if key in fields}) for doc in self._data[gene]]

    def set(self, gene: str, documents: list[EvidenceDocument]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"gene": gene, "documents": [asdict(doc) for doc in documents]}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._data[gene] = record["documents"]


def _parse_pubmed_xml(xml_text: str) -> list[EvidenceDocument]:
    root = ET.fromstring(xml_text)
    docs: list[EvidenceDocument] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article, ".//PMID")
        title = _text(article, ".//ArticleTitle")
        abstract_parts = [node.text or "" for node in article.findall(".//Abstract/AbstractText")]
        abstract = " ".join(part.strip() for part in abstract_parts if part.strip())
        journal = _text(article, ".//Journal/Title") or None
        year = _text(article, ".//PubDate/Year") or None
        pmcid = _article_id(article, "pmc") or None
        if pmid and (title or abstract):
            docs.append(EvidenceDocument(pmid=pmid, title=title, abstract=abstract, journal=journal, year=year, pmcid=pmcid))
    return docs


def parse_exclude_pmids(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[;,]\s*|\s+", value)
    else:
        parts = [str(item) for item in value]
    return [part.strip() for part in parts if part.strip()]


def _parse_pmc_full_text_xml(xml_text: str) -> dict[str, str]:
    root = ET.fromstring(xml_text)
    records: dict[str, str] = {}
    for article in root.findall(".//article"):
        pmcid = _article_id(article, "pmc")
        if not pmcid:
            continue
        sections = []
        for path in (
            ".//abstract",
            ".//body//sec",
            ".//body//p",
            ".//fig//caption",
        ):
            for node in article.findall(path):
                text = " ".join("".join(node.itertext()).split())
                if text and text not in sections:
                    sections.append(text)
        text = " ".join(sections)
        if text:
            records[pmcid] = text[:80000]
    return records


def _article_id(root: ET.Element, id_type: str) -> str:
    aliases = {"pmc": {"pmc", "pmcid"}, "pmid": {"pmid"}}
    accepted = aliases.get(id_type, {id_type})
    for node in root.findall(".//ArticleId"):
        if node.attrib.get("IdType", "").lower() in accepted:
            return "".join(node.itertext()).strip()
    for node in root.findall(".//article-id"):
        if node.attrib.get("pub-id-type", "").lower() in accepted:
            value = "".join(node.itertext()).strip()
            if value and "pmc" in accepted and not value.upper().startswith("PMC"):
                return f"PMC{value}"
            return value
    return ""


def _text(root: ET.Element, path: str) -> str:
    node = root.find(path)
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _gene_pubmed_query(gene: str) -> str:
    aliases = {
        "T": ("T", "Tbxt", "Brachyury"),
    }.get(gene, (gene,))
    return " OR ".join(f'"{alias}"[Title/Abstract]' for alias in aliases)
