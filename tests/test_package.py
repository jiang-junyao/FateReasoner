from __future__ import annotations

import pandas as pd
import pytest

import fatereasoner
from fatereasoner import (
    GenePairScoringConfig,
    GeneScoringConfig,
    GRNModuleScoringConfig,
    score_gene_pairs,
    score_genes,
    score_grn_modules,
)
from fatereasoner import distilled_prior


def test_public_api():
    assert fatereasoner.__version__
    assert "claude" in fatereasoner.PROVIDERS


def test_cli_help(capsys):
    from fatereasoner.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    assert "score-genes" in capsys.readouterr().out


def test_score_genes_prepare_only(tmp_path, data_dir):
    result = score_genes(
        genes=data_dir / "hesc_candidate_genes.csv",
        gene_col="GENE",
        context=data_dir / "hesc_context.txt",
        out=tmp_path,
        config=GeneScoringConfig(prepare_only=True, delay_seconds=0),
    )
    assert result.scores is None
    assert set(result.packet_table["GENE"]) >= {"TBXT", "EOMES"}
    assert result.packets_path.exists()


def test_score_gene_pairs_prepare_only(tmp_path, data_dir):
    result = score_gene_pairs(
        pairs=data_dir / "ligand_receptor_pairs.csv",
        context=data_dir / "tumor_context.txt",
        out=tmp_path,
        config=GenePairScoringConfig(prepare_only=True, delay_seconds=0),
    )
    assert len(result.packet_table) == 4
    assert (result.candidates["relation_type"] == "ligand_receptor").all()


def test_score_grn_modules_prepare_only(tmp_path, data_dir):
    result = score_grn_modules(
        network=data_dir / "grn_modules.tsv",
        species="mouse",
        context=data_dir / "retina_context.txt",
        out=tmp_path,
        config=GRNModuleScoringConfig(prepare_only=True, top_module_pairs=5, delay_seconds=0),
    )
    assert 0 < len(result.candidates) <= 5
    assert result.packets_path.exists()


def test_scoring_requires_provider(tmp_path, data_dir):
    with pytest.raises(ValueError, match="No LLM provider"):
        score_genes(
            genes=["TBXT", "SOX17"],
            context="human embryonic stem cell differentiation toward definitive endoderm",
            out=tmp_path,
            config=GeneScoringConfig(api_key="dummy", delay_seconds=0),
        )


def test_score_genes_with_fake_llm(tmp_path, monkeypatch):
    def fake_llm(**kwargs):
        genes = [line.split(":", 1)[1].strip() for line in kwargs["prompt"].splitlines() if line.startswith("## GENE:")]
        rows = ["GENE,method,score,evidence_level,supporting_pmids,rationale,caveats"]
        rows += [f"{g},distilled_plus_prior,{0.9 if g != 'ACTB' else 0.05},strong,1001,test,none" for g in genes]
        return "\n".join(rows), {"fake": True}

    monkeypatch.setattr(distilled_prior, "call_llm", fake_llm)
    result = score_genes(
        genes=["TBXT", "SOX17", "ACTB"],
        context="human embryonic stem cell differentiation toward definitive endoderm",
        out=tmp_path,
        config=GeneScoringConfig(provider="claude", api_key="dummy", delay_seconds=0),
    )
    scores = pd.read_csv(result.result_path)
    assert list(scores["GENE"]) == ["TBXT", "SOX17", "ACTB"]
    assert scores.set_index("GENE").loc["ACTB", "score"] < 0.5


def test_full_text_is_default_and_attached(monkeypatch):
    from fatereasoner.models import EvidenceDocument
    from fatereasoner.retrieval import PubMedClient

    assert GeneScoringConfig().full_text == "pmc-oa"
    assert GenePairScoringConfig().full_text == "pmc-oa"
    assert GRNModuleScoringConfig().full_text == "pmc-oa"

    def fake_fetch(self, pmids):
        return [EvidenceDocument("2001", "title", "abstract", "J", "2024", pmcid="PMC123")]

    monkeypatch.setattr(PubMedClient, "fetch", fake_fetch)
    monkeypatch.setattr(PubMedClient, "fetch_pmc_full_texts", lambda self, ids: {"PMC123": "full text body"})
    client = PubMedClient(delay_seconds=0)
    docs = client.fetch_with_full_text(["2001"])
    assert docs[0].full_text == "full text body"
    assert PubMedClient(full_text_mode="none", delay_seconds=0).fetch_with_full_text(["2001"])[0].full_text == ""
