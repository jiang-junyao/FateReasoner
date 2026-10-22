"""FateReasoner: evidence-grounded LLM reasoning for prioritizing cell-fate regulators.

Public API
----------
score_genes        Score candidate genes (e.g. differentially expressed genes, TFs).
score_gene_pairs   Score directed gene pairs (TF -> target edges, ligand -> receptor pairs).
score_grn_modules  Score module -> module relationships in a gene regulatory network.
"""

from .api import (
    GenePairScoringConfig,
    GeneScoringConfig,
    GRNModuleScoringConfig,
    ScoringResult,
    score_gene_pairs,
    score_genes,
    score_grn_modules,
)
from .llm import PROVIDERS

__version__ = "0.1.0"

__all__ = [
    "GeneScoringConfig",
    "GenePairScoringConfig",
    "GRNModuleScoringConfig",
    "PROVIDERS",
    "ScoringResult",
    "score_gene_pairs",
    "score_genes",
    "score_grn_modules",
    "__version__",
]
