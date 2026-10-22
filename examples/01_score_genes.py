"""Example 1: prioritize candidate genes (e.g. DEGs or TFs) for a cell state.

    python examples/01_score_genes.py                      # evidence only, no API key
    python examples/01_score_genes.py --provider claude    # full scoring (needs ANTHROPIC_API_KEY)
"""
from _common import DATA, llm_settings, parse_args

from fatereasoner import GeneScoringConfig, score_genes

args = parse_args("Score candidate genes for a definitive-endoderm differentiation context.")
result = score_genes(
    genes=DATA / "hesc_candidate_genes.csv",
    gene_col="GENE",
    context=DATA / "hesc_context.txt",
    out=args.out or "results/01_genes",
    config=GeneScoringConfig(**llm_settings(args)),
)

if result.result is None:
    print(result.packet_table[["GENE", "retrieved_count", "evidence_count", "distilled_pmids"]])
    print(f"Evidence packets written to {result.packets_path}")
else:
    print(result.result.sort_values("score", ascending=False)[["GENE", "score", "evidence_level", "supporting_pmids"]])
    print(f"Scores written to {result.result_path}")
