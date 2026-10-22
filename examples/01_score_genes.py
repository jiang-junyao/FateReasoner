"""Score candidate genes or TFs (run from the repository root; needs DEEPSEEK_API_KEY)."""
from fatereasoner import GeneScoringConfig, score_genes

result = score_genes("examples/data/hesc_candidate_genes.csv", "examples/data/hesc_context.txt", "results/genes", gene_col="GENE", config=GeneScoringConfig(provider="deepseek"))
print(result.result)
