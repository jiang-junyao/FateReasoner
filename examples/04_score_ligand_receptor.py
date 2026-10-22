"""Score ligand -> receptor pairs from cell-cell communication analysis (run from the repository root; needs DEEPSEEK_API_KEY)."""
from fatereasoner import GenePairScoringConfig, score_gene_pairs

result = score_gene_pairs("examples/data/ligand_receptor_pairs.csv", "examples/data/tumor_context.txt", "results/lr", config=GenePairScoringConfig(provider="deepseek"))
print(result.result)
