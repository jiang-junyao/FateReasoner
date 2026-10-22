"""Score TF -> target edges from a GRN (run from the repository root; needs DEEPSEEK_API_KEY)."""
from fatereasoner import GenePairScoringConfig, score_gene_pairs

result = score_gene_pairs("examples/data/gene_pairs.csv", "examples/data/retina_context.txt", "results/tf_target", config=GenePairScoringConfig(provider="deepseek"))
print(result.result)
