"""Score module -> module regulation in a module-annotated GRN (run from the repository root; needs DEEPSEEK_API_KEY)."""
from fatereasoner import GRNModuleScoringConfig, score_grn_modules

result = score_grn_modules("examples/data/grn_modules.tsv", "mouse", "examples/data/retina_context.txt", "results/modules", config=GRNModuleScoringConfig(provider="deepseek"))
print(result.result)
