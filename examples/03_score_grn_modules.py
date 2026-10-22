"""Example 3: score module -> module relationships in a module-annotated GRN.

The network table needs the columns regulator_symbol, source_module, target_symbol and
target_module (expression_correlation and regulation_type are optional). IReNA output for
mouse or zebrafish is also accepted directly.

    python examples/03_score_grn_modules.py
    python examples/03_score_grn_modules.py --provider gpt --model gpt-4.1
"""
from _common import DATA, llm_settings, parse_args

from fatereasoner import GRNModuleScoringConfig, score_grn_modules

args = parse_args("Score GRN module pairs for a Muller glia quiescence context.")
result = score_grn_modules(
    network=DATA / "grn_modules.tsv",
    species="mouse",
    context=DATA / "retina_context.txt",
    out=args.out or "results/03_grn_modules",
    config=GRNModuleScoringConfig(top_module_pairs=10, **llm_settings(args)),
)

if result.result is None:
    print(result.candidates[["candidate_id", "data_score"]])
    print(f"Evidence packets written to {result.packets_path}")
else:
    cols = ["candidate_id", "data_score", "fatereasoner_score", "evidence_quadrant"]
    print(result.result[[c for c in cols if c in result.result.columns]])
