"""Example 2: prioritize directed gene pairs.

Two kinds of pairs are scored with the same function:
  * TF -> target edges from a GRN (mouse Muller glia context)
  * ligand -> receptor pairs from a cell-cell communication tool such as CellChat (tumor context)

    python examples/02_score_gene_pairs.py
    python examples/02_score_gene_pairs.py --provider deepseek --model deepseek-chat
"""
from _common import DATA, llm_settings, parse_args

from fatereasoner import GenePairScoringConfig, score_gene_pairs

args = parse_args("Score TF -> target and ligand -> receptor candidates.")
out = args.out or "results/02_gene_pairs"

runs = [
    ("tf_target", DATA / "gene_pairs.csv", DATA / "retina_context.txt"),
    ("ligand_receptor", DATA / "ligand_receptor_pairs.csv", DATA / "tumor_context.txt"),
]
for name, pairs, context in runs:
    result = score_gene_pairs(
        pairs=pairs,
        context=context,
        out=f"{out}/{name}",
        config=GenePairScoringConfig(**llm_settings(args)),
    )
    print(f"\n== {name} ==")
    if result.result is None:
        print(result.packet_table[["candidate_id", "retrieved_count", "evidence_count"]])
    else:
        cols = ["candidate_id", "data_score", "fatereasoner_score", "evidence_quadrant"]
        print(result.result[[c for c in cols if c in result.result.columns]])
