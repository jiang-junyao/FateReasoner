# Examples

Each script is a single call to the FateReasoner API on the small input files in [`data`](data). Run them from the repository root after setting an LLM key (here `DEEPSEEK_API_KEY`; any supported provider works).

| Script | Candidates | Input | Context |
| --- | --- | --- | --- |
| `01_score_genes.py` | genes or TFs | `hesc_candidate_genes.csv` | hESC differentiation toward definitive endoderm |
| `02_score_tf_target.py` | TF → target edges | `gene_pairs.csv` | mouse Müller glia returning to quiescence |
| `03_score_grn_modules.py` | module → module regulation | `grn_modules.tsv` | mouse Müller glia returning to quiescence |
| `04_score_ligand_receptor.py` | ligand → receptor pairs | `ligand_receptor_pairs.csv` | oral squamous cell carcinoma |

```bash
python examples/01_score_genes.py
```

To inspect the evidence packets without calling an LLM, replace `provider="deepseek"` with `prepare_only=True`. Results are written to `results/`.
