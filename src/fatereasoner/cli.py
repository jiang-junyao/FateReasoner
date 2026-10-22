from __future__ import annotations

import argparse

from . import __version__
from .distilled_prior import add_score_genes_parser
from .gene_pair import add_score_gene_pairs_parser
from .grn import add_score_grn_modules_parser


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fatereasoner",
        description="FateReasoner: evidence-grounded LLM reasoning for prioritizing cell-fate regulators.",
    )
    parser.add_argument("--version", action="version", version=f"fatereasoner {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_score_genes_parser(subparsers)
    add_score_gene_pairs_parser(subparsers)
    add_score_grn_modules_parser(subparsers)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
