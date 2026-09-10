#!/usr/bin/env python3
"""CLI entrypoint for the FPL data pipeline.

    python run_pipeline.py ingest     # fetch all sources, build processed table
    python run_pipeline.py validate   # run the validation suite
    python run_pipeline.py all        # both, in order (default)
"""
import argparse
import logging
import sys

from pipeline import build_dataset, validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", default="all", choices=["ingest", "validate", "all"]
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command in ("ingest", "all"):
        build_dataset.run()

    if args.command in ("validate", "all"):
        results = validate.run()
        if not all(r.passed for r in results):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
