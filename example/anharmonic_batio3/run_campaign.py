"""Run the full BaTiO3 anharmonic-LWF acceptance campaign (story 023).

Usage (from the repository root or this directory, with the lawaf
environment active)::

    python run_campaign.py [--outdir outputs]

Produces, under ``outputs/``:
- ``bato3_anharmonic_model.nc``  netCDF with the ``anharmonic`` AND
  ``symmetry`` groups (loads standalone)
- ``results.json`` / ``results.md``  every measured gate number
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from campaign import (
    REDUCED_CONFIG,
    REDUCED_CONFIG_2X2X2,
    TWO_BY_TWO_CONFIG,
    CampaignConfig,
    run,
    run_2x2x2,
    write_reports,
)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    is_2x2x2 = "--2x2x2" in argv
    outdir = Path("outputs_2x2x2" if is_2x2x2 else "outputs")
    if "--outdir" in argv:
        outdir = Path(argv[argv.index("--outdir") + 1])
    if "--orders" in argv:
        orders = tuple(
            int(v) for v in argv[argv.index("--orders") + 1].split(",")
        )
    else:
        orders = None
    if is_2x2x2:
        cfg = REDUCED_CONFIG_2X2X2 if "--reduced" in argv else TWO_BY_TWO_CONFIG
        runner = run_2x2x2
    else:
        cfg = REDUCED_CONFIG if "--reduced" in argv else CampaignConfig()
        runner = run
    if orders is not None:
        cfg = replace(cfg, orders=orders)
    results = runner(cfg, outdir=outdir)
    paths = write_reports(results, outdir)
    print(f"results: {paths['json']}")
    print(f"report : {paths['markdown']}")
    print("gates  :", results["gates_summary"])
    print("ALL PASS" if results["all_gates_pass"] else "SOME GATES FAILED")
    return 0 if results["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
