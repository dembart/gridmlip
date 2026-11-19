"""Command line interface: ``gridmlip dataset ...``."""
import argparse
import sys


def _dataset(args):
    import yaml
    from ase.io import read

    from .dataset.recipe import DEFAULTS, merge_settings, run
    from .dataset.settings import load_settings

    if args.print_config:
        print(yaml.safe_dump(DEFAULTS, sort_keys=False, default_flow_style=False), end="")
        return 0
    if args.input is None:
        print("error: an input structure file is required", file=sys.stderr)
        return 2

    overrides = {k: v for k, v in (("specie", args.specie), ("seed", args.seed)) if v is not None}
    settings = merge_settings(load_settings(args.config) if args.config else {}, overrides)

    try:
        i, n = (int(x) for x in args.chunk.split("/"))
        assert 0 <= i < n
    except Exception:
        print(f"error: --chunk must look like i/n with 0 <= i < n, got {args.chunk!r}", file=sys.stderr)
        return 2

    atoms_list = read(args.input, index=":")
    rows = run(atoms_list, args.output, settings, dry_run=args.dry_run, overwrite=args.overwrite,
               chunk=(i, n), progress=not args.quiet)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"{len(rows)} materials: {ok} ok, {len(rows) - ok} skipped/existing/failed "
          f"(see {args.output}/summary*.csv)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gridmlip")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dataset", help="generate a defect dataset for MLIP training",
                       description="defects -> SOAP -> farthest point sampling -> rattle; "
                                   "one .xyz per material, plus config.yaml and summary.csv")
    p.add_argument("input", nargs="?", help="structures (any file ASE reads; info['material_id'] names outputs)")
    p.add_argument("--config", help="YAML settings (see --print-config); a previous config.yaml works too")
    p.add_argument("--output", default="dataset", help="output directory (default: dataset)")
    p.add_argument("--specie", type=int, help="atomic number of the mobile specie (overrides config)")
    p.add_argument("--seed", type=int, help="dataset seed (overrides config)")
    p.add_argument("--chunk", default="0/1", help="process materials i, i+n, ... (e.g. $SLURM_ARRAY_TASK_ID/32)")
    p.add_argument("--dry-run", action="store_true", help="only count candidates per material")
    p.add_argument("--overwrite", action="store_true", help="recompute materials whose file exists")
    p.add_argument("--print-config", action="store_true", help="print default settings as YAML and exit")
    p.add_argument("--quiet", action="store_true", help="no progress bar")
    p.set_defaults(func=_dataset)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
