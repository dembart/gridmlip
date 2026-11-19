"""
The dataset recipe behind ``gridmlip dataset``:

    defects -> soap_features -> select_fps -> rattle -> one .xyz per material

It uses only public building blocks, so the README loop and the command give identical files.
"""
import copy
import csv
import numbers
from pathlib import Path

DEFAULTS = {
    "specie": None,
    "seed": 42,
    "defects": {
        "types": ["interstitial", "vacancy", "frenkel_pair"],
        "resolution": 0.25,
        "r_min": 1.5,
        "r_cut": 5.0,
        "symprec": 0.1,
        "max_sep_dist": "auto",
        "upper_bound": 6.0,
        "supercell_size": None,
    },
    "limits": {"max_atoms": 150, "max_candidates": 500_000},
    "soap": {"r_cut": 6.0, "n_max": 3, "l_max": 2, "sigma": 0.3, "periodic": True,
             "normalize": True, "dtype": "float64"},
    "selection": {"pca": 0.95, "max_samples": 500, "rel_tol": 1e-5, "start_idx": "centroid"},
    "rattle": {"copies": 1, "pos_std": 0.05, "strain_std": 0.02, "max_volume_change": 0.05,
               "max_principal_strain": 0.05, "max_abs_shear_strain": 0.05, "d_min": 1.0,
               "max_attempts": 100},
}

_SUMMARY_FIELDS = ["material_id", "n_atoms", "candidates", "selected", "status"]


def merge_settings(*overrides):
    """DEFAULTS updated group by group with the given dicts (unknown keys raise)."""
    settings = copy.deepcopy(DEFAULTS)
    for override in overrides:
        for key, value in (override or {}).items():
            if key not in settings:
                if key in ("versions", "created"):  # written by save_settings
                    continue
                raise KeyError(f"Unknown setting {key!r}; allowed: {sorted(DEFAULTS)}")
            if isinstance(settings[key], dict):
                if value is None:
                    settings[key] = None  # e.g. rattle: null -> no perturbation
                    continue
                unknown = set(value) - set(DEFAULTS[key])
                if unknown:
                    raise KeyError(f"Unknown {key} setting(s) {sorted(unknown)}; allowed: {sorted(DEFAULTS[key])}")
                settings[key].update(value)
            else:
                settings[key] = value
    return settings


def _generate(atoms, specie, d, dry_run=False):
    from ..defects import frenkel_pairs, interstitials, vacancies

    common = dict(resolution=d["resolution"], r_min=d["r_min"], r_cut=d["r_cut"],
                  symprec=d["symprec"], supercell_size=d["supercell_size"], dry_run=dry_run)
    hop = dict(max_sep_dist=d["max_sep_dist"], upper_bound=d["upper_bound"])
    funcs = {"interstitial": (interstitials, {}), "vacancy": (vacancies, hop),
             "frenkel_pair": (frenkel_pairs, hop)}
    out = 0 if dry_run else []
    for kind in d["types"]:
        if kind not in funcs:
            raise ValueError(f"Unknown defect type {kind!r}; allowed: {list(funcs)}")
        func, extra = funcs[kind]
        res = func(atoms, specie, **common, **extra)
        out += res["points"] if dry_run else res
    return out


def build_material(atoms, settings, dry_run=False):
    """
    Run the recipe for one structure.

    Returns
    -------
    (configurations, summary row)
    """
    from ..core import Grid
    from .features import soap_features
    from .rattle import rattle
    from .selection import select_fps

    s = settings
    mid = atoms.info["material_id"]
    d = s["defects"]
    n_atoms = len(Grid._repeat_atoms(atoms, d["r_cut"], supercell_size=d["supercell_size"])) + 1
    row = {"material_id": mid, "n_atoms": n_atoms, "candidates": "", "selected": "", "status": "ok"}
    if n_atoms > s["limits"]["max_atoms"]:
        row["status"] = f"skipped: {n_atoms} atoms > max_atoms"
        return [], row

    n_candidates = _generate(atoms, s["specie"], d, dry_run=True)
    row["candidates"] = n_candidates
    if dry_run:
        row["status"] = "dry run"
        return [], row
    if n_candidates > s["limits"]["max_candidates"]:
        row["status"] = f"skipped: {n_candidates} candidates > max_candidates"
        return [], row

    cfgs = _generate(atoms, s["specie"], d)
    for c in cfgs:
        c.info["material_id"] = mid

    if s["selection"] is None:
        idx = list(range(len(cfgs)))
    else:
        X = soap_features(cfgs, center=-1, **s["soap"])
        idx = select_fps(X, **s["selection"])

    if s["rattle"] is None:
        train = [cfgs[j] for j in idx]
    else:
        r = dict(s["rattle"])
        copies = int(r.pop("copies"))
        train = [rattle(cfgs[j], seed=s["seed"], copy=k, **r) for j in idx for k in range(copies)]
    row["selected"] = len(train)
    return train, row


_RANGES = {
    # (group, key): (check, message)
    ("defects", "resolution"): (lambda v: v > 0, "must be > 0"),
    ("defects", "r_min"): (lambda v: v >= 0, "must be >= 0"),
    ("defects", "r_cut"): (lambda v: v > 0, "must be > 0"),
    ("defects", "upper_bound"): (lambda v: v > 0, "must be > 0"),
    ("defects", "max_sep_dist"): (lambda v: v == "auto" or v > 0, "must be 'auto' or > 0"),
    ("limits", "max_atoms"): (lambda v: v >= 2, "must be >= 2"),
    ("limits", "max_candidates"): (lambda v: v >= 1, "must be >= 1"),
    ("soap", "r_cut"): (lambda v: v > 0, "must be > 0"),
    ("soap", "sigma"): (lambda v: v > 0, "must be > 0"),
    ("selection", "max_samples"): (lambda v: v >= 1, "must be >= 1"),
    ("selection", "rel_tol"): (lambda v: v >= 0, "must be >= 0"),
    ("selection", "pca"): (lambda v: v is None or (0 < v < 1) or (int(v) == v and v >= 1),
                           "must be null, a fraction in (0, 1) or a number of components >= 1"),
    ("rattle", "copies"): (lambda v: int(v) == v and v >= 1, "must be an integer >= 1"),
    ("rattle", "pos_std"): (lambda v: v >= 0, "must be >= 0"),
    ("rattle", "strain_std"): (lambda v: v >= 0, "must be >= 0"),
    ("rattle", "d_min"): (lambda v: v > 0, "must be > 0"),
}


def validate_settings(settings):
    """Raise ValueError for missing or out-of-range settings (all problems at once)."""
    from ase.data import chemical_symbols

    problems = []
    specie = settings.get("specie")
    if specie is None:
        problems.append("specie is required (atomic number of the mobile ion)")
    elif not (isinstance(specie, numbers.Integral) and 0 < specie < len(chemical_symbols)):
        problems.append(f"specie must be an atomic number, got {specie!r}")
    if not isinstance(settings.get("seed"), numbers.Integral):
        problems.append(f"seed must be an integer, got {settings.get('seed')!r}")
    d = settings["defects"]
    if d["r_cut"] <= d["r_min"]:
        problems.append(f"defects.r_cut ({d['r_cut']}) must be larger than defects.r_min ({d['r_min']})")
    if not d["types"]:
        problems.append("defects.types is empty")
    for (group, key), (ok, msg) in _RANGES.items():
        if settings.get(group) is None:  # selection: null / rattle: null switch the step off
            continue
        value = settings[group][key]
        try:
            good = ok(value)
        except TypeError:
            good = False
        if not good:
            problems.append(f"{group}.{key} {msg}, got {value!r}")
    if problems:
        raise ValueError("Invalid dataset settings:\n  - " + "\n  - ".join(problems))


def _check_structures(atoms_list, specie):
    """Material ids must be unique and usable as file names; every structure must contain the ion."""
    ids = [str(a.info.get("material_id", k)) for k, a in enumerate(atoms_list)]
    seen, dups = set(), set()
    for mid in ids:
        (dups if mid in seen else seen).add(mid)
    problems = []
    if dups:
        problems.append(f"duplicate material_id (they would share one output file): {sorted(dups)[:10]}")
    bad = [m for m in ids if not m or "/" in m or "\\" in m or m in (".", "..")]
    if bad:
        problems.append(f"material_id not usable as a file name: {bad[:10]}")
    missing = [mid for mid, a in zip(ids, atoms_list) if specie not in a.numbers]
    if missing:
        problems.append(f"{len(missing)} structure(s) contain no atom with Z={specie}: {missing[:10]}")
    if problems:
        raise ValueError("Invalid input structures:\n  - " + "\n  - ".join(problems))
    return ids


def _comparable(settings):
    """Settings as they look after a YAML round trip, without the run record."""
    import yaml
    from .settings import to_serializable

    record = {k: v for k, v in settings.items() if k not in ("versions", "created")}
    return yaml.safe_load(yaml.safe_dump(to_serializable(record)))


def _check_output(out, settings, overwrite):
    """Refuse to mix two different configurations in one output folder."""
    from .settings import load_settings

    path = out / "config.yaml"
    if not path.exists():
        return False
    previous = _comparable(load_settings(path))
    current = _comparable(settings)
    if previous == current:
        return True
    diff = sorted(k for k in set(previous) | set(current) if previous.get(k) != current.get(k))
    if not overwrite:
        raise ValueError(
            f"{out} already holds a dataset made with different settings (differs in: {diff}). "
            "Use a new output folder, or overwrite=True / --overwrite to recompute everything with the new settings."
        )
    return False


def run(atoms_list, output_dir, settings, dry_run=False, overwrite=False, chunk=(0, 1), progress=True):
    """
    Build the dataset for a list of structures (what ``gridmlip dataset`` runs).

    All settings and structures are checked before any work starts.

    Parameters
    ----------
    atoms_list: list of ase.Atoms
        structures; atoms.info['material_id'] names the output files (default: index in the list)
    output_dir: str or Path
    settings: dict
        as returned by merge_settings()
    dry_run: bool
        only count candidates, write only the summary
    overwrite: bool
        recompute materials whose file exists (default: skip them, so reruns resume)
    chunk: (i, n)
        process materials i, i+n, i+2n, ... (for job arrays); results do not depend on the split

    Returns
    -------
    list of summary rows (dicts)
    """
    from ase.io import write
    from tqdm import tqdm

    from .settings import save_settings

    validate_settings(settings)
    ids = _check_structures(atoms_list, settings["specie"])
    i, n = chunk
    if not (0 <= i < n):
        raise ValueError(f"chunk must be (i, n) with 0 <= i < n, got {chunk}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    same_settings = _check_output(out, settings, overwrite)
    if not dry_run and not same_settings:
        save_settings(out / "config.yaml", **settings)

    rows = []
    selected = [(k, a) for k, a in enumerate(atoms_list) if k % n == i]
    for k, atoms in tqdm(selected, desc="materials", disable=not progress):
        atoms = atoms.copy()
        mid = ids[k]
        atoms.info["material_id"] = atoms.info.get("material_id", k)
        path = out / f"dataset_{mid}.xyz"
        if path.exists() and not overwrite and not dry_run:
            rows.append({"material_id": mid, "n_atoms": "", "candidates": "", "selected": "",
                         "status": "exists"})
            continue
        try:
            train, row = build_material(atoms, settings, dry_run=dry_run)
        except Exception as exc:  # one bad structure must not stop a 10k-structure run
            train, row = [], {"material_id": mid, "n_atoms": "", "candidates": "", "selected": "",
                              "status": f"error: {type(exc).__name__}: {exc}"}
        if train:
            write(path, train)
        rows.append(row)

    name = "summary.csv" if n == 1 else f"summary_{i}of{n}.csv"
    with open(out / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def build_dataset(
    structures,
    config,
    output_dir="dataset",
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    chunk=(0, 1),
    progress: bool = True,
):
    """
    Build a training set: defects -> SOAP -> farthest point sampling -> rattle, one .xyz per material.
    This is exactly what ``gridmlip dataset`` runs.

    Parameters
    ----------
    structures: str, Path or list of ase.Atoms
        structures, or a file ASE can read; atoms.info['material_id'] names the outputs
        (must be unique; default: index in the list)
    config: str, Path or dict
        everything that defines the dataset (specie, seed, defect, SOAP, selection and rattle
        settings): a YAML file (see ``gridmlip dataset --print-config``), a config.yaml written by
        an earlier run, or a dict with the same groups. ``specie`` is required.
    output_dir: str or Path, 'dataset' by default
        one folder per dataset; it refuses settings different from the ones it was made with
    dry_run: bool
        only count candidates per material (writes summary.csv only)
    overwrite: bool
        recompute materials whose file exists; also allows new settings in an existing folder
    chunk: (i, n) or 'i/n'
        process materials i, i+n, ... (for job arrays); results do not depend on the split

    Returns
    -------
    list of summary rows: {'material_id', 'n_atoms', 'candidates', 'selected', 'status'}

    Example
    -------
    build_dataset("structures.xyz", config="li.yaml", output_dir="datasets/li", dry_run=True)
    """
    from .settings import load_settings

    if isinstance(config, dict) or config is None:
        base = config or {}
    else:
        base = load_settings(config)
    settings = merge_settings(base)

    if isinstance(chunk, str):
        try:
            chunk = tuple(int(x) for x in chunk.split("/"))
        except ValueError:
            raise ValueError(f"chunk must look like 'i/n', got {chunk!r}") from None
    if isinstance(structures, (str, Path)):
        from ase.io import read
        atoms_list = read(structures, index=":")
    else:
        atoms_list = list(structures)
    return run(atoms_list, output_dir, settings, dry_run=dry_run, overwrite=overwrite,
               chunk=tuple(chunk), progress=progress)
