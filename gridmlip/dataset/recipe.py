"""
The dataset recipe behind ``gridmlip dataset``:

    defects -> soap_features -> select_fps -> rattle -> one .xyz per material

It uses only public building blocks, so the README loop and the command give identical files.
"""
import copy
import csv
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


def run(atoms_list, output_dir, settings, dry_run=False, overwrite=False, chunk=(0, 1), progress=True):
    """
    Build the dataset for a list of structures (what ``gridmlip dataset`` runs).

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

    if settings["specie"] is None:
        raise ValueError("specie is required (atomic number of the mobile specie)")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    i, n = chunk
    if i == 0 and not dry_run:
        save_settings(out / "config.yaml", **settings)

    rows = []
    selected = [(k, a) for k, a in enumerate(atoms_list) if k % n == i]
    for k, atoms in tqdm(selected, desc="materials", disable=not progress):
        atoms = atoms.copy()
        atoms.info.setdefault("material_id", k)
        mid = atoms.info["material_id"]
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
