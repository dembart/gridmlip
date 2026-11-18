"""
Generate reference results with a previous gridmlip version.

Usage
-----
python tests/make_reference.py /path/to/dir/containing/old/gridmlip [output.json]

The old version is run WITHOUT symmetry (symprec=None), because its symmetry
reduction is wrong for some space groups (1/3, 1/6 translations; pseudo-symmetric cells).
These references are therefore the old code's correct answers.
"""
import json
import sys
import warnings
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore")

old_root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "data" / "reference_v0_2_7.json"

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(old_root))

import gridmlip  # noqa: E402
import gridmlip.integrations.bvse as bvse_mod  # noqa: E402
from gridmlip import Grid  # noqa: E402
from structures import BVSE_CASES, SYMMETRY_CASES  # noqa: E402

assert Path(gridmlip.__file__).resolve().is_relative_to(old_root), gridmlip.__file__

PARAMS = dict(resolution=0.2, r_cut=10.0, r_min=0.8, k=100, f=0.74)

bvse_mod.Grid = partial(Grid, symprec=None)
ref = {"params": PARAMS, "source": str(old_root), "bvse": {}, "radii": {}}

for name, (build, ion) in BVSE_CASES.items():
    grid, _ = bvse_mod.bvse_distribution(build(), ion, empty_framework=True, **PARAMS)
    ref["bvse"][name] = {
        "barriers": grid.percolation_barriers(backend="numba"),
        "mesh_shape": list(grid.mesh_shape),
    }
    print(name, ref["bvse"][name])

for name, build in SYMMETRY_CASES.items():
    atoms = build()
    g = Grid(atoms, specie=3, resolution=0.2, r_cut=6.0, r_min=0.5, symprec=None,
             empty_framework=True, verbose=False)
    g.generate_mesh()
    ref["radii"][name] = g.percolation_radii(backend="numba")
    print(name, ref["radii"][name])

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(ref, indent=2))
print("written", out)
