"""gridmlip.view: scene data (VESTA polyhedra) and the standalone page."""
import json
import re
import subprocess
import sys

import numpy as np
import pytest
from ase.io import read

import gridmlip as gm
from gridmlip.view import View, view
from gridmlip.view._scene import polyhedra
from gridmlip.view._vesta import VESTA_POLYHEDRA

from structures import licoo2, lifepo4, nacl


@pytest.fixture(scope="module")
def emap():
    pytest.importorskip("ions")
    return gm.bvse(lifepo4(), "Li+", resolution=0.3, verbose=False)


def page_data(v):
    return json.loads(re.search(r"const D = (\{.*?\});\n", v.html, re.S).group(1).replace("<\\/", "</"))


def test_vesta_polyhedra_lifepo4():
    polys = polyhedra(lifepo4(), "Li")
    cn = {}
    for symbol, _, ligands, xyz in polys:
        cn.setdefault(symbol, set()).add(len(ligands))
        assert {l["elem"] for l in ligands} == {"O"}
        assert np.all(np.linalg.norm(xyz - lifepo4().positions[_], axis=1) <= VESTA_POLYHEDRA[symbol]["O"])
    assert cn == {"Fe": {6}, "P": {4}} and len(polys) == 8   # FeO6 and PO4, never LiO6


def test_polyhedra_exclude_mobile_ion_and_anions():
    centers = {s for s, *_ in polyhedra(licoo2(), "Li")}
    assert centers == {"Co"}
    assert not polyhedra(nacl(), "Na")   # Cl is never a center


def test_view_page_is_self_contained(emap, tmp_path):
    v = emap.view()
    assert isinstance(v, View)
    assert "<script src=" not in v.html            # 3Dmol.js embedded: works offline
    assert "3Dmol.js 2.5.5" in v.html
    assert v._repr_html_().startswith("<iframe srcdoc=")
    v.save(tmp_path / "m.html")
    assert (tmp_path / "m.html").read_text() == v.html


def test_view_options_reach_the_page(emap):
    d = page_data(emap.view())
    assert d["opts"]["supercell"] == [1, 1, 1] and d["opts"]["theme"] == "dark"
    assert d["opts"]["sites"] is False and d["opts"]["hide"] == []
    assert d["opts"]["level"] == pytest.approx(emap.percolation_barriers()["e1d"] + 0.01)

    d = page_data(view(emap, level=0.7, supercell=(1, 2, 2), theme="light", bonds=False, along="c"))
    assert d["opts"]["level"] == 0.7 and d["opts"]["supercell"] == [1, 2, 2]
    assert d["opts"]["theme"] == "light" and d["opts"]["bonds"] is False and d["opts"]["view"] == "c"


def test_view_scene_contents(emap):
    d = page_data(emap.view())
    assert d["mobile"] == "Li"
    assert sum(a["elem"] == "Li" for a in d["atoms"]) == 4          # mobile-ion sublattice is shown
    p = next(p for p in d["polyhedra"] if p["elem"] == "Fe")
    assert len(p["ligands"]) == 6 and p["volume"] > 10 and 0 < p["distortion"] < 0.1
    n = d["grid"]["n"]
    assert len(d["grid"]["values"].split()) == n[0] * n[1] * n[2]
    assert d["elements"]["O"]["color"] == "#ff0300"                  # VESTA color


def test_cdn_mode(emap):
    v = view(emap, offline=False)
    assert '<script src="https://cdn.jsdelivr.net/npm/3dmol@2.5.5' in v.html
    assert len(v.html) < len(emap.view().html) - 400_000


def test_view_does_not_need_pymatgen():
    code = ("import sys, numpy as np; from ase.build import bulk; import gridmlip as gm; "
            "from gridmlip.view import view; a = bulk('NaCl', 'rocksalt', a=5.64, cubic=True); "
            "g = gm.Grid(a, 'Na', resolution=0.5, r_min=1.0); "
            "g.points; e = g.load(-g._min_dists[g._min_dists > g.r_min]); view(e); "
            "print('pymatgen' in ' '.join(sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
