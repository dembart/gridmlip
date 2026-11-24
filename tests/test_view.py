"""gridmlip.view: the bundled viewer and an EnergyMap in it."""
import base64
import json
import re

import numpy as np
import pytest

import gridmlip as gm
from gridmlip.view import View, payload, structure_view, view

from structures import lifepo4


@pytest.fixture(scope="module")
def emap():
    pytest.importorskip("ions")
    return gm.bvse(lifepo4(), "Li+", resolution=0.3, verbose=False)


def page(v):
    d = json.loads(re.search(r"/\*chucha-data\*/(\{.*?\})/\*end-data\*/", v.html, re.S).group(1).replace("<\\/", "</"))
    return {**d["tabs"][0], "opts": d["opts"]}


def test_payload(emap):
    p = payload(emap)
    b = emap.percolation_barriers()
    assert p["presets"] == {"1D": pytest.approx(b["e1d"]), "2D": pytest.approx(b["e2d"]), "3D": pytest.approx(b["e3d"])}
    assert p["level"] == pytest.approx(b["e1d"] + 0.01)
    assert p["volume"].shape == emap.values.shape and p["volume"].min() == 0 and np.isfinite(p["volume"]).all()
    assert "polyhedra" not in p and "hide_bonds" not in p        # drawn as chucha draws any structure


def test_view_page(emap, tmp_path):
    v = emap.view(save=tmp_path / "m.html")
    assert isinstance(v, View) and (tmp_path / "m.html").read_text() == v.html
    d = page(v)
    assert len(d["frames"][0]["xyz"]) == len(emap.atoms)
    (vol,) = d["volumes"]
    assert vol["n"] == list(emap.values.shape) and vol["label"] == "eV above min"
    assert vol["level"] == pytest.approx(emap.percolation_barriers()["e1d"] + 0.01)
    grid = np.frombuffer(base64.b64decode(vol["b64"]), "<f4").reshape(vol["n"])
    assert grid.min() == pytest.approx(0, abs=1e-6)
    assert d["opts"]["centers"] is None and d["opts"]["polyhedra"]
    assert d["opts"]["hide_bonds"] == []                         # all bonds drawn; hide_bonds=["Li"] to hide
    assert d["markers"] == []


def test_view_options_pass_through(emap):
    d = page(view(emap, level=0.7, supercell=(1, 2, 2), theme="light", bonds=False, along="c", hide=["Li"]))
    o = d["opts"]
    assert d["volumes"][0]["level"] == 0.7
    assert o["supercell"] == [1, 2, 2] and o["theme"] == "light" and not o["bonds"] and o["view"] == "c"
    assert o["hide"] == ["Li"]
    o = page(view(emap, hide_bonds=["Li"], polyhedra=["Fe", "P"]))["opts"]
    assert o["hide_bonds"] == ["Li"] and o["centers"] == ["Fe", "P"]


def test_page_is_self_contained_and_offline(emap):
    html = emap.view().html
    assert "3Dmol.js 2.5.5" in html and "cdn.jsdelivr" not in html
    assert "cdn.jsdelivr" in emap.view(offline=False).html


def test_structure_view_without_a_map():
    d = page(structure_view(lifepo4(), supercell=(2, 1, 1)))
    assert len(d["frames"][0]["xyz"]) == 28 and d["volumes"] == [] and d["opts"]["supercell"] == [2, 1, 1]
