"""
Interactive 3D view of an EnergyMap in Jupyter or a web browser.

    emap.view()                         # inline in Jupyter
    emap.view(theme="light", supercell=(1, 2, 2), level=0.6)
    emap.view().save("map.html")        # standalone page (works offline)
    emap.view().open()                  # separate browser window

The page shows the structure (VESTA colors, bonds, coordination polyhedra) and the energy map as an
isosurface with 1D/2D/3D percolation-barrier presets; e.g. polyhedra=["Fe", "P"] or hide_bonds=["Li"]
leave out the mobile ion's polyhedra or bonds. 3Dmol.js (BSD-3-Clause) is embedded in every page.

The viewer (_page, _inputs, _scene, _vesta, static/) is a copy of chucha (MIT, same author):
structure_view() shows any structure or file without an energy map.
"""
import numpy as np

from ._page import View
from ._page import view as structure_view

__all__ = ["view", "structure_view", "payload", "View"]


def payload(emap, level=None):
    """structure_view arguments for an EnergyMap: structure, map (eV above min), presets."""
    values = np.asarray(emap.values, dtype=float)
    finite = np.isfinite(values)
    vmin = values[finite].min()
    values = np.where(finite, values - vmin, values[finite].max() - vmin)  # blocked points never inside a surface
    barriers = emap.percolation_barriers()
    presets = {f"{k[1]}D": float(v) for k, v in barriers.items() if np.isfinite(v)}
    if level is None:
        level = presets["1D"] + 0.01 if "1D" in presets else 0.5
    model = emap.info.get("model", "")
    out = {
        "structure": emap.atoms, "volume": values, "level": float(level), "presets": presets,
        "volume_label": "eV above min",
    }
    if model:
        out["title"] = f"{emap.atoms.get_chemical_formula()} · {model}"
    return out


def view(emap, *, level=None, **kwargs):
    """
    Interactive view of an EnergyMap.

    Parameters
    ----------
    level: float, optional
        isosurface level in eV above the minimum; default: 1D percolation barrier + 0.01 eV
    **kwargs
        any structure_view option, e.g. supercell=(1, 2, 2), theme="light", style="soft", along="c",
        rotate_about="c", polyhedra=["Fe", "P"], hide=["Li"], hide_bonds=["Li"], bonds=False, polyhedra_opacity=0.6, axes=False, height=600,
        save="map.html"

    Returns
    -------
    View: displayed inline in Jupyter; .save(path), .open()
    """
    args = payload(emap, level=level)
    args.update(kwargs)
    return structure_view(args.pop("structure"), **args)
