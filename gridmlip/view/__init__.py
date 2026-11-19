"""
Interactive 3D view of an EnergyMap in Jupyter or a web browser.

    emap.view()                         # inline in Jupyter
    emap.view(theme="light", supercell=(1, 2, 2), level=0.6)
    emap.view().save("map.html")        # standalone page (works offline)
    emap.view().open()                  # separate browser window

The page shows the unit cell, atoms (VESTA colors), bonds and coordination polyhedra following VESTA's
default rules and bond lengths, and an isosurface of the energy map (default: 1D percolation barrier).
3Dmol.js (BSD-3-Clause) is bundled and embedded in every page; only numpy, ase and scipy are needed.
"""
import html as _html
import json
import time
from pathlib import Path

from ase.data import chemical_symbols

from ._scene import payload

__all__ = ["view", "View"]

CDN = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"
_HERE = Path(__file__).parent
_JS = _HERE / "3Dmol-min.js"
_PAGE = _HERE / "page.html"


class View:
    """Result of view(): shows inline in Jupyter; .html is the standalone page."""

    def __init__(self, html, height, seconds):
        self.html, self.height, self.seconds = html, height, seconds

    def _repr_html_(self):
        return (f'<iframe srcdoc="{_html.escape(self.html, quote=True)}" '
                f'style="width:100%;height:{self.height}px;border:1px solid #ddd;border-radius:4px"></iframe>')

    def save(self, filename):
        with open(filename, "w") as f:
            f.write(self.html)
        return filename

    def open(self):
        """Open in the default web browser (via a temporary file)."""
        import tempfile, webbrowser
        f = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False)
        f.write(self.html); f.close()
        webbrowser.open("file://" + f.name)
        return f.name


def view(emap, *, level=None, supercell=(1, 1, 1), isosurface=True, polyhedra=True, bonds=True,
         sites=False, atoms=True, mobile_ions=True, polyhedra_opacity=0.75, along=None, rotate_about=None, theme="dark", offline=True,
         mobile=None, height=600, save=None):
    """
    Interactive view of an EnergyMap: structure, VESTA-style polyhedra and an energy isosurface.

    Parameters
    ----------
    level: float, optional
        isosurface level in eV above the minimum; default: 1D percolation barrier + 0.01 eV
    supercell: (nx, ny, nz), (1, 1, 1) by default
        starting supercell (1 to 4 per axis; can be changed in the page)
    isosurface, polyhedra, bonds, sites, atoms, mobile_ions: bool
        what is shown at the start (each can be toggled in the page); sites are the local energy
        minima (off by default), mobile_ions the mobile-ion sublattice of emap.atoms
    polyhedra_opacity: float, 0.75 by default
    along: 'a', 'b' or 'c', optional
        start looking down this axis
    rotate_about: 'a', 'b', 'c' or None
        lock mouse rotation to a crystal axis (None: free)
    theme: 'dark' or 'light', 'dark' by default (can be switched in the page)
    offline: bool, True by default
        embed 3Dmol.js (~0.5 MB) in the page; False loads it from a CDN (needs internet)
    mobile: str, optional
        mobile ion symbol (default: emap.specie); never a polyhedron center
    height: int, 600 by default
        height of the inline view in Jupyter (px)
    save: str, optional
        also write the standalone page to this .html file

    Returns
    -------
    View: displayed inline in Jupyter; View.save(path), View.open() for a browser window
    """
    t0 = time.perf_counter()
    if mobile is None and emap.specie is not None:
        mobile = chemical_symbols[emap.specie]
    data = payload(emap, mobile)
    b = data["barriers"]
    data["opts"] = {
        "level": float(level) if level is not None else ((b["e1d"] + 0.01) if b["e1d"] is not None else 0.5),
        "supercell": [int(x) for x in supercell], "isosurface": bool(isosurface), "polyhedra": bool(polyhedra),
        "bonds": bool(bonds), "sites": bool(sites), "atoms": bool(atoms), "mobile_ions": bool(mobile_ions),
        "polyhedra_opacity": float(polyhedra_opacity), "view": along, "axis": rotate_about, "theme": theme,
    }
    if offline and _JS.exists():
        lib = "<script>/* 3Dmol.js 2.5.5, BSD-3-Clause (see 3Dmol-LICENSE.txt), https://3dmol.org */\n" \
              + _JS.read_text() + "\n</script>"
    else:
        lib = f'<script src="{CDN}"></script>'
    page = _PAGE.read_text()
    data_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")  # safe inside <script>
    html = (page.replace("__TITLE__", _html.escape(data["title"])).replace("__3DMOL__", lib)
            .replace("__DATA__", data_json))
    result = View(html, height, time.perf_counter() - t0)
    if save:
        result.save(save)
    return result
