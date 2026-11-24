"""view(): collect frames and grids, wrap them in a self-contained HTML page (the page builds the scene)."""
import base64
import html as _html
import json
import time
import warnings
from pathlib import Path

import numpy as np

from ._inputs import load, volume_grid
from ._scene import encode_grid, frame_payload, is_periodic, tables

CDN = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"
_STATIC = Path(__file__).parent / "static"


class View:
    """Result of view(): shows inline in Jupyter; .html is the standalone page."""

    def __init__(self, html, height, seconds):
        self.html, self.height, self.seconds = html, height, seconds

    def _repr_html_(self):
        return (f'<iframe srcdoc="{_html.escape(self.html, quote=True)}" '
                f'allow="clipboard-write" style="width:100%;height:{self.height}px;border:1px solid #ddd;border-radius:4px"></iframe>')

    def save(self, filename):
        """Write the standalone page (works offline unless view(..., offline=False))."""
        Path(filename).write_text(self.html)
        return str(filename)

    def open(self):
        """Open in the default web browser (via a temporary file)."""
        import tempfile
        import webbrowser

        f = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False)
        f.write(self.html)
        f.close()
        webbrowser.open("file://" + f.name)
        return f.name


def _site_data(frames, site_data):
    """[{name: per-atom values}] per frame from atoms.arrays keys and/or explicit arrays."""
    out = [{} for _ in frames]
    if site_data is None:
        return out
    if isinstance(site_data, str):
        site_data = [site_data]
    if not isinstance(site_data, dict):
        site_data = {k: None for k in site_data}
    nf = len(frames)
    for name, values in site_data.items():
        if values is None:  # an atoms.arrays key, per frame
            for k, a in enumerate(frames):
                if name not in a.arrays:
                    raise KeyError(f"site_data: no array {name!r} in frame {k} (has {sorted(a.arrays)})")
            per = [a.arrays[name] for a in frames]
        else:
            v = np.asarray(values)
            n = len(frames[0])
            if v.shape[:1] == (n,) and not (nf > 1 and v.shape[:2] == (nf, n)):
                per = [v] * nf
            elif v.shape[:2] == (nf, len(frames[0])):
                per = list(v)
            else:
                raise ValueError(f"site_data[{name!r}]: shape {v.shape}, expected ({n}, ...) or ({nf}, {n}, ...)")
        for k, p in enumerate(per):
            p = np.asarray(p)
            out[k][str(name)] = np.round(p.astype(float), 4).tolist() if p.dtype.kind in "biuf" else [str(x) for x in p]
    return out


def _markers(markers):
    out = []
    for k, m in enumerate(markers or ()):
        if isinstance(m, dict):
            xyz, label = m["xyz"], m.get("label", "")
        elif len(m) == 2 and np.ndim(m[0]) == 1:
            xyz, label = m
        else:
            xyz, label = m, ""
        out.append({"xyz": [float(x) for x in xyz], "label": str(label) or f"marker {k}"})
    return out


def _volume(grid, cell, name, level, presets, label, limit, signed=None):
    shape, b64 = encode_grid(grid, limit)
    out = {"name": name, "n": shape, "b64": b64, "cell": np.round(np.asarray(cell), 6).tolist(), "label": str(label),
           "presets": {str(k): float(v) for k, v in (presets or {}).items()}}
    if signed is not None:
        out["signed"] = bool(signed)
    if level is not None:
        out["level"] = float(level)
    elif out["presets"]:
        out["level"] = next(iter(out["presets"].values()))
    return out


def _tab(structure=None, *, volume=None, level=None, presets=None, volume_label="", grid_limit=1_000_000,
         site_data=None, labels=None, markers=None, frames=None, max_frames=1000, title=None, symmetry_frames=200,
         channel="total", signed=None):
    """Payload of one tab: frames, volumetric grids, markers."""
    frame_list, grids = [], []
    source = Path(structure).name if isinstance(structure, (str, Path)) else ""
    if structure is not None:
        frame_list, file_grid = load(structure, frames, channel)
        if len(frame_list) > max_frames:
            step = -(-len(frame_list) // max_frames)
            warnings.warn(f"{len(frame_list)} frames: showing every {step}th (pass frames=... or max_frames=...)")
            frame_list = frame_list[::step]
        if file_grid is not None:
            grids.append((file_grid, source + (f" ({channel})" if channel != "total" else "")))
    vols = volume if isinstance(volume, (list, tuple)) else ([] if volume is None else [volume])
    for k, v in enumerate(vols):
        name = Path(v).name + (f" ({channel})" if channel != "total" else "") if isinstance(v, (str, Path)) else f"volume {k}"
        grids.append((volume_grid(v, channel), name))
    if grids and not frame_list:
        raise ValueError("volumetric data needs a structure")
    if grids and not is_periodic(frame_list[0]):
        raise ValueError("volumetric data needs a periodic cell")

    data = _site_data(frame_list, site_data)
    for d, a in zip(data, frame_list):
        if "oxi_states" in a.arrays and "oxi_states" not in d:
            d["oxi_states"] = np.round(a.arrays["oxi_states"].astype(float), 3).tolist()
    if labels is not None:
        labels = [str(x) for x in labels]
        if frame_list and len(labels) != len(frame_list[0]):
            raise ValueError(f"labels: {len(labels)} given for {len(frame_list[0])} atoms")

    many = len(frame_list) > 1
    fpay = []
    for k, (a, d) in enumerate(zip(frame_list, data)):
        res = getattr(a.calc, "results", None) or {}  # e.g. an OUTCAR step: forces on hover, energy in the frame name
        if "forces" in res and "force" not in d:
            d["force"] = np.round(np.asarray(res["forces"], float), 4).tolist()
        e = res.get("free_energy", res.get("energy"))
        name = (f"{source or 'frame'} #{k}" if many else source) + (f" · E = {float(e):.4f} eV" if e is not None else "")
        f = frame_payload(a, d, labels, name, symmetry=k < symmetry_frames)
        if k and f["symbols"] == fpay[0]["symbols"]:
            f["symbols"] = None  # same atoms as frame 0 (saves space for trajectories)
        fpay.append(f)

    levels = level if isinstance(level, (list, tuple)) else [level] + [None] * (len(grids) - 1)
    cell0 = np.asarray(frame_list[0].cell) if frame_list else None
    volumes = [_volume(g, cell0, name, levels[k] if k < len(levels) else None,
                       presets if k == 0 else None, volume_label if k == 0 else "", grid_limit, signed)
               for k, (g, name) in enumerate(grids)]
    real = [a for a in frame_list if len(a)]
    return {"title": title or (real[0].get_chemical_formula() if real else source),
            "frames": fpay, "volumes": volumes, "markers": _markers(markers)}


def _bond_keys(hide_bonds):
    """("Li", "O"), "Li-O" -> "Li|O" (the page's bond-type key, symbols sorted); "Li" -> "Li" (all its bonds)."""
    if isinstance(hide_bonds, str):
        hide_bonds = [hide_bonds]
    keys = []
    for b in hide_bonds or ():
        pair = b.split("-") if isinstance(b, str) else list(b)
        if not 1 <= len(pair) <= 2 or not all(isinstance(e, str) and e.strip() for e in pair):
            raise ValueError(f"hide_bonds: expected an element, 'A-B' or ('A', 'B'), got {b!r}")
        keys.append("|".join(sorted(e.strip() for e in pair)))
    return keys


def page_options(supercell=(1, 1, 1), polyhedra=True, bonds=True, atoms=True, axes=True, isosurface=True,
                 show_markers=True, hide=(), hide_bonds=(), polyhedra_opacity=0.9, along=None, rotate_about=None, theme="dark",
                 style="classic", grid_limit=1_000_000):
    """Settings the page starts with (each can be changed in the page)."""
    return {
        "supercell": [int(x) for x in supercell], "polyhedra": bool(polyhedra),
        "centers": None if isinstance(polyhedra, bool) else [str(e) for e in polyhedra],
        "bonds": bool(bonds), "atoms": bool(atoms), "axes": bool(axes), "isosurface": bool(isosurface),
        "markers": bool(show_markers), "hide": [str(e) for e in hide],
        "hide_bonds": _bond_keys(hide_bonds), "polyhedra_opacity": float(polyhedra_opacity),
        "view": along, "axis": rotate_about, "theme": theme, "style": style, "grid_limit": int(grid_limit or 0),
    }


def render(payload, offline=True, head=""):
    """The self-contained page for a payload {"tabs": [...], "opts": {...}}; `head`: extra <head> markup (the app)."""
    if offline:
        lib = ("<script>/* 3Dmol.js 2.5.5, BSD-3-Clause (see 3Dmol-LICENSE.txt), https://3dmol.org */\n"
               + (_STATIC / "3Dmol-min.js").read_text() + "\n</script>")
    else:
        lib = f'<script src="{CDN}"></script>'
    core = "<script>\n" + (_STATIC / "chucha-core.js").read_text() + "\n</script>"
    page = (_STATIC / "page.html").read_text()
    js = lambda x: json.dumps(x, separators=(",", ":")).replace("</", "<\\/")  # safe inside <script>
    title = payload["tabs"][0]["title"] if payload.get("tabs") else ""
    png = lambda f: "data:image/png;base64," + base64.b64encode((_STATIC / f).read_bytes()).decode()
    favicon, logo = png("favicon-64.png"), png("icon-192.png")  # logo: the empty page, sharp on retina screens
    return (page.replace("__TITLE__", _html.escape(title or "chucha")).replace("__HEAD__", head)
            .replace("__FAVICON__", favicon).replace("__LOGO__", logo)
            .replace("__3DMOL__", lib).replace("__CORE__", core)
            .replace("__TABLES__", js(tables())).replace("__DATA__", js(payload)))


def view(structure=None, *, volume=None, level=None, presets=None, volume_label="", grid_limit=1_000_000,
         channel="total", signed=None,
         site_data=None, labels=None, markers=None, frames=None, max_frames=1000,
         supercell=(1, 1, 1), polyhedra=True, bonds=True, atoms=True, axes=True, isosurface=True, show_markers=True,
         hide=(), hide_bonds=(), polyhedra_opacity=0.9, along=None, rotate_about=None, theme="dark", style="classic",
         title=None, offline=True, height=600, save=None):
    """
    Interactive 3D view of crystal structures (and trajectories) in Jupyter or a web browser.

    Files can also be dropped onto the page (CIF, POSCAR/CONTCAR, XDATCAR, OUTCAR, .traj, xyz/extxyz, .cube, .grd, CHGCAR/LOCPOT):
    the page asks whether to replace what is shown, add the file (a structure as another frame, volumetric
    data as another isosurface) or open it in a new tab.

    Parameters
    ----------
    structure: ase.Atoms, pymatgen Structure, list of frames, a file path, a dict, or None
        anything ase.io.read reads (CIF, POSCAR, XDATCAR, extxyz, .traj, ...); a .cube, .grd or
        CHGCAR/LOCPOT/ELFCAR file also gives volumetric data (.grd has no atoms: only the cell).
        A dict opens one tab per entry: {"LFP": atoms, "map": {"structure": atoms, "volume": grid, ...}}
        (a dict value may carry its own structure, volume, level, presets, volume_label, site_data,
        labels, markers and frames). None: an empty page to drop files on
    volume: 3D array, file path, or a list of them, optional
        values on a periodic grid over the unit cell (point [i, j, k] at fractional (i/n0, j/n1, k/n2)),
        shown as isosurfaces
    level: float or list, optional
        starting isosurface level(s) (default: the first preset, else the mean of the data)
    presets: dict, optional
        named levels shown as buttons for the first volume, e.g. {"1D": 0.32, "2D": 0.51, "3D": 0.77}
    volume_label: str
        meaning/unit of the values of the first volume, e.g. "eV above min"
    channel: str, "total" by default
        which data of a VASP CHGCAR-type file: total, magnetization (spin up − down), up, down,
        or mx, my, mz for noncollinear runs (the page can switch channels of dropped files)
    signed: bool, optional
        draw the isosurface at +level and −level (two colors); by default on for data with both
        signs in comparable amounts (magnetization, density differences)
    grid_limit: int, 1e6 by default
        grids with more points are resampled (periodic linear interpolation) to keep the page small
    site_data: str, list of str or dict, optional
        per-atom values shown on hover and click: atoms.arrays keys and/or {name: array of shape
        (n_atoms, ...) or (n_frames, n_atoms, ...)}; 'oxi_states' is shown when present
    labels: list of str, optional
        per-atom names replacing Fe0, O12, ...
    markers: list, optional
        extra points (Cartesian, unit cell): xyz, (xyz, label) or {"xyz":..., "label":...},
        e.g. interstitial sites or energy minima; repeated in every supercell copy
    frames: int, slice or list of ints, optional
        frames of a trajectory to show; more than max_frames frames are strided down (with a warning)
    supercell: (nx, ny, nz), 1 to 4 per axis (can be changed in the page)
    polyhedra: bool or list of element symbols
        VESTA default polyhedra; a list restricts the centers, e.g. ["Fe", "P"]
    bonds, atoms, axes, isosurface, show_markers: bool
        layers shown at the start (each can be toggled in the page)
    hide: element symbols not drawn at the start (Layers > Atoms > in the page)
    hide_bonds: bond types not drawn at the start (Layers > Bonds > in the page): pairs ("Li", "O") or "Li-O",
        or an element ("Li") for all of its bonds
    polyhedra_opacity: float, 0.9 by default
    along: 'a', 'b' or 'c', optional — start looking down this axis
    rotate_about: 'a', 'b', 'c' or None — lock mouse rotation to a crystal axis
    theme: 'dark' or 'light'
    style: 'classic' (VESTA-like), 'soft' (ambient occlusion, diffuse light), 'glossy' (reflections) or 'outline'
    title: str, optional (default: chemical formula)
    offline: bool, True by default — embed 3Dmol.js (~0.5 MB); False loads it from a CDN
    height: int, height of the inline view in Jupyter (px)
    save: str, optional — also write the standalone page to this .html file

    The space group is shown when spglib is installed (optional); point groups are found in the page.

    Returns
    -------
    View: displayed inline in Jupyter; View.save(path), View.open()
    """
    t0 = time.perf_counter()
    common = dict(volume=volume, level=level, presets=presets, volume_label=volume_label, grid_limit=grid_limit,
                  channel=channel, signed=signed,
                  site_data=site_data, labels=labels, markers=markers, frames=frames, max_frames=max_frames)
    if isinstance(structure, dict):
        tabs = []
        for name, spec in structure.items():
            kw = {"grid_limit": grid_limit, "max_frames": max_frames, "frames": frames}
            if isinstance(spec, dict):
                kw.update(spec)
            else:
                kw["structure"] = spec
            tabs.append(_tab(**kw, title=str(name)))
    else:
        tabs = [_tab(structure, **common, title=title)]

    payload = {"tabs": tabs, "opts": page_options(
        supercell=supercell, polyhedra=polyhedra, bonds=bonds, atoms=atoms, axes=axes, isosurface=isosurface,
        show_markers=show_markers, hide=hide, hide_bonds=hide_bonds, polyhedra_opacity=polyhedra_opacity, along=along, rotate_about=rotate_about,
        theme=theme, style=style, grid_limit=grid_limit)}
    out = render(payload, offline=offline)
    result = View(out, height, time.perf_counter() - t0)
    if save:
        result.save(save)
    return result
