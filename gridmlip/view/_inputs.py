"""Turn whatever the user passes into a list of ase.Atoms frames and an optional volumetric grid."""
from pathlib import Path

import numpy as np
from ase import Atoms

_VASP_VOLUMETRIC = ("CHGCAR", "CHG", "LOCPOT", "PARCHG", "AECCAR", "ELFCAR")


def _is_vasp_volumetric(path):
    name = Path(path).name.upper()
    return any(name.startswith(k) for k in _VASP_VOLUMETRIC) or name.endswith((".CHGCAR", ".LOCPOT"))


def _from_pymatgen(obj):
    """pymatgen Structure -> Atoms (oxidation states kept as the 'oxi_states' array)."""
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(obj)
    oxi = [getattr(getattr(site, "specie", None), "oxi_state", None) for site in obj]
    if any(o for o in oxi):
        atoms.set_array("oxi_states", np.array([o or 0.0 for o in oxi], dtype=float))
    for k in ("oxi_states", "magmom"):  # AseAtomsAdaptor may also store these as site properties
        if k in atoms.arrays and atoms.arrays[k].dtype == object:
            del atoms.arrays[k]
    return atoms


def _as_atoms(obj):
    if isinstance(obj, Atoms):
        return obj
    if type(obj).__module__.startswith("pymatgen"):
        return _from_pymatgen(obj)
    raise TypeError(f"cannot view {type(obj).__name__}: expected ase.Atoms, a pymatgen Structure, "
                    "a file path or a list of frames")


def _select(frames, index):
    if index is None:
        return frames
    if isinstance(index, (int, np.integer)):
        return [frames[index]]
    if isinstance(index, slice):
        return frames[index]
    return [frames[i] for i in index]


def read_grd(path):
    """VESTA .grd (bvlain, gridmlip): comment, cell parameters, grid size, values in C order; no atoms."""
    from ase.geometry import cellpar_to_cell

    lines = [l for l in Path(path).read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    par = [float(x) for x in lines[0].split()[:6]]
    n = [int(x) for x in lines[1].split()[:3]]
    values = np.array(" ".join(lines[2:]).split(), dtype=float)
    if values.size < n[0] * n[1] * n[2]:
        raise ValueError(f"{path}: {values.size} of {n[0] * n[1] * n[2]} grid values")
    return Atoms(cell=cellpar_to_cell(par), pbc=True), values[: n[0] * n[1] * n[2]].reshape(n)


CHANNELS = ("total", "magnetization", "up", "down", "mx", "my", "mz")


def read_chgcar(path):
    """
    VASP CHGCAR / LOCPOT / ELFCAR / PARCHG / AECCAR: the structure and every grid block, as
    {"total"} | {"total", "magnetization"} (spin-polarized) | {"total", "mx", "my", "mz"} (noncollinear).
    CHG-type files are divided by the cell volume (e/Å³, like ASE); other files keep their values.
    """
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    cell = cell * (np.cbrt(-scale / abs(np.linalg.det(cell))) if scale < 0 else scale)
    k = 5
    if all(t.isdigit() for t in lines[5].split()):  # VASP 4: no species line
        species, counts = lines[0].split(), [int(t) for t in lines[5].split()]
        k = 6
    else:
        species, counts = lines[5].split(), [int(t) for t in lines[6].split()]
        k = 7
    species = [s.split("_")[0].split("/")[0] for s in species[:len(counts)]]
    if lines[k].strip()[:1].lower() == "s":  # selective dynamics
        k += 1
    cart = lines[k].strip()[:1].lower() in "ck"
    k += 1
    natoms = sum(counts)
    pos = np.array([[float(x) for x in lines[k + i].split()[:3]] for i in range(natoms)])
    pos = pos * scale if cart else pos @ cell
    symbols = [s for s, c in zip(species, counts) for _ in range(c)]
    k += natoms
    while not lines[k].strip():
        k += 1
    head = lines[k].split()
    n = [int(t) for t in head[:3]]
    total = n[0] * n[1] * n[2]
    div = abs(np.linalg.det(cell)) if Path(path).name.upper().startswith(("CHG", "PARCHG", "AECCAR")) else 1.0
    blocks = []
    while k < len(lines) and len(blocks) < 4:
        if lines[k].split() != head:  # augmentation occupancies and moments sit between the blocks
            k += 1
            continue
        k += 1
        vals = []
        while k < len(lines) and len(vals) < total:
            vals.extend(lines[k].split())
            k += 1
        if len(vals) < total:
            if not blocks:
                raise ValueError(f"{path}: {len(vals)} of {total} grid values")
            break
        blocks.append(np.array(vals[:total], float).reshape(n[2], n[1], n[0]).transpose(2, 1, 0) / div)  # x fastest
    names = ["total", "mx", "my", "mz"] if len(blocks) >= 4 else ["total", "magnetization"] if len(blocks) == 2 else ["total"]
    return Atoms(symbols, positions=pos, cell=cell, pbc=True), dict(zip(names, blocks))


def channel_of(channels, channel):
    """One channel of a volumetric file; spin up / down from total and magnetization."""
    if channel in channels:
        return channels[channel]
    if channel in ("up", "down") and "magnetization" in channels:
        return 0.5 * (channels["total"] + (1 if channel == "up" else -1) * channels["magnetization"])
    raise ValueError(f"channel {channel!r}: this file has {sorted(channels)}"
                     + (" (and up, down)" if "magnetization" in channels else ""))


def read_volume(path, channel="total"):
    """(frames, grid) from a volumetric file: VESTA .grd, Gaussian .cube or VASP CHGCAR/LOCPOT/ELFCAR...;
    channel: total, magnetization, up, down (spin-polarized) or mx, my, mz (noncollinear) for VASP files"""
    path = str(path)
    if path.lower().endswith(".grd"):
        atoms, grid = read_grd(path)
        return [atoms], grid
    if path.lower().endswith(".cube"):
        from ase.io.cube import read_cube_data

        data, atoms = read_cube_data(path)
        return [atoms], np.asarray(data, dtype=float)
    atoms, channels = read_chgcar(path)
    return [atoms], channel_of(channels, channel)


def load(obj, index=None, channel="total"):
    """
    Frames and (if the input carries one) a volumetric grid.

    obj: ase.Atoms | pymatgen Structure | path | list/tuple/iterable of Atoms or Structures
    index: int, slice or list of ints selecting frames of a trajectory
    """
    grid = None
    if isinstance(obj, (str, Path)):
        path = Path(obj)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in (".cube", ".grd") or _is_vasp_volumetric(path):
            frames, grid = read_volume(path, channel)
        else:
            from ase.io import read

            idx = index if isinstance(index, (int, slice)) else ":"
            frames = read(str(path), index=idx)
            frames = frames if isinstance(frames, list) else [frames]
            if not isinstance(index, (int, slice)):
                frames = _select(frames, index)
            index = None
    elif isinstance(obj, Atoms) or type(obj).__module__.startswith("pymatgen"):
        frames = [_as_atoms(obj)]
    elif hasattr(obj, "__iter__"):
        frames = [_as_atoms(a) for a in obj]
    else:
        frames = [_as_atoms(obj)]
    frames = _select(list(frames), index)
    if not frames:
        raise ValueError("no frames to view")
    return frames, grid


def volume_grid(volume, channel="total"):
    """ndarray (3D) from an array or a volumetric file path."""
    if volume is None:
        return None
    if isinstance(volume, (str, Path)):
        return read_volume(volume, channel)[1]
    grid = np.asarray(volume, dtype=float)
    if grid.ndim != 3:
        raise ValueError(f"volume must be a 3D grid over the unit cell, got shape {grid.shape}")
    return grid
