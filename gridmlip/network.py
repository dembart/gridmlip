from collections import deque

import numpy as np
import spglib
from scipy.spatial import QhullError, Voronoi

from .grid import _as_atomic_number

__author__ = "Artem Dembitskiy"


def _polygon_area(vertices, normal):
    """Area of a planar convex polygon given its vertices in any order."""
    c = vertices.mean(axis=0)
    normal = normal / np.linalg.norm(normal)
    u = vertices[0] - c
    u -= normal * (u @ normal)
    if np.linalg.norm(u) < 1e-12:
        return 0.0
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    d = vertices - c
    order = np.argsort(np.arctan2(d @ v, d @ u))
    x, y = (d @ u)[order], (d @ v)[order]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _voronoi_faces(points, cell, site):
    """
    Faces of the periodic Voronoi cell of points[site]: list of (target, offset, area, length), all faces.
    The images taken into account grow until the cell is provably complete (every vertex within half the radius).
    """
    cell = np.asarray(cell, dtype=float)
    frac = np.linalg.solve(cell.T, points.T).T
    heights = abs(np.linalg.det(cell)) / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
    center = points[site]
    radius = 2.0 * max(heights.min(), 1.0)
    while True:
        n = np.ceil(radius / heights).astype(int) + 1
        shifts = np.array(np.meshgrid(*[np.arange(-k, k + 1) for k in n], indexing="ij")).reshape(3, -1).T
        idx = np.repeat(np.arange(len(points)), len(shifts))
        off = np.tile(shifts, (len(points), 1))
        pos = (frac[idx] + off) @ cell
        near = np.linalg.norm(pos - center, axis=1) <= radius
        idx, off, pos = idx[near], off[near], pos[near]
        me = int(np.flatnonzero((idx == site) & ~off.any(axis=1))[0])
        try:
            vor = Voronoi(pos)
        except QhullError:
            radius *= 1.5
            continue
        region = vor.regions[vor.point_region[me]]
        if region and -1 not in region:
            reach = np.linalg.norm(vor.vertices[region] - center, axis=1).max()
            if 2.0 * reach < radius:
                break
        radius *= 1.5
    faces = []
    for (a, b), verts in zip(vor.ridge_points, vor.ridge_vertices):
        if me not in (a, b):
            continue
        other = b if a == me else a
        d = pos[other] - center
        area = _polygon_area(vor.vertices[verts], d)
        faces.append((int(idx[other]), tuple(int(x) for x in off[other]), area, float(np.linalg.norm(d))))
    return faces


def voronoi_edges(atoms, specie, min_face: float = 0.02, max_length: float | None = None, dim: int = 3):
    """
    Hops between Voronoi neighbours of the mobile sublattice (the framework is ignored).
    A hop [source, target, ox, oy, oz] goes from site ``source`` of the unit cell to site ``target``
    translated by the lattice vector (ox, oy, oz), as in ions' Percolator.

    Parameters
    ----------
    atoms: ase.Atoms
        unit cell

    specie: int or str
        mobile specie

    min_face: float, 0.02 by default
        drop hops through faces smaller than this fraction of the site's Voronoi cell surface
        (tiny faces join distant sites that are separated by others)

    max_length: float, optional
        drop hops longer than this (Angstrom)

    dim: int, 3 by default
        the network must percolate in at least this many dimensions (see ``periodicity``),
        otherwise ValueError

    Returns
    -------
    (N, 5) int array of all hops [source, target, ox, oy, oz], each with its reverse
    """
    z = _as_atomic_number(specie)
    sites = np.flatnonzero(atoms.numbers == z)
    if len(sites) == 0:
        raise ValueError(f"No atoms of specie {specie} in the structure")
    points = atoms.positions[sites]
    kept = set()
    for k in range(len(sites)):
        faces = _voronoi_faces(points, atoms.cell, k)
        surface = sum(f[2] for f in faces)
        for t, off, area, length in faces:
            if area < min_face * surface or (max_length is not None and length > max_length):
                continue
            s_, t_ = int(sites[k]), int(sites[t])
            kept.add((s_, t_) + off)
            kept.add((t_, s_) + tuple(-x for x in off))
    edges = np.array(sorted(kept), dtype=int).reshape(-1, 5)
    d = periodicity(edges)
    if d < dim:
        raise ValueError(f"The Voronoi hop network percolates in {d}D, not {dim}D: lower min_face, "
                         f"raise max_length or pass dim={d}")
    return edges


def _component_dims(edges):
    """For every edge, the number of independent lattice translations of its connected component."""
    edges = np.asarray(edges, dtype=int).reshape(-1, 5)
    adjacency = {}
    for k, (s, t, *o) in enumerate(edges):
        o = np.array(o)
        adjacency.setdefault(s, []).append((t, o, k))
        adjacency.setdefault(t, []).append((s, -o, k))
    position, dims = {}, np.zeros(len(edges), dtype=int)
    for start in adjacency:
        if start in position:
            continue
        position[start] = np.zeros(3, dtype=int)
        queue, translations, members = deque([start]), [], []
        while queue:
            i = queue.popleft()
            for j, o, k in adjacency[i]:
                members.append(k)
                pj = position[i] + o
                if j not in position:
                    position[j] = pj
                    queue.append(j)
                elif np.any(pj != position[j]):
                    translations.append(pj - position[j])
        if translations:
            dims[members] = np.linalg.matrix_rank(np.array(translations, dtype=float))
    return dims


def periodicity(edges):
    """
    Percolation dimensionality of a periodic network: the largest number of independent lattice
    translations along which one connected component extends (0 to 3).

    Parameters
    ----------
    edges: (N, 5) int array [source, target, ox, oy, oz]; directions are not needed

    Returns
    -------
    int
    """
    dims = _component_dims(edges)
    return int(dims.max()) if dims.size else 0


def _site_mapper(atoms, symprec):
    frac = atoms.get_scaled_positions(wrap=False)
    sym = spglib.get_symmetry((atoms.cell[:], frac, atoms.numbers), symprec=symprec)
    if sym is None:
        raise ValueError("spglib could not find the symmetry of the structure")
    cell = np.asarray(atoms.cell)
    tol = max(10 * symprec, 0.05)

    def locate(f):
        """Site index and lattice translation of the fractional point f."""
        diff = f - frac
        shift = np.round(diff)
        dist = np.linalg.norm((diff - shift) @ cell, axis=1)
        i = int(np.argmin(dist))
        if dist[i] > tol:
            raise ValueError(f"A symmetry image of a site is {dist[i]:.3f} A from any site: raise symprec")
        return i, shift[i].astype(int)

    return frac, sym["rotations"], sym["translations"], locate


def _images(edge, frac, rotations, translations, locate):
    """All symmetry images of the hop and of its reverse."""
    s, t, *o = (int(x) for x in edge)
    out = set()
    for R, tau in zip(rotations, translations):
        s2, ls = locate(R @ frac[s] + tau)
        t2, lt = locate(R @ (frac[t] + o) + tau)
        off = tuple(int(x) for x in lt - ls)
        out.add((s2, t2) + off)
        out.add((t2, s2) + tuple(-x for x in off))
    return out


def expand_edges(atoms, edges, symprec: float = 1e-3):
    """
    All hops symmetry-equivalent to the given ones (a hop and its reverse are equivalent).

    Returns
    -------
    network: (M, 5) int array, every hop of the network with its reverse
    inverse: (M,) int array, network[k] is equivalent to edges[inverse[k]]
    """
    edges = np.asarray(edges, dtype=int).reshape(-1, 5)
    frac, rotations, translations, locate = _site_mapper(atoms, symprec)
    label = {}
    for i, e in enumerate(edges):
        for image in _images(e, frac, rotations, translations, locate):
            label.setdefault(image, i)
    network = np.array(sorted(label), dtype=int).reshape(-1, 5)
    return network, np.array([label[tuple(e)] for e in network], dtype=int)


def unique_edges(atoms, edges, symprec: float = 1e-3):
    """
    Symmetry-inequivalent hops among ``edges`` (one per class, a hop and its reverse being one class).

    Returns
    -------
    unique: (K, 5) int array, the first hop of each class (in the order of ``edges``)
    inverse: (N,) int array, edges[k] is equivalent to unique[inverse[k]]
    """
    edges = np.asarray(edges, dtype=int).reshape(-1, 5)
    frac, rotations, translations, locate = _site_mapper(atoms, symprec)
    label, unique, inverse = {}, [], []
    for e in edges:
        key = (int(e[0]), int(e[1])) + tuple(int(x) for x in e[2:])
        if key not in label:
            for image in _images(e, frac, rotations, translations, locate):
                label.setdefault(image, len(unique))
            unique.append(e)
        inverse.append(label[key])
    return np.array(unique, dtype=int).reshape(-1, 5), np.array(inverse, dtype=int)


def percolation_barriers(network, emins, emaxs):
    """
    1D, 2D and 3D percolation barriers of a hop network from per-hop energies: the lowest emax at
    which the hops form a network percolating in that dimension, minus the lowest emin of the
    connected components that percolate (as ions' Percolator.find_percolation_barriers, restricted to
    the percolating components).

    Parameters
    ----------
    network: (N, 5) int array of hops
    emins, emaxs: (N,) arrays, lowest and highest energy along each hop (eV, one reference)

    Returns
    -------
    dict {'e1d': float, 'e2d': float, 'e3d': float}; np.inf where the network does not percolate
    """
    network = np.asarray(network, dtype=int).reshape(-1, 5)
    emins, emaxs = np.asarray(emins, dtype=float), np.asarray(emaxs, dtype=float)
    finite = np.flatnonzero(np.isfinite(emaxs) & np.isfinite(emins))
    out = {"e1d": np.inf, "e2d": np.inf, "e3d": np.inf}
    for E in np.unique(emaxs[finite]):
        open_ = finite[emaxs[finite] <= E]
        dims = _component_dims(network[open_])
        for dim in (1, 2, 3):
            if np.isinf(out[f"e{dim}d"]) and (dims >= dim).any():
                out[f"e{dim}d"] = float(E - emins[open_[dims >= dim]].min())
        if np.isfinite(out["e3d"]):
            break
    return out
