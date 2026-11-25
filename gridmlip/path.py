import dataclasses
from pathlib import Path as _FilePath

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order, dijkstra

from ase.io import write

__author__ = "Artem Dembitskiy"


@dataclasses.dataclass
class Path:
    """
    Lowest-barrier path of a mobile ion between two sites of an energy map,
    as returned by ``EnergyMap.path()`` and ``HopMap.path()``.

    Attributes
    ----------
    points: np.ndarray of shape (n_images, 3)
        Cartesian points from start to end, equally spaced along the route
        (unwrapped, the exact sites at both ends)

    level: float
        lowest energy at which the two sites are connected on the map (eV)

    barrier: float
        level minus the lower of the two site energies (eV)

    e_start, e_end: float
        energies of the two sites (lowest mesh points near them)

    saddle: np.ndarray of shape (3,)
        highest point of the route

    route: np.ndarray of shape (m, 3)
        the route through mesh points before resampling

    energies: np.ndarray of shape (n_images,)
        map energies at the points, interpolated along the route (eV)

    hop: Hop, optional
        the hop the path belongs to; needed for ``images()`` and ``write()``

    Example
    -------
    path = emap.path(p0, p1, n_images=12)
    path.barrier
    """

    points: np.ndarray
    level: float
    barrier: float
    e_start: float
    e_end: float
    saddle: np.ndarray
    route: np.ndarray
    energies: np.ndarray
    hop: object = None

    def __repr__(self):
        hop = f", hop={self.hop.source}->{self.hop.target}" if self.hop is not None else ""
        return f"Path(n_images={len(self.points)}, barrier={self.barrier:.3f} eV{hop})"

    def images(self):
        """
        NEB images: the hop's template with the moving ion placed on each point.

        Returns
        -------
        list of ase.Atoms with a boolean "moving" array (the moving ion)
        """
        if self.hop is None:
            raise ValueError("images() needs a hop: use HopMap.path() or Hops")
        images = []
        for p in self.points:
            image = self.hop.template.copy()
            image.positions[self.hop.moving] = p
            images.append(image)
        return images

    def write(self, filename, format: str = "extxyz"):
        """
        Write the NEB images.

        Parameters
        ----------
        filename: str or Path
            a file for 'extxyz' (all images), a folder for 'vasp' (00/POSCAR, 01/POSCAR, ...)

        format: str, 'extxyz' by default
            'extxyz' or 'vasp'
        """
        _write_images(self.images(), filename, format)


def _write_images(images, target, format):
    if format == "extxyz":
        write(str(target), images, format="extxyz")
    elif format == "vasp":
        folder = _FilePath(target)
        width = max(2, len(str(len(images) - 1)))
        for i, image in enumerate(images):
            sub = folder / f"{i:0{width}d}"
            sub.mkdir(parents=True, exist_ok=True)
            write(str(sub / "POSCAR"), image, format="vasp", direct=True, sort=False)
    else:
        raise ValueError(f"Unknown format {format!r}: use 'extxyz' or 'vasp'")


def _mic(vectors, cell):
    frac = np.linalg.solve(cell.T, np.atleast_2d(vectors).T).T
    frac -= np.round(frac)
    return frac @ cell


def _neighbour_graph(values, unwrapped, cell, weight):
    """
    Sparse graph of the finite mesh points: every point is linked to its 26 neighbours
    (periodic mesh) only where their unwrapped positions are one mesh step apart,
    which cuts the mesh at the seam half a cell away from the reference point.
    """
    shape = np.asarray(values.shape)
    finite = np.flatnonzero(np.isfinite(values.ravel()))
    idx = np.array(np.unravel_index(finite, values.shape)).T
    pos = -np.ones(values.size, dtype=int)
    pos[finite] = np.arange(len(finite))
    rows, cols, vectors = [], [], []
    for step in [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]:
        nb = pos[np.ravel_multi_index(((idx + step) % shape).T, values.shape)]
        r = np.flatnonzero(nb >= 0)
        c = nb[r]
        d = unwrapped[c] - unwrapped[r]
        same = np.linalg.norm(d - (np.asarray(step) / shape) @ cell, axis=1) < 1e-6
        rows.append(r[same])
        cols.append(c[same])
        vectors.append(d[same])
    rows, cols, vectors = np.concatenate(rows), np.concatenate(cols), np.concatenate(vectors)
    n = len(finite)
    graph = coo_matrix((weight(rows, cols, vectors), (rows, cols)), shape=(n, n)).tocsr()
    return graph + graph.T


def minimum_energy_path(values, cell, start, end, n_images=12, well_radius=0.5, margin=0.0):
    """
    Lowest-barrier path between two sites of a periodic energy map.

    1. Mesh points are placed at their minimum-image positions relative to `start`
       (the search covers half a cell around it) and linked to their 26 neighbours.
    2. The level is the lowest energy at which the two sites are connected: the
       highest point on the minimum spanning tree path (edge weight = higher energy
       of its two points), i.e. the bottleneck (minimax) path.
    3. The route is the shortest one (Euclidean length) through points with
       energy <= level + margin; it is resampled to `n_images` equally spaced points.

    Parameters
    ----------
    values: np.ndarray of shape (nx, ny, nz)
        energies on the mesh, +inf where not accessible

    cell: (3, 3) array
        cell of the mesh

    start, end: (3,) arrays
        Cartesian positions of the two sites; end - start must be shorter than
        half the cell in every direction

    n_images: int, 12 by default
        number of points of the path, endpoints included

    well_radius: float, 0.5 by default
        each site is the lowest accessible mesh point within this distance (Angstrom)

    margin: float, 0.0 by default
        energy allowance above the level for the route (eV)

    Returns
    -------
    Path
    """
    values = np.asarray(values, dtype=float)
    cell = np.asarray(cell, dtype=float)
    shape = np.asarray(values.shape)
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)

    if np.linalg.norm(_mic(end - start, cell)[0] - (end - start)) > 1e-6:
        raise ValueError("end - start is longer than half the cell: use a larger supercell")

    flat = values.ravel()
    finite = np.flatnonzero(np.isfinite(flat))
    energies = flat[finite]
    idx = np.array(np.unravel_index(finite, values.shape)).T
    unwrapped = start + _mic((idx / shape) @ cell - start, cell)

    def well(p):
        near = np.flatnonzero(np.linalg.norm(unwrapped - p, axis=1) <= well_radius)
        if len(near) == 0:
            raise ValueError(f"No accessible mesh point within {well_radius} A of {p}")
        return near[np.argmin(energies[near])]

    s, t = well(start), well(end)

    # bottleneck level along the minimum spanning tree (weights shifted to stay > 0)
    shift = 1.0 - energies.min()
    graph = _neighbour_graph(values, unwrapped, cell, lambda r, c, v: np.maximum(energies[r], energies[c]) + shift)
    tree = minimum_spanning_tree(graph)
    _, pred = breadth_first_order(tree + tree.T, s, directed=False, return_predecessors=True)
    if t != s and pred[t] < 0:
        raise ValueError("The two sites are not connected inside the evaluated region")
    node, level = t, max(energies[s], energies[t])
    while node != s:
        level = max(level, energies[node])
        node = pred[node]

    # shortest route through points at or below the level
    allowed = energies <= level + margin + 1e-12
    length = _neighbour_graph(values, unwrapped, cell, lambda r, c, v: np.linalg.norm(v, axis=1)).tocoo()
    ok = allowed[length.row] & allowed[length.col]
    sub = coo_matrix((length.data[ok], (length.row[ok], length.col[ok])), shape=length.shape).tocsr()
    _, prev = dijkstra(sub, directed=False, indices=s, return_predecessors=True)
    nodes = [t]
    while nodes[-1] != s:
        nodes.append(prev[nodes[-1]])
    nodes = nodes[::-1]

    route = np.vstack([start, unwrapped[nodes[1:-1]], end]) if len(nodes) > 2 else np.vstack([start, end])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))])
    targets = np.linspace(0.0, arc[-1], n_images)
    points = np.array([[np.interp(x, arc, route[:, c]) for c in range(3)] for x in targets])
    points[0], points[-1] = start, end
    along = np.concatenate([[energies[s]], energies[nodes[1:-1]], [energies[t]]]) if len(nodes) > 2 \
        else np.array([energies[s], energies[t]])

    return Path(
        points=points,
        level=float(level),
        barrier=float(level - min(energies[s], energies[t])),
        e_start=float(energies[s]),
        e_end=float(energies[t]),
        saddle=unwrapped[nodes[int(np.argmax(energies[nodes]))]],
        route=route,
        energies=np.interp(targets, arc, along),
    )
