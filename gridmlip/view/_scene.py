"""Scene data for the viewer: atoms, VESTA polyhedra and bonds, low-energy sites, the energy grid."""
import numpy as np

from ._vesta import VESTA, VESTA_POLYHEDRA


def _inside(p, cell, tol=1e-3):
    f = np.linalg.solve(np.asarray(cell).T, p)
    return bool(np.all(f > -tol) and np.all(f < 1 - tol))


def polyhedra(atoms, mobile, min_cn=4):
    """
    Coordination polyhedra as VESTA draws them by default.

    Centers, ligands and maximum bond lengths follow VESTA's pair rules (VESTA_POLYHEDRA): an atom is
    a center if its element has rules, and its ligands are the atoms of the paired elements within
    the rule's maximum distance (periodic images included). The mobile ion is not a center.

    Returns a list of (symbol, index, ligand info, ligand xyz).
    """
    from ase.neighborlist import neighbor_list

    sym = np.array(atoms.get_chemical_symbols())
    present = set(sym)
    rules = {c: {l: d for l, d in r.items() if l in present}
             for c, r in VESTA_POLYHEDRA.items() if c in present and c != mobile}
    rules = {c: r for c, r in rules.items() if r}
    if not rules:
        return []
    i, j, d, S = neighbor_list("ijdS", atoms, max(max(r.values()) for r in rules.values()))
    cell = np.asarray(atoms.cell)
    out = []
    for c in np.flatnonzero(np.isin(sym, list(rules))):
        r = rules[sym[c]]
        keep = [k for k in np.flatnonzero(i == c) if sym[j[k]] in r and 0 < d[k] <= r[sym[j[k]]]]
        if len(keep) >= min_cn:
            ligands = [{"elem": str(sym[j[k]]), "idx": int(j[k]), "image": [int(x) for x in S[k]]} for k in keep]
            xyz = np.array([atoms.positions[j[k]] + S[k] @ cell for k in keep])
            out.append((str(sym[c]), int(c), ligands, xyz))
    return out


def hull_mesh(ligands):
    """Flat-shaded triangles and the real edges of the convex hull of the ligands."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(ligands)
    center = ligands.mean(axis=0)
    vertices, normals, faces = [], [], []
    for tri in hull.simplices:
        a, b, c = ligands[tri]
        n = np.cross(b - a, c - a)
        if np.dot(n, a - center) < 0:
            b, c, n = c, b, -n
        n = n / np.linalg.norm(n)
        k = len(vertices)
        vertices += [a.tolist(), b.tolist(), c.tolist()]
        normals += [n.tolist()] * 3
        faces += [k, k + 1, k + 2]
    edges = set()
    for tri in hull.simplices:
        for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
            edges.add(tuple(sorted(e)))
    real = []
    for i, j in edges:
        fs = [k for k, tri in enumerate(hull.simplices) if i in tri and j in tri]
        if len(fs) == 2 and abs(abs(np.dot(hull.equations[fs[0], :3], hull.equations[fs[1], :3])) - 1) < 1e-6:
            continue  # splits a flat face, not a real edge
        real.append([ligands[i].tolist(), ligands[j].tolist()])
    return vertices, normals, faces, real


def payload(emap, mobile):
    """Everything the page needs, as JSON-serializable data (one unit cell; the page builds supercells)."""
    atoms = emap.atoms
    vmin = emap.values[np.isfinite(emap.values)].min()
    values = emap.values - vmin
    finite = np.isfinite(values)
    values = np.where(finite, values, values[finite].max())  # blocked points never inside a surface
    barriers = emap.percolation_barriers()
    oxi = atoms.get_array("oxi_states") if "oxi_states" in atoms.arrays else np.zeros(len(atoms))
    cell = np.asarray(atoms.cell)

    polys, images = [], []
    from scipy.spatial import ConvexHull

    for symbol, i, lig_info, ligands in polyhedra(atoms, mobile):
        try:
            v, n, f, e = hull_mesh(ligands)
        except Exception:  # coplanar ligands (e.g. square-planar): no 3D polyhedron
            continue
        d = np.linalg.norm(ligands - atoms.positions[i], axis=1)
        order = np.argsort(d)
        polys.append({
            "center": f"{symbol}{i}", "elem": symbol, "idx": i, "q": float(oxi[i]),
            "color": VESTA.get(symbol, ("#999999", 1.0))[0],  # VESTA: polyhedra in the central atom's color
            "vertices": v, "normals": n, "faces": f, "edges": e,
            "ligands": [{**lig_info[k], "d": float(d[k]), "xyz": ligands[k].tolist()} for k in order],
            "d_mean": float(d.mean()),
            "distortion": float(np.mean(np.abs(d - d.mean())) / d.mean()),  # Baur distortion index
            "volume": float(ConvexHull(ligands).volume),
        })
        images += [(l["elem"], p) for l, p in zip(lig_info, ligands) if not _inside(p, cell)]
    seen, image_atoms = set(), []
    for s, p in images:  # a ligand shared by two polyhedra appears once
        key = tuple(np.round(p, 3))
        if key not in seen:
            seen.add(key)
            image_atoms.append({"elem": s, "xyz": p.tolist(),
                                "frac": np.linalg.solve(cell.T, p).round(4).tolist()})

    window = barriers["e1d"] if np.isfinite(barriers["e1d"]) else 0.5
    minima = [{"xyz": np.asarray(pos).tolist(), "e": float(e - vmin)}
              for pos, e, _ in emap.local_minima(min_distance=1.0, energy_window=window, coord_type="cart")]

    return {
        "atoms": [{"elem": s, "xyz": p.tolist(), "q": float(q), "frac": f.round(4).tolist()}
                  for s, p, q, f in zip(atoms.get_chemical_symbols(), atoms.positions, oxi,
                                        atoms.get_scaled_positions(wrap=False))],
        "images": image_atoms,
        "mobile": mobile,
        "elements": {e: {"color": VESTA.get(e, ("#999999", 1.0))[0], "radius": 0.4 * VESTA.get(e, ("", 1.0))[1]}
                     for e in set(atoms.get_chemical_symbols()) | {l["elem"] for l in image_atoms}},
        "polyhedra": polys,
        "minima": minima,
        "cell": cell.tolist(),
        "barriers": {k: (float(v) if np.isfinite(v) else None) for k, v in barriers.items()},
        "vmax": float(np.percentile(values, 99)),
        "title": f"{atoms.get_chemical_formula()} · {emap.info.get('model', '')}",
        "grid": {"n": list(values.shape), "values": " ".join("%.4g" % v for v in values.ravel())},
    }
