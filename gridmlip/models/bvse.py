import re
import warnings
from collections import Counter

import numpy as np
from ase.data import atomic_numbers, covalent_radii

from .._neighborhood import nn_list
from .base import PointModel

__author__ = "Artem Dembitskiy"

# bvlain settings, plus S- (no BVSE parameters for S-1)
_FORBIDDEN_SPECIES = ["O-", "P3-", "S-"]

# oxidation states that are common enough not to warn about
_UNUSUAL = {"P": {3, 4}, "O": {-1}}


def parse_ion(ion):
    """'Li+' -> ('Li', 3, 1); 'O2-' -> ('O', 8, -2); 'Mg2+' -> ('Mg', 12, 2)."""
    element = re.sub(r"[\d+\-]", "", ion)
    digits = re.sub(r"\D", "", ion)
    sign = -1 if "-" in ion else 1
    charge = sign * (int(digits) if digits else 1)
    return element, atomic_numbers[element], charge


def assign_oxidation_states(atoms, forbidden_species=_FORBIDDEN_SPECIES):
    """
    Oxidation states from pymatgen's BVAnalyzer (bvlain settings; O-, P3- and S- forbidden).

    Returns
    -------
    np.ndarray of int, one per atom
    """
    try:
        from pymatgen.analysis.bond_valence import BVAnalyzer
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise ImportError(
            "Assigning oxidation states requires pymatgen (`pip install pymatgen`), or set "
            "atoms.arrays['oxi_states'] / pass BVSE(oxidation_states=...)."
        ) from exc
    atoms = atoms.copy()
    for key in ("oxi_states", "initial_charges"):
        if key in atoms.arrays:
            del atoms.arrays[key]
    structure = AseAtomsAdaptor.get_structure(atoms)
    decorated = BVAnalyzer(forbidden_species=forbidden_species).get_oxi_state_decorated_structure(structure)
    return np.array([int(round(site.specie.oxi_state)) for site in decorated])


def _summary(symbols, charges):
    counts = Counter(zip(symbols, charges))
    return ", ".join(f"{s}{q:+d}×{n}" for (s, q), n in sorted(counts.items()))


class BVSE(PointModel):
    """
    Bond valence site energy of a mobile ion.

    Oxidation states of the structure are taken, in this order, from
    ``oxidation_states``, ``atoms.arrays['oxi_states']``, or assigned with
    pymatgen's BVAnalyzer (printed, and recorded in ``EnergyMap.info``).

    Example
    -------
    emap = Grid(atoms, specie="Na", resolution=0.2).evaluate(BVSE("Na+"))
    """

    def __init__(
        self,
        ion: str,
        cutoff: float = 10.0,
        k: int = 100,
        f: float = 0.74,
        oxidation_states=None,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        ion: str
            mobile ion with its charge, e.g. 'Li+', 'Mg2+', 'F-'

        cutoff: float, 10.0 by default
            interaction cutoff (Angstrom)

        k: int, 100 by default
            maximum number of neighbors within the cutoff

        f: float, 0.74 by default
            screening factor of the Coulomb term

        oxidation_states: dict or array, optional
            {'Fe': 3, 'O': -2, ...} or one value per atom; overrides atoms.arrays['oxi_states']

        verbose: bool, True by default
            print oxidation states when they are assigned automatically
        """
        self.ion = ion
        self.symbol, self.z, self.charge = parse_ion(ion)
        self.cutoff = cutoff
        self.k = k
        self.f = f
        self.oxidation_states = oxidation_states
        self.verbose = verbose
        self._source = None
        self._charges = None

    def __repr__(self):
        return f"BVSE({self.ion!r}, cutoff={self.cutoff}, k={self.k}, f={self.f})"

    def resolve_oxidation_states(self, atoms):
        """Oxidation states for all atoms of the structure (see class docstring)."""
        if self.oxidation_states is not None:
            if isinstance(self.oxidation_states, dict):
                charges = np.array([self.oxidation_states[s] for s in atoms.get_chemical_symbols()])
            else:
                charges = np.asarray(self.oxidation_states)
            source = "user (BVSE(oxidation_states=...))"
        elif "oxi_states" in atoms.arrays:
            charges = np.asarray(atoms.arrays["oxi_states"])
            source = "atoms.arrays['oxi_states']"
        else:
            charges = assign_oxidation_states(atoms)
            source = "pymatgen BVAnalyzer"
            symbols = atoms.get_chemical_symbols()
            if self.verbose:
                print(f"BVSE: oxidation states assigned by {source}: {_summary(symbols, charges)}")
            odd = sorted({f"{s}{q:+d}" for s, q in zip(symbols, charges) if q in _UNUSUAL.get(s, ())})
            mixed = sorted(s for s in set(symbols)
                           if len({q for t, q in zip(symbols, charges) if t == s}) > 1)
            if odd or mixed:
                warnings.warn(
                    "Check the assigned oxidation states: "
                    + (f"unusual {', '.join(odd)}; " if odd else "")
                    + (f"mixed valence of {', '.join(mixed)}" if mixed else ""),
                    UserWarning, stacklevel=3,
                )
        charges = np.asarray(charges, dtype=int)
        if charges.shape != (len(atoms),):
            raise ValueError(f"Expected {len(atoms)} oxidation states, got {charges.shape}")
        self._source = source
        return charges

    # ------------------------------------------------------------ PointModel

    def prepare(self, atoms, host_mask):
        from ions.decorate import assign_bvse_params

        atoms = atoms.copy()
        atoms.set_array("oxi_states", self.resolve_oxidation_states(atoms))
        host = atoms[host_mask]
        try:
            return assign_bvse_params(host, self.symbol, self.charge, self_interaction=True)
        except KeyError:
            ions = sorted({f"{s}{int(q):+d}" for s, q in zip(host.get_chemical_symbols(), host.get_array("oxi_states"))
                           if q * self.charge < 0})
            raise ValueError(
                f"No BVSE parameters for {self.ion} with some of {ions}. Check the oxidation states "
                "(e.g. BVSE(oxidation_states={'P': 5, 'S': -2, ...}) or atoms.arrays['oxi_states'])."
            ) from None

    def symmetry_labels(self, host):
        pairs = np.stack([host.numbers, host.get_array("oxi_states")], axis=1)
        _, labels = np.unique(pairs, axis=0, return_inverse=True)
        return labels.ravel()

    def __call__(self, points, host):
        from ions.calculators.potential import BVFF
        from ions.data import principle_number

        q1 = self.charge
        n1, rc1 = principle_number[self.z], covalent_radii[self.z]

        distances, _, ids = nn_list(host.positions, points, r_cut=self.cutoff,
                                    cell=host.cell, k=self.k, return_orig_ids=True)

        def take(name):
            # index -1 ("no neighbor") picks the appended nan
            arr = np.concatenate([np.asarray(host.get_array(name), dtype=float), [np.nan]])
            return np.take(arr, ids, axis=-1)

        q2 = take("oxi_states")
        q2 = np.where(q2 * q1 > 0, q2, 0)
        morse = np.nan_to_num(
            BVFF.Morse(distances, take("r_min"), take("d0"), take("alpha")), copy=False, nan=0.0
        ).sum(axis=1)
        coulomb = np.nan_to_num(
            BVFF.Coulomb(q1, q2, distances, rc1, take("rc2"), n1, take("n2"), f=self.f),
            copy=False, nan=0.0,
        ).sum(axis=1)
        return morse + coulomb

    def describe(self, host):
        return {
            "model": repr(self),
            "oxidation_states": _summary(host.get_chemical_symbols(), host.get_array("oxi_states")),
            "oxidation_states_source": self._source,
        }


def bvse(
    atoms,
    ion: str,
    resolution: float = 0.2,
    r_min: float = 0.8,
    cutoff: float = 10.0,
    k: int = 100,
    f: float = 0.74,
    symprec: float = 0.1,
    empty_framework: bool = True,
    oxidation_states=None,
    verbose: bool = True,
):
    """
    Bond valence site energy map of a mobile ion in one call.

    Example
    -------
    emap = gm.bvse(atoms, "Na+")
    emap.percolation_barriers()

    Returns
    -------
    EnergyMap
    """
    from ..core import Grid

    model = BVSE(ion, cutoff=cutoff, k=k, f=f, oxidation_states=oxidation_states, verbose=verbose)
    atoms = atoms.copy()
    # charges first, so that the grid symmetry already respects charge ordering
    atoms.set_array("oxi_states", model.resolve_oxidation_states(atoms))
    model.oxidation_states = atoms.get_array("oxi_states")
    source = model._source
    grid = Grid(atoms, specie=model.z, resolution=resolution, r_cut=cutoff, r_min=r_min,
                symprec=symprec, empty_framework=empty_framework)
    emap = grid.evaluate(model)
    emap.info["oxidation_states_source"] = source
    return emap
