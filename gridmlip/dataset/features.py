import numpy as np


def soap_features(
    cfgs,
    center=-1,
    r_cut: float = 6.0,
    n_max: int = 3,
    l_max: int = 2,
    sigma: float = 0.3,
    periodic: bool = True,
    normalize: bool = True,
    dtype: str = "float64",
):
    """
    SOAP descriptor of one atom per configuration (by default the mobile ion, the last atom).

    Parameters
    ----------
    cfgs: list of ase.Atoms
    center: int or None, -1 by default
        index of the atom to describe; None averages over all atoms
    r_cut, n_max, l_max, sigma, periodic:
        SOAP settings (see dscribe.descriptors.SOAP)
    normalize: bool, True by default
        scale each feature vector to unit length

    Returns
    -------
    np.ndarray of shape (len(cfgs), n_features); shape (0, 0) for an empty list
    """
    from ..integrations.dscribe import featurize_soap

    if len(cfgs) == 0:
        return np.empty((0, 0))
    X = featurize_soap(cfgs, site_id=center, r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma,
                       periodic=periodic, dtype=dtype, normalize=normalize)
    return np.atleast_2d(X)
