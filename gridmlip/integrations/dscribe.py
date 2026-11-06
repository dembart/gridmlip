import numpy as np

def featurize_soap(
    cfgs,
    site_id=-1,
    r_cut=6.0,
    n_max=3,
    l_max=2,
    sigma=0.3,
    periodic=True,
    dtype="float64",
    normalize=True,
):

    try:
        from dscribe.descriptors import SOAP
    except ImportError as exc:
        raise ImportError(
            "SOAP descriptors require DScribe. "
            "Install with `pip install 'gridmlip[soap]'`."
        ) from exc
    
    if not cfgs:
        return np.empty((0, 0))

    species = sorted({
        symbol
        for atoms in cfgs
        for symbol in atoms.get_chemical_symbols()
    })

    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        periodic=periodic,
        sparse=False,
        dtype=dtype,
        average="off" if site_id is not None else "inner"
    )

    features = soap.create(
        cfgs,
        centers=[[site_id]] * len(cfgs) if site_id is not None else None,
    )

    if features.ndim > 2:
        features = np.squeeze(features)

    if normalize:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = np.divide(
            features,
            norms,
            out=np.zeros_like(features),
            where=norms > 0,
        )

    return features