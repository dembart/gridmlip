import numpy as np

from ..select import fps_with_stop


def select_fps(
    X,
    pca: float | int | None = 0.95,
    max_samples: int = 500,
    rel_tol: float = 1e-5,
    start_idx="centroid",
):
    """
    Farthest point sampling of feature vectors (deterministic).

    Parameters
    ----------
    X: np.ndarray of shape (n, d)
        features, e.g. from soap_features()
    pca: float, int or None, 0.95 by default
        reduce X with PCA first (full SVD): a float keeps that fraction of the variance,
        an int that many components; None skips PCA
    max_samples: int, 500 by default
        upper limit on the number of selected points
    rel_tol: float, 1e-5 by default
        stop when the FPS distance changes by less than this relative amount between steps
    start_idx: int or 'centroid'
        first point; 'centroid' = the point closest to the mean

    Returns
    -------
    np.ndarray of int: indices into X, in selection order
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n == 0 or max_samples <= 0:
        return np.empty(0, dtype=np.int64)
    if n <= 2:
        return np.arange(min(n, max_samples), dtype=np.int64)
    if pca is not None:
        from sklearn.decomposition import PCA
        X = PCA(n_components=pca, svd_solver="full").fit_transform(X)
    return fps_with_stop(X, max_samples=min(max_samples, n), rel_tol=rel_tol, start_idx=start_idx)
