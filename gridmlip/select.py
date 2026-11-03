import numpy as np
import fpsample

def fps_with_stop(
    points: np.ndarray,
    max_samples: int = 100,
    rel_tol: float = 0.01,
    start_idx: int | None | str = 'centroid',
    rollback: bool = True,
    h: int = 5,
) -> np.ndarray:
    """
    FPS sampling with a relative-distance-change stopping criterion.

    Parameters
    ----------

    points : np.ndarray
        sample features
    
    max_samples : int
        maximum number of samples to select
    
    rel_tol : float
        stopping criterion for relative distance change
    
    start_idx : int | None | "centroid"
        index of the first sample
        if "centroid" the nearest to the centroid sample will be used

    rollback : bool
        include the point breaking the stopping criterion

    h : int
        parameters of the fpsample.bucket_fps_kdline_sampling
        see the fpsample docs

    

    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    N, D = points.shape
    if max_samples >= N:
        max_samples = N
    assert max_samples > 0

    # over-sample with fpsample

    if start_idx == 'centroid':
        centroid = points.mean(axis=0)
        start_idx = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))
        
    if D > 8:
        # KD-tree backend supports at most 8 dims
        indices = fpsample.fps_sampling(
            points, max_samples, start_idx=start_idx
        ).astype(np.int64)
    else:
        # clamp h so the bucket size is valid: 2**h <= N
        
        h_eff = max(1, min(h, int(np.floor(np.log2(max(N, 1))))))

        indices = fpsample.bucket_fps_kdline_sampling(
            points, max_samples, h=h_eff, start_idx=start_idx
        ).astype(np.int64)

    if len(indices) <= 2:
        return indices

    # traverse and apply stopping criterion
    selected = [int(indices[0])]
    min_dists = np.linalg.norm(points - points[indices[0]], axis=1)
    prev_d = None

    for k in range(1, len(indices)):
        idx = int(indices[k])
        curr_d = float(min_dists[idx])

        if prev_d is not None and prev_d > 0.0:
            if abs(curr_d - prev_d) / prev_d < rel_tol:
                if not rollback:
                    selected.append(idx)
                break

        selected.append(idx)
        prev_d = curr_d

        new_d = np.linalg.norm(points - points[idx], axis=1)
        np.minimum(min_dists, new_d, out=min_dists)

    return np.asarray(selected, dtype=np.int64)