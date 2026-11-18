
import importlib.util


class Percolyze:

    def __init__(self, data):
        self.data = data

    @staticmethod
    def _get_backend(backend):
        if backend == "auto":
            backend = (
                "numba"
                if importlib.util.find_spec("numba") is not None
                else "scipy"
            )

        if backend == "numba":
            try:
                from ._percolyze._percolyze_numba_backend import Percolyze as Impl
            except ImportError as exc:
                raise ImportError(
                    "Numba backend unavailable. "
                    "Install with: pip install 'gridmlip[fast]'"
                ) from exc

        elif backend == "scipy":
            from ._percolyze._percolyze_scipy_backend import Percolyze as Impl

        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. "
                "Choose 'auto', 'numba', or 'scipy'."
            )

        return Impl

    def percolation_barriers(
        self, upper_limit=10.0, backend="auto"
    ):
        Impl = self._get_backend(backend)
        return Impl(self.data).percolation_barriers(
            upper_limit=upper_limit
        )

    def percolation_radii(self, backend="auto"):
        Impl = self._get_backend(backend)
        return Impl(self.data).percolation_radii()
