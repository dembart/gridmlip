class PointModel:
    """
    A model evaluated directly at Cartesian points (no configurations are built),
    e.g. bond valence site energies.

    ``Grid.evaluate(model)`` calls, in order:

    1. ``host = model.prepare(atoms, host_mask)``: the host atoms the model needs
       (same atoms and order as ``atoms[host_mask]``, with any extra per-atom arrays);
    2. ``model.symmetry_labels(host)``: per-host-atom labels that symmetry must preserve,
       or None (the grid then drops symmetry operations that would mix different labels);
    3. ``model(points, host)``: energies at the points;
    4. ``model.describe(host)``: provenance stored in ``EnergyMap.info``.
    """

    def prepare(self, atoms, host_mask):
        return atoms[host_mask]

    def symmetry_labels(self, host):
        return None

    def __call__(self, points, host):
        raise NotImplementedError

    def describe(self, host):
        return {"model": type(self).__name__}
