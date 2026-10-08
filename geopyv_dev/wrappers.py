from geopyv_dev import plots


class SubsetWrapper:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def inspect(self, **kwargs):
        return plots.inspect_subset(self._inner, **kwargs)

    def convergence(self, **kwargs):
        return plots.convergence_subset(self._inner, **kwargs)


class MeshWrapper:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def inspect(self, **kwargs):
        """See ``geopyv_dev.plots.inspect_mesh`` -- accepts ``subset_idx=``,
        ``zones=True`` (zonal-masking map), ``show_areas=True``."""
        return plots.inspect_mesh(self._inner, **kwargs)

    def convergence(self, quantity="C_ZNCC", **kwargs):
        return plots.convergence_mesh(self._inner, quantity, **kwargs)

    def contour(self, quantity, **kwargs):
        return plots.contour_mesh(self._inner, quantity, **kwargs)


class ParticleWrapper:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def inspect(self, **kwargs):
        return plots.inspect_particle(self._inner, **kwargs)

    def history(self, quantity="warps", **kwargs):
        return plots.history_particle(self._inner, quantity, **kwargs)

    def trace(self, quantity="warps", component=0, **kwargs):
        return plots.trace_particle(self._inner, quantity, component, **kwargs)


