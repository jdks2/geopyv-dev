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
        return plots.inspect_mesh(self._inner, **kwargs)

    def convergence(self, quantity="C_ZNCC", **kwargs):
        return plots.convergence_mesh(self._inner, quantity, **kwargs)

    def contour(self, quantity, **kwargs):
        return plots.contour_mesh(self._inner, quantity, **kwargs)


class SequenceSolutionWrapper:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    @property
    def mesh_solutions(self):
        return [MeshWrapper(m) for m in self._inner.mesh_solutions]

    def inspect(self, mesh_idx, **kwargs):
        return plots.inspect_sequence(self, mesh_idx=mesh_idx, **kwargs)

    def convergence(self, mesh_idx=None, quantity="C_ZNCC", **kwargs):
        return plots.convergence_sequence(self, mesh_idx=mesh_idx, quantity=quantity, **kwargs)


class ParticleWrapper:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def inspect(self, **kwargs):
        return plots.inspect_particle(self._inner, **kwargs)


class FieldWrapper:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    @property
    def particles(self):
        return [ParticleWrapper(p) for p in self._inner.particles]

    def inspect(self, **kwargs):
        return plots.inspect_field(self._inner, **kwargs)

    def contour(self, quantity, **kwargs):
        return plots.contour_field(self._inner, quantity, **kwargs)
