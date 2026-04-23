from ._geopyv_dev import *
import geopyv_dev._geopyv_dev as _core

from .plots import (
    inspect_subset,
    inspect_mesh,
    inspect_sequence,
    inspect_particle,
    inspect_field,
    convergence_subset,
    convergence_mesh,
    convergence_sequence,
    contour_mesh,
    contour_field,
)
from .wrappers import (
    SubsetWrapper,
    MeshWrapper,
    SequenceSolutionWrapper,
    ParticleWrapper,
    FieldWrapper,
)


# ---------------------------------------------------------------------------
# Auto-wrap helpers
# ---------------------------------------------------------------------------

def _wrap(raw):
    if isinstance(raw, _core.MeshSolution):
        return MeshWrapper(raw)
    if isinstance(raw, _core.SequenceSolution):
        return SequenceSolutionWrapper(raw)
    if isinstance(raw, _core.FieldSolution):
        return FieldWrapper(raw)
    if isinstance(raw, _core.ParticleSolution):
        return ParticleWrapper(raw)
    return raw


# ---------------------------------------------------------------------------
# IO — shadow _core.save / _core.load
# ---------------------------------------------------------------------------

def save(path, obj):
    """Serialise a solution object to a .pyv file.

    Accepts wrapped or raw MeshSolution / FieldSolution / SequenceSolution.
    """
    _core.save(path, getattr(obj, '_inner', obj))


def load(path):
    """Load a .pyv file and return an auto-wrapped solution object."""
    return _wrap(_core.load(path))


# ---------------------------------------------------------------------------
# Solve-wrapping classes — shadow the PyO3 originals
# ---------------------------------------------------------------------------

class Subset:
    """Reference subset for DIC. Wraps the Rust Subset with inspect/convergence."""

    def __init__(self, *args, **kwargs):
        self._inner = _core.Subset(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def solve_icgn(self, *args, **kwargs):
        return self._inner.solve_icgn(*args, **kwargs)

    def solve_fagn(self, *args, **kwargs):
        return self._inner.solve_fagn(*args, **kwargs)

    def inspect(self, **kwargs):
        return inspect_subset(self._inner, **kwargs)

    def convergence(self, **kwargs):
        return convergence_subset(self._inner, **kwargs)


class Mesh:
    """DIC mesh. solve() returns a MeshWrapper with inspect/convergence/contour."""

    def __init__(self, *args, **kwargs):
        self._inner = _core.Mesh(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def solve(self, *args, **kwargs):
        return MeshWrapper(self._inner.solve(*args, **kwargs))


class Sequence:
    """Multi-pair DIC sequence. solve() returns a SequenceSolutionWrapper."""

    def __init__(self, *args, **kwargs):
        self._inner = _core.Sequence(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def solve(self, *args, **kwargs):
        return SequenceSolutionWrapper(self._inner.solve(*args, **kwargs))


class Field:
    """Particle field. solve() returns a FieldWrapper with inspect/contour."""

    def __init__(self, *args, **kwargs):
        self._inner = _core.Field(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def solve(self, *args, **kwargs):
        return FieldWrapper(self._inner.solve(*args, **kwargs))


class Particle:
    """Lagrangian/Eulerian particle. solve() returns a ParticleWrapper."""

    def __init__(self, *args, **kwargs):
        self._inner = _core.Particle(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def solve(self, *args, **kwargs):
        return ParticleWrapper(self._inner.solve(*args, **kwargs))

    def solve_increment(self, *args, **kwargs):
        return self._inner.solve_increment(*args, **kwargs)
