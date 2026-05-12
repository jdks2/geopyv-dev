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
    if isinstance(raw, _core.Subset):
        return Subset._new_from_inner(raw)
    if isinstance(raw, _core.Mesh):
        return Mesh._new_from_inner(raw)
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
    """Reference subset for DIC. Wraps PySubset with inspect/convergence."""

    def __init__(self, coord, local_mask, f_img, g_img, subset_order=1):
        self._inner = _core.Subset(coord, local_mask, f_img, g_img, subset_order=subset_order)

    @classmethod
    def _new_from_inner(cls, inner):
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def solve(self, p_0=None, algorithm="icgn",
              max_norm=1e-3, max_iterations=50, tolerance=0.75):
        """Run the DIC solver. Mutates the object; returns None.

        Parameters
        ----------
        p_0 : list[float], optional
            Initial warp vector. Defaults to zeros matching order.
        algorithm : str, optional
            "icgn" (default) or "fagn".
        max_norm : float, optional
            Convergence norm threshold. Default 1e-3.
        max_iterations : int, optional
            Iteration limit. Default 50.
        tolerance : float, optional
            Minimum acceptable C_ZNCC for solved=True. Default 0.75.
        """
        import warnings
        algo = algorithm.lower()
        kwargs = dict(p_0=p_0, max_norm=max_norm,
                      max_iterations=max_iterations, tolerance=tolerance)
        if algo == "icgn":
            self._inner.solve_icgn(**kwargs)
        elif algo == "fagn":
            self._inner.solve_fagn(**kwargs)
        else:
            warnings.warn(
                f"Unknown algorithm '{algo}'; falling back to 'icgn'.",
                UserWarning,
                stacklevel=2,
            )
            self._inner.solve_icgn(**kwargs)

    def save(self, path):
        """Save the solved subset to a .pyv file."""
        self._inner.save(path)

    def inspect(self, **kwargs):
        return inspect_subset(self._inner, **kwargs)

    def convergence(self, **kwargs):
        return convergence_subset(self._inner, **kwargs)


class Mesh:
    """DIC mesh. solve() mutates in place and returns None."""

    def __init__(self, boundary, target_nodes, f_img, g_img,
                 size=(1, 1000), exclusions=None, exclusions_hard=None, mesh_order=2):
        self._inner = _core.Mesh(
            boundary, target_nodes, f_img, g_img,
            size=size, exclusions=exclusions, exclusions_hard=exclusions_hard,
            mesh_order=mesh_order,
        )

    @classmethod
    def _new_from_inner(cls, inner):
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def solve(self, local_mask, seed_coord, seed_warp=None, **kwargs):
        """Run the DIC solver. Mutates in place; returns None."""
        self._inner.solve(local_mask, seed_coord, seed_warp=seed_warp, **kwargs)

    def save(self, path):
        """Save the solved mesh to a .pyv file."""
        self._inner.save(path)

    def inspect(self, **kwargs):
        return inspect_mesh(self._inner, **kwargs)

    def convergence(self, quantity="C_ZNCC", **kwargs):
        return convergence_mesh(self._inner, quantity, **kwargs)

    def contour(self, quantity, **kwargs):
        return contour_mesh(self._inner, quantity, **kwargs)


class Sequence:
    """Multi-pair DIC sequence. solve() returns a SequenceSolutionWrapper."""

    def __init__(self, image_dir, boundary, target_nodes,
                 size=(1.0, 1000.0), exclusions=None, mesh_order=2):
        self._inner = _core.Sequence(
            image_dir, boundary, target_nodes,
            size=size, exclusions=exclusions, mesh_order=mesh_order,
        )

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
