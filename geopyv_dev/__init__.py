from ._geopyv_dev import *
import geopyv_dev._geopyv_dev as _core

from .calibration import Calibration, CalibrationParams

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
    contour_sequence,
    contour_field,
    history_particle,
    history_field,
    trace_particle,
    trace_field,
    standard_error_validation,
    mean_error_validation,
    noise_standard_error_validation,
    noise_mean_error_validation,
    strain_error_validation,
    spatial_error_validation,
)
from .wrappers import (
    SubsetWrapper,
    MeshWrapper,
    ParticleWrapper,
)



# ---------------------------------------------------------------------------
# Auto-wrap helpers
# ---------------------------------------------------------------------------

def _wrap(raw):
    if isinstance(raw, _core.Subset):
        return Subset._new_from_inner(raw)
    if isinstance(raw, _core.Mesh):
        return Mesh._new_from_inner(raw)
    if isinstance(raw, _core.Sequence):
        return Sequence._new_from_inner(raw)
    if isinstance(raw, _core.Particle):
        return Particle._new_from_inner(raw)
    if isinstance(raw, _core.Field):
        return Field._new_from_inner(raw)
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
    """Multi-pair DIC sequence. solve() mutates in place and returns None."""

    def __init__(self, image_dir, boundary, target_nodes,
                 size=(1.0, 1000.0), exclusions=None, mesh_order=2):
        self._inner = _core.Sequence(
            image_dir, boundary, target_nodes,
            size=size, exclusions=exclusions, mesh_order=mesh_order,
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

    def solve(self, *args, **kwargs):
        self._inner.solve(*args, **kwargs)

    @property
    def mesh_solutions(self):
        return [MeshWrapper(m) for m in self._inner.mesh_solutions]

    @property
    def meshes(self):
        return self.mesh_solutions

    @property
    def solved(self):
        return self._inner.solved

    @property
    def unsolvable(self):
        return self._inner.unsolvable

    @property
    def override_log(self):
        return self._inner.override_log

    @property
    def reference_updates(self):
        return self._inner.reference_updates

    @property
    def mesh_paths(self):
        return self._inner.mesh_paths

    def mesh_solution_at(self, idx):
        return MeshWrapper(self._inner.mesh_solution_at(idx))

    def all_c_zncc(self):
        return self._inner.all_c_zncc()

    def inspect(self, mesh_idx, subset_idx=None, **kwargs):
        return inspect_sequence(self, mesh_idx=mesh_idx, subset_idx=subset_idx, **kwargs)

    def convergence(self, mesh_idx=None, quantity="C_ZNCC", **kwargs):
        return convergence_sequence(self, mesh_idx=mesh_idx, quantity=quantity, **kwargs)

    def contour(self, quantity, mesh_idx, **kwargs):
        return contour_sequence(self, mesh_idx=mesh_idx, quantity=quantity, **kwargs)

    def save(self, path):
        _core.save(path, self._inner)


class Field:
    """Particle field built from a solved Sequence. solve() mutates in place and returns None."""

    def __init__(self, sequence_solution, track=True, depth=1.0,
                 coordinates=None, volumes=None):
        raw = getattr(sequence_solution, '_inner', sequence_solution)
        self._inner = _core.Field(raw, track=track, depth=depth,
                                  coordinates=coordinates, volumes=volumes)

    @classmethod
    def _new_from_inner(cls, inner):
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def solve(self, factor=0.0, true_incs=True, calibration=None):
        cal = getattr(calibration, '_inner', calibration)
        self._inner.solve(factor=factor, true_incs=true_incs, calibration=cal)

    @property
    def particles(self):
        return [ParticleWrapper(p) for p in self._inner.particles]

    def inspect(self, **kwargs):
        return inspect_field(self, **kwargs)

    def contour(self, quantity, **kwargs):
        return contour_field(self, quantity, **kwargs)

    def history(self, particle_index, quantity="warps", **kwargs):
        return history_field(self, particle_index, quantity, **kwargs)

    def trace(self, quantity="warps", component=0, **kwargs):
        return trace_field(self, quantity, component, **kwargs)

    def save(self, path):
        _core.save(path, self._inner)


class Particle:
    """Lagrangian/Eulerian particle. solve() mutates in place and returns None."""

    def __init__(self, source, coordinate, initial_warp=None, track=True):
        raw = getattr(source, '_inner', source)
        self._inner = _core.Particle(raw, coordinate, initial_warp=initial_warp, track=track)

    @classmethod
    def _new_from_inner(cls, inner):
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self):
        return repr(self._inner)

    def solve(self, factor=0.0, true_incs=True, calibration=None):
        cal = getattr(calibration, '_inner', calibration)
        self._inner.solve(factor=factor, true_incs=true_incs, calibration=cal)

    def solve_increment(self, m):
        return self._inner.solve_increment(m)

    def inspect(self, **kwargs):
        return inspect_particle(self._inner, **kwargs)

    def history(self, quantity="warps", **kwargs):
        return history_particle(self._inner, quantity, **kwargs)

    def trace(self, quantity="warps", component=0, **kwargs):
        return trace_particle(self._inner, quantity, component, **kwargs)

    def save(self, path):
        _core.save(path, self._inner)


class Validation:
    """Validation of DIC results against ground-truth Speckle warps."""

    def __init__(self, speckle, fields, labels):
        raw_speckle = getattr(speckle, '_inner', speckle)
        raw_fields  = [getattr(f, '_inner', f) for f in fields]
        self._inner = _core.Validation(raw_speckle, raw_fields, labels)
        self._solution = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def solve(self, cumulative=True, skim=None):
        self._solution = self._inner.solve(cumulative=cumulative, skim=skim)

    def standard_error(self, component, **kwargs):
        self._check_solved()
        return standard_error_validation(self._solution, component, **kwargs)

    def mean_error(self, component, **kwargs):
        self._check_solved()
        return mean_error_validation(self._solution, component, **kwargs)

    def noise_standard_error(self, component, **kwargs):
        self._check_solved()
        return noise_standard_error_validation(self._solution, component, **kwargs)

    def noise_mean_error(self, component, **kwargs):
        self._check_solved()
        return noise_mean_error_validation(self._solution, component, **kwargs)

    def strain_error(self, **kwargs):
        self._check_solved()
        return strain_error_validation(self._solution, **kwargs)

    def spatial_error(self, field_index, time_index, **kwargs):
        self._check_solved()
        return spatial_error_validation(self._solution, field_index, time_index, **kwargs)

    @property
    def solution(self):
        self._check_solved()
        return self._solution

    def n_frames(self, field_index=0):
        self._check_solved()
        import numpy as np
        return np.asarray(self._solution.fields[field_index].applied).shape[0]

    def n_particles(self, field_index=0):
        self._check_solved()
        import numpy as np
        return np.asarray(self._solution.fields[field_index].applied).shape[1]

    def _check_solved(self):
        if self._solution is None:
            raise RuntimeError("call solve() first")
