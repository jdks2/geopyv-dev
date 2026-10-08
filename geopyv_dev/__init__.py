from functools import cached_property
import warnings

import numpy as np

from ._geopyv_dev import *
import geopyv_dev._geopyv_dev as _core

from .calibration import Calibration, CalibrationParams

from .plots import (
    inspect_subset,
    inspect_mesh,
    inspect_sequence,
    inspect_particle,
    inspect_field,
    inspect_calibration,
    visualise_calibration,
    contour_calibration,
    error_calibration,
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
    if isinstance(raw, _core.CalibrationSolution):
        return Calibration._from_solution(raw)
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
# `solver` / `solver_options` -- see geopyv_dev_fresh/solver_options_restructure.md
#
# `solver=` names the per-node numerical kernel ("icgn"/"fagn") on all three
# of Subset.solve/Mesh.solve/Sequence.solve, replacing the previously
# inconsistent algorithm=/method= names. `solver_options` (Mesh/Sequence
# only) is a plain dict over three orthogonal pipeline axes -- topology,
# masking, preconditioning -- each optionally taking its own nested
# sub-dict of tunables. Validated in full, before any solve work is
# entered, so a mistyped or not-yet-implemented option never silently
# no-ops and never panics inside Rust (contrast py_sequence.rs's own
# `method` string dispatch, which silently maps any unrecognised value to
# SolveMethod::Icgn -- the wart this restructure exists to not repeat one
# layer up).
# ---------------------------------------------------------------------------

_SOLVER_OPTION_AXES = {
    "topology": {"uniform", "adaptive"},
    "masking": {"uniform", "zonal"},
    "preconditioning": {"RG", "layer-RG"},
}
_SOLVER_OPTION_DEFAULTS = {"topology": "uniform", "masking": "uniform", "preconditioning": "RG"}
_SOLVER_OPTION_SUBDICTS = {"zonal", "layer_rg"}
# (axis, value) -> (subdict key, subdict allowed keys) -- which sub-dict, if
# any, is legal only when that axis is at that non-default value.
_ZONAL_KEYS = {"k", "smoothing_sigma", "meshless_params", "iterations", "zone_map"}
# Sequence.solve: no zone_map -- one caller-supplied map has no sensible
# meaning across a sequence of increments.
_ZONAL_KEYS_SEQUENCE = _ZONAL_KEYS - {"zone_map"}
_SUBDICT_FOR_AXIS_VALUE = {
    ("masking", "zonal"): ("zonal", _ZONAL_KEYS),
    ("preconditioning", "layer-RG"): ("layer_rg", {"max_workers", "batch_factor", "root_rel_eps"}),
}
# (axis, value) accepted by the schema (_SOLVER_OPTION_AXES) but not yet
# implemented anywhere -- rejected here, before _inner.solve() is ever
# entered, so a value is never simultaneously "documented as valid" and
# "silently broken." Shrinks by one entry each time the corresponding plan
# lands (see solver_options_restructure.md §7) -- masking="zonal" already
# dropped out (Stage B, src/mesh.rs::solve_zonal_masking_impl).
_NOT_YET_IMPLEMENTED = {
    ("topology", "adaptive"),        # geopyv_dev_fresh/to_do.md Step 3
    # preconditioning="layer-RG": landed (src/mesh.rs::expand_parallel /
    # Preconditioning::LayerRg, geopyv_dev_fresh/layer_rg_plan.md §3).
}


def _validate_solver_options(solver_options, zonal_keys=_ZONAL_KEYS):
    """Validate + normalise a `solver_options` dict per
    solver_options_restructure.md §6. Returns
    (topology, masking, preconditioning, zonal_opts, layer_rg_opts) --
    every axis filled with its default if absent, every sub-dict {} if
    absent. Raises ValueError for any unrecognised key/value or a sub-dict
    supplied for an axis not at the value it belongs to; NotImplementedError
    for a schema-valid but not-yet-implemented (axis, value)."""
    if solver_options is None:
        solver_options = {}
    if not isinstance(solver_options, dict):
        raise TypeError(f"solver_options must be a dict, got {type(solver_options).__name__}")

    allowed_top = set(_SOLVER_OPTION_AXES) | _SOLVER_OPTION_SUBDICTS
    unknown_top = set(solver_options) - allowed_top
    if unknown_top:
        raise ValueError(
            f"unknown solver_options key(s) {sorted(unknown_top)!r}; valid keys are "
            f"{sorted(allowed_top)!r}"
        )

    axis_values = {}
    for axis, allowed in _SOLVER_OPTION_AXES.items():
        value = solver_options.get(axis, _SOLVER_OPTION_DEFAULTS[axis])
        if value not in allowed:
            raise ValueError(
                f"solver_options[{axis!r}] must be one of {sorted(allowed)!r}, got {value!r}"
            )
        axis_values[axis] = value

    sub_opts = {"zonal": {}, "layer_rg": {}}
    for (axis, active_value), (sub_key, sub_allowed_keys) in _SUBDICT_FOR_AXIS_VALUE.items():
        present = sub_key in solver_options
        active = axis_values[axis] == active_value
        if present and not active:
            raise ValueError(
                f"solver_options[{sub_key!r}] was given but solver_options[{axis!r}] "
                f"is {axis_values[axis]!r} (needs to be {active_value!r} for it to apply)"
            )
        if sub_key == "zonal":
            sub_allowed_keys = zonal_keys
        sub = dict(solver_options.get(sub_key, {}))
        unknown_sub = set(sub) - sub_allowed_keys
        if unknown_sub:
            raise ValueError(
                f"unknown solver_options[{sub_key!r}] key(s) {sorted(unknown_sub)!r}; "
                f"valid keys are {sorted(sub_allowed_keys)!r}"
            )
        sub_opts[sub_key] = sub

    for axis, value in axis_values.items():
        if (axis, value) in _NOT_YET_IMPLEMENTED:
            raise NotImplementedError(
                f"solver_options[{axis!r}]={value!r} is not implemented yet -- see "
                "geopyv_dev_fresh/solver_options_restructure.md §7 for what's landed."
            )

    it = sub_opts["zonal"].get("iterations", 1)
    if not (isinstance(it, int) and not isinstance(it, bool) and it >= 1):
        raise ValueError(
            f"solver_options['zonal']['iterations'] must be an int >= 1, got {it!r}"
        )

    return (axis_values["topology"], axis_values["masking"], axis_values["preconditioning"],
            sub_opts["zonal"], sub_opts["layer_rg"])


def _resolve_solver(explicit_solver, kwargs, deprecated_name):
    """Combine the new `solver=` parameter (`explicit_solver`, `None` if not
    given) with a possibly-present deprecated kwarg (`"method"` or
    `"algorithm"`) inside `kwargs` (popped if present). Emits
    DeprecationWarning for the old name; raises ValueError if both are given
    with conflicting values. Returns the resolved solver string, default
    `"icgn"` if neither given."""
    have_old = deprecated_name in kwargs
    old_val = kwargs.pop(deprecated_name, None)
    if have_old:
        warnings.warn(
            f"{deprecated_name}= is deprecated, use solver= instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    if explicit_solver is not None and have_old and explicit_solver != old_val:
        raise ValueError(
            f"both solver={explicit_solver!r} and {deprecated_name}={old_val!r} given "
            "with different values"
        )
    if explicit_solver is not None:
        return explicit_solver
    if have_old:
        return old_val
    return "icgn"


_SOLVERS = {"icgn", "fagn"}


def _checked_solver(solver):
    """Lower-case `solver` and check it is a known per-node kernel. An
    unknown value warns and falls back to ``"icgn"`` -- the same behaviour
    `Subset.solve` has always had, now shared by `Mesh.solve` and
    `Sequence.solve`."""
    algo = str(solver).lower()
    if algo not in _SOLVERS:
        warnings.warn(
            f"Unknown solver '{algo}'; falling back to 'icgn'.",
            UserWarning,
            stacklevel=3,
        )
        return "icgn"
    return algo


def _zonal_kwargs(zonal_opts):
    """`solver_options["zonal"]` -> the binding's flat ``zonal_*`` kwargs,
    passing only keys the caller actually gave (defaults live in Rust,
    `ZonalConfig::default()`)."""
    out = {}
    for key in ("k", "smoothing_sigma", "meshless_params", "iterations"):
        if zonal_opts.get(key) is not None:
            out[f"zonal_{key}"] = zonal_opts[key]
    if zonal_opts.get("zone_map") is not None:
        out["zonal_zone_map"] = np.ascontiguousarray(zonal_opts["zone_map"], dtype=np.uint8)
    return out


def _layer_rg_kwargs(layer_rg_opts):
    """`solver_options["layer_rg"]` -> the binding's flat ``layer_rg_*``
    kwargs (only read by Rust when ``preconditioning == "layer-RG"``)."""
    return {f"layer_rg_{key}": layer_rg_opts[key]
            for key in ("batch_factor", "root_rel_eps", "max_workers")
            if key in layer_rg_opts}


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

    def solve(self, p_0=None, solver=None,
              max_norm=1e-3, max_iterations=50, tolerance=0.75, **extra_kwargs):
        """Run the DIC solver. Mutates the object in place; returns None
        (read the result back via ``.p``, ``.c_zncc``, etc.).

        Parameters
        ----------
        p_0 : list[float], optional
            Initial warp vector. Defaults to zeros matching order.
        solver : str, optional
            "icgn" (default) or "fagn" -- the per-node numerical kernel.
            ``algorithm=`` is a deprecated alias for this parameter (see
            `geopyv_dev_fresh/solver_options_restructure.md`).
        max_norm : float, optional
            Convergence norm threshold. Default 1e-3.
        max_iterations : int, optional
            Iteration limit. Default 50.
        tolerance : float, optional
            Minimum acceptable C_ZNCC for solved=True. Default 0.75.
        """
        algo = _checked_solver(_resolve_solver(solver, extra_kwargs, "algorithm"))
        kwargs = dict(p_0=p_0, max_norm=max_norm,
                      max_iterations=max_iterations, tolerance=tolerance)
        if algo == "fagn":
            self._inner.solve_fagn(**kwargs)
        else:
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

    _ADAPTIVE_KWARG_PREFIX = "adaptive_"
    # Old flat adaptive_* kwarg -> new solver_options["zonal"] key, for the
    # method="adaptive" deprecation shim (solver_options_restructure.md §5).
    # adaptive_min_pixel_fraction deliberately has no entry -- see below.
    _ADAPTIVE_KWARG_MAP = {
        "adaptive_k": "k",
        "adaptive_meshless_params": "meshless_params",
    }

    def solve(self, local_mask, seed_coord, seed_warp=None, solver=None,
              solver_options=None, **kwargs):
        """Run the DIC solver. Mutates in place; returns None.

        solver : str, optional
            "icgn" (default) or "fagn" -- the per-node numerical kernel
            ONLY. An unknown value warns and falls back to "icgn" (as on
            `Subset.solve`). ``method=`` is a deprecated alias for this
            parameter (not for ``solver_options`` -- see below). Never means
            "adaptive".

        solver_options : dict, optional
            Three orthogonal pipeline axes, each defaulting to its first
            (today's-behaviour) value: ``{"topology": "uniform", "masking":
            "uniform", "preconditioning": "RG"}``. Note this configures the
            *pipeline*, not the numerical kernel -- which subset solver
            runs per node is ``solver=`` alone, never a key in this dict.
            See `geopyv_dev_fresh/solver_options_restructure.md` for the
            full design.

            - ``topology`` : ``"uniform"`` (default) or ``"adaptive"``
              (mesh-relocation pipeline -- not yet implemented, raises
              ``NotImplementedError``; see `geopyv_dev_fresh/to_do.md` Step 3).
            - ``masking`` : ``"uniform"`` (default) or ``"zonal"`` -- a
              two-pass pipeline, implemented entirely in Rust core
              (``Mesh::solve_zonal_masking_impl``, ``src/mesh.rs``) --
              this Python layer only translates kwargs, it does not
              orchestrate: (1) an ordinary ``solver="icgn"`` solve; (2) a
              ``Field`` built at the mesh's own node positions, solved
              meshlessly (Rust-to-Rust, no PyO3 round-trip), for each
              node's shear strain ``gamma_max``; (3) rasterise ``gamma_max``
              to a dense image (meshless nearest-node), Gaussian-smooth it,
              take its Sobel magnitude ``|∇γ|``; the connected low-``|∇γ|``
              regions are the deformation regimes and the ``|∇γ|`` ridges
              are the interfaces -- a marker-controlled watershed of
              ``|∇γ|`` from the regime cores lays the zone boundary on the
              ridge crest (``src/geometry/rasterize.rs``); (4) a second
              ``solver="icgn"`` solve, same node positions and seed, with
              that zone label applied internally as a per-subset exclusion
              mask. ``self`` ends up holding pass 2's solution (or pass
              1's, if pass 2 fails -- the masking refinement never makes
              an already-solvable mesh unsolvable). `Sequence.solve` gets
              this "for free" too -- its per-pair loop calls the same
              ``Mesh::solve`` unchanged. Sub-options via
              ``solver_options["zonal"]`` (only read when
              ``masking == "zonal"``):

              - ``k`` (float, default 2.0) -- robust-threshold multiplier
                for the ``|∇γ|`` ridge cut-off
                (``median + k*1.4826*MAD`` over the smoothed ``|∇γ|`` image).
              - ``smoothing_sigma`` (float, default ``None``) -- Gaussian σ
                (pixels) applied to the dense ``gamma_max`` image before the
                Sobel step; ``None`` ⇒ the mesh's own node spacing.
              - ``meshless_params`` (MeshlessParams, default ``None``) --
                params for the Stage-2 field solve.
              - ``iterations`` (int, default 1) -- classifier refit passes;
                ``>1`` re-solves the Stage-2 field ``zone_aware`` seeded by
                the current map and reclassifies until the partition
                converges.
              - ``zone_map`` ((H, W) uint8 array, default ``None``) -- a
                caller-supplied whole-image zone-label grid. When given, the
                classifier is skipped entirely and this grid drives the
                per-subset masking directly (``iterations`` is then
                ignored). Use it to mask from a partition already known from
                specimen geometry rather than one detected from the strain
                field. Reserve label ``0`` for "no zone".

              (The pass-2 minimum-pixel-fraction guard is a fixed internal
              safety constant, not configurable here -- see
              `solver_options_restructure.md` §3.)
            - ``preconditioning`` : ``"RG"`` (default) or ``"layer-RG"``
              (layer-parallel RG frontier traversal; agrees with ``"RG"``
              within Tier C and is deterministic -- see
              `geopyv_dev_fresh/layer_rg_plan.md`). Sub-options via
              ``solver_options["layer_rg"]``: ``batch_factor``,
              ``root_rel_eps``, ``max_workers``.

        **Deprecated:** ``method="adaptive"`` (with ``adaptive_*`` kwargs)
        is a deprecated alias for ``solver_options={"masking": "zonal",
        "zonal": {...}}`` -- emits ``DeprecationWarning`` and forwards for
        one release; see `solver_options_restructure.md` §5's mapping
        table. ``adaptive_min_pixel_fraction`` specifically has no new
        home (it's no longer configurable at all) and raises ``ValueError``
        if passed, regardless of value."""
        if kwargs.get("method") == "adaptive":
            return self._solve_adaptive_deprecated_shim(
                local_mask, seed_coord, seed_warp, solver, solver_options, kwargs,
            )

        stray_adaptive = {k for k in kwargs if k.startswith(self._ADAPTIVE_KWARG_PREFIX)}
        if stray_adaptive:
            raise ValueError(
                f"{sorted(stray_adaptive)!r} are old adaptive_* kwargs (see "
                "solver_options_restructure.md §5) -- only meaningful together with "
                "the deprecated method='adaptive'. Use "
                "solver_options={'masking': 'zonal', 'zonal': {...}} instead."
            )

        resolved_solver = _checked_solver(_resolve_solver(solver, kwargs, "method"))
        topology, masking, preconditioning, zonal_opts, layer_rg_opts = \
            _validate_solver_options(solver_options)
        # topology has no implemented non-default value yet (rejected inside
        # _validate_solver_options) -- only "uniform" ever reaches here.
        self._inner.solve(
            local_mask, seed_coord, seed_warp=seed_warp,
            method=resolved_solver, preconditioning=preconditioning, masking=masking,
            **(_zonal_kwargs(zonal_opts) if masking == "zonal" else {}),
            **_layer_rg_kwargs(layer_rg_opts), **kwargs,
        )

    def _solve_adaptive_deprecated_shim(self, local_mask, seed_coord, seed_warp,
                                         solver, solver_options, kwargs):
        """`method="adaptive"` -> `solver_options={"masking": "zonal", ...}`
        forwarding shim -- solver_options_restructure.md §5."""
        warnings.warn(
            "method='adaptive' is deprecated, use solver_options={'masking': 'zonal', "
            "...} instead. See geopyv_dev_fresh/solver_options_restructure.md.",
            DeprecationWarning,
            stacklevel=3,
        )
        if solver_options is not None:
            raise ValueError(
                "solver_options= cannot be combined with the deprecated method='adaptive'."
            )
        kwargs.pop("method")

        present = {k for k in kwargs if k.startswith(self._ADAPTIVE_KWARG_PREFIX)}
        if "adaptive_min_pixel_fraction" in present:
            raise ValueError(
                "adaptive_min_pixel_fraction is no longer configurable (it is a fixed "
                "internal safety constant, not a tuning knob) -- remove this kwarg. "
                "See geopyv_dev_fresh/solver_options_restructure.md §5."
            )
        unrecognised = present - set(self._ADAPTIVE_KWARG_MAP)
        if unrecognised:
            raise ValueError(
                f"{sorted(unrecognised)!r} only apply to method='adaptive' "
                "(and are not recognised adaptive_* kwargs)."
            )
        zonal = {new_key: kwargs.pop(old_key) for old_key, new_key in self._ADAPTIVE_KWARG_MAP.items()
                 if old_key in kwargs}

        return self.solve(local_mask, seed_coord, seed_warp=seed_warp, solver=solver,
                           solver_options={"masking": "zonal", "zonal": zonal}, **kwargs)

    def save(self, path):
        """Save the solved mesh to a .pyv file."""
        self._inner.save(path)

    def inspect(self, **kwargs):
        """Render the mesh over its reference image.

        ``subset_idx=i`` highlights one node; when the solve recorded its
        subset template (schema ``0x07`` on) the plot becomes a crop of that
        node's template, and for a ``solver_options={"masking": "zonal"}``
        solve the zone-removed template pixels are shaded and the pre/post
        pixel counts, ``|∇γ|`` and any minimum-pixel-guard fallback are
        captioned. ``zones=True`` overlays the whole-image zone-label map
        from a zonal solve (raises if the mesh was not solved that way).
        ``show_areas=True`` draws node-area discs. See
        ``geopyv_dev.plots.inspect_mesh``.
        """
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

    def solve(self, *args, solver=None, solver_options=None, **kwargs):
        """Run the DIC solver over every image pair. Mutates in place;
        returns None.

        solver : str, optional
            "icgn" (default) or "fagn" -- the per-node numerical kernel,
            forwarded to every pair's `Mesh.solve`. An unknown value warns
            and falls back to "icgn". ``method=`` is a deprecated alias --
            see `geopyv_dev_fresh/solver_options_restructure.md`.
        solver_options : dict, optional
            Same three axes as `Mesh.solve` -- see that method's docstring
            and `geopyv_dev_fresh/solver_options_restructure.md`.
            ``solver_options={"masking": "zonal"}`` now works here too
            (Stage B, `src/mesh.rs::Mesh::solve_zonal_masking_impl`) --
            forwarded straight through to every pair's `Mesh.solve`
            unchanged; `Sequence::solve`'s own per-pair Rust loop needed no
            structural change to support it (it already called
            `mesh.solve` once per pair), so it composes correctly with the
            existing reference-update/override retry machinery. The
            ``"zonal"`` sub-dict takes the same keys as on `Mesh.solve`
            except ``zone_map`` (one caller-supplied map has no sensible
            meaning across a sequence of increments).

        **Deprecated:** ``method="adaptive"`` -- forwards to
        ``solver_options={"masking": "zonal"}`` with the usual
        ``DeprecationWarning``."""
        if kwargs.get("method") == "adaptive":
            warnings.warn(
                "method='adaptive' is deprecated, use solver_options={'masking': "
                "'zonal'} instead. See geopyv_dev_fresh/solver_options_restructure.md.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.pop("method")
            solver_options = solver_options if solver_options is not None else {}
            solver_options = {**solver_options, "masking": "zonal"}

        resolved_solver = _checked_solver(_resolve_solver(solver, kwargs, "method"))
        topology, masking, preconditioning, zonal_opts, layer_rg_opts = \
            _validate_solver_options(solver_options, zonal_keys=_ZONAL_KEYS_SEQUENCE)
        self._inner.solve(
            *args, method=resolved_solver, preconditioning=preconditioning, masking=masking,
            **(_zonal_kwargs(zonal_opts) if masking == "zonal" else {}),
            **_layer_rg_kwargs(layer_rg_opts), **kwargs,
        )

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

    def solve(self, factor=0.0, true_incs=True, calibration=None, strain_method=None):
        cal = getattr(calibration, '_inner', calibration)
        sm = getattr(strain_method, '_inner', strain_method)
        self._inner.solve(factor=factor, true_incs=true_incs, calibration=cal, strain_method=sm)

    @cached_property
    def particles(self):
        """Every particle's strain-path solution. Reconstructed once and cached
        — use ``particle_at(idx)`` instead if only one particle is needed."""
        return [ParticleWrapper(p) for p in self._inner.particles]

    def particle_at(self, idx):
        """A single particle's strain-path solution, without materialising the rest."""
        return ParticleWrapper(self._inner.particle_at(idx))

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

    def solve(self, factor=0.0, true_incs=True, calibration=None, strain_method=None):
        cal = getattr(calibration, '_inner', calibration)
        sm = getattr(strain_method, '_inner', strain_method)
        self._inner.solve(factor=factor, true_incs=true_incs, calibration=cal, strain_method=sm)

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
            raise RuntimeError(
                f"{type(self).__name__} has not been solved; call solve() first"
            )
