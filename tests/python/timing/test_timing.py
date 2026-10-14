"""Image timing (`time=`, file-name timestamps) and the rate/leapfrog/sequential
plot options built on it."""

import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import geopyv_dev as gp
from geopyv_dev import Field, Image, Mask, Mesh, PathRegion, Sequence, SequenceOptions, Subset
from geopyv_dev.plots import contour_field, contour_mesh, contour_sequence, history_particle, trace_field

_HERE = os.path.dirname(__file__)
REF_IMG = os.path.abspath(os.path.join(_HERE, "../../../../geopyv/tests/ref.jpg"))
TAR_IMG = os.path.abspath(os.path.join(_HERE, "../../../../geopyv/tests/tar.jpg"))
IMAGES_AVAILABLE = os.path.isfile(REF_IMG) and os.path.isfile(TAR_IMG)
pytestmark = pytest.mark.skipif(not IMAGES_AVAILABLE, reason="test images not found")

CX, CY, HALF = 500.0, 500.0, 100.0
BOUNDARY = np.array([[CX - HALF, CY - HALF], [CX + HALF, CY - HALF],
                     [CX + HALF, CY + HALF], [CX - HALF, CY + HALF]])
# Out of order on disk-name sort by last number (000 < 500 < 000) -- the
# timestamps put them at 0, 0.5 and 2.0 s.
NAMES = ["img_20261014_120000_000.jpg", "img_20261014_120000_500.jpg", "img_20261014_120002_000.jpg"]


def _solve(seq):
    seq.solve(local_mask=Mask(mask_type="local", shape="circle", size=10),
              seed_coord=[CX, CY], max_norm=1e-3, max_iterations=100, subset_order=1,
              tolerance=0.0, options=SequenceOptions(guide=False), border=20)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("timed")
    for name, src in zip(NAMES, [REF_IMG, TAR_IMG, TAR_IMG]):
        shutil.copy(src, d / name)
    return d


@pytest.fixture(scope="module")
def timed_sequence(image_dir):
    """ref -> tar -> tar (the last two identical) against a fixed reference,
    timed from the file names."""
    seq = Sequence(image_dir=str(image_dir), boundary=PathRegion(BOUNDARY), target_nodes=15,
                   size=(10.0, 100.0), mesh_order=1)
    _solve(seq)
    return seq


@pytest.fixture(scope="module")
def timed_field(timed_sequence):
    field = Field(timed_sequence)
    field.solve()
    return field


# ---------------------------------------------------------------------------
# time= input
# ---------------------------------------------------------------------------

class TestTimeInput:
    def test_sequence_reads_and_orders_by_file_name_timestamps(self, image_dir, capsys):
        seq = Sequence(image_dir=str(image_dir), boundary=BOUNDARY, target_nodes=15,
                       size=(10.0, 100.0), mesh_order=1)
        assert [os.path.basename(p) for p in seq.image_paths] == NAMES
        np.testing.assert_allclose(seq.time, [0.0, 0.5, 2.0])
        assert seq.time_source == "filename (date-time)"
        assert "date-time" in capsys.readouterr().out

    def test_sequence_scalar_and_array_time(self, image_dir):
        seq = Sequence(image_dir=str(image_dir), boundary=BOUNDARY, target_nodes=15,
                       size=(10.0, 100.0), time=0.1)
        np.testing.assert_allclose(seq.time, [0.0, 0.1, 0.2])
        assert seq.time_source == "given"
        seq = Sequence(image_dir=str(image_dir), boundary=BOUNDARY, target_nodes=15,
                       size=(10.0, 100.0), time=[0.0, 1.0, 3.0])
        np.testing.assert_allclose(seq.time, [0.0, 1.0, 3.0])

    @pytest.mark.parametrize("bad", [[0.0, 1.0], 0.0, -1.0, True, "1s"])
    def test_sequence_bad_time_raises(self, image_dir, bad):
        # Core validation errors surface as RuntimeError (package-wide), bad
        # types as TypeError.
        with pytest.raises((RuntimeError, TypeError)):
            Sequence(image_dir=str(image_dir), boundary=BOUNDARY, target_nodes=15,
                     size=(10.0, 100.0), time=bad)

    def test_frame_counter_names_give_no_time(self, tmp_path):
        shutil.copy(REF_IMG, tmp_path / "frame_000001.jpg")
        shutil.copy(TAR_IMG, tmp_path / "frame_000002.jpg")
        seq = Sequence(image_dir=str(tmp_path), boundary=BOUNDARY, target_nodes=15, size=(10.0, 100.0))
        assert seq.time is None
        assert seq.time_source == "none"

    def test_mesh_and_subset_time(self, image_dir):
        f = Image(filepath=str(image_dir / NAMES[0]))
        g = Image(filepath=str(image_dir / NAMES[2]))
        mesh = Mesh(BOUNDARY, 15, f, g, size=(10.0, 100.0))
        assert mesh.time == (0.0, 2.0)
        assert mesh.dt == 2.0
        assert Mesh(BOUNDARY, 15, f, g, size=(10.0, 100.0), time=0.25).time == (0.0, 0.25)
        assert Mesh(BOUNDARY, 15, f, g, size=(10.0, 100.0), time=(1.0, 1.5)).dt == 0.5
        with pytest.raises(RuntimeError, match="one value per image"):
            Mesh(BOUNDARY, 15, f, g, size=(10.0, 100.0), time=[1.0, 2.0, 3.0])
        subset = Subset([CX, CY], Mask(mask_type="local", shape="circle", size=10), f, g)
        assert subset.time == (0.0, 2.0)
        assert subset.time_source == "filename (date-time)"


# ---------------------------------------------------------------------------
# Solutions carry time
# ---------------------------------------------------------------------------

class TestSolvedTime:
    def test_pairs_carry_reference_and_target_times(self, timed_sequence):
        assert timed_sequence.reference_updates == [False, False]
        assert timed_sequence.mesh_solution_at(0).time == (0.0, 0.5)
        assert timed_sequence.mesh_solution_at(1).time == (0.0, 2.0)

    def test_field_rows_carry_image_times(self, timed_field):
        np.testing.assert_allclose(timed_field.time, [0.0, 0.5, 2.0])
        np.testing.assert_allclose(timed_field.particle_at(0).time, [0.0, 0.5, 2.0])

    def test_time_survives_save_load(self, timed_sequence, timed_field, tmp_path):
        timed_sequence.save(str(tmp_path / "seq.pyv"))
        seq = gp.load(str(tmp_path / "seq.pyv"))
        np.testing.assert_allclose(seq.time, [0.0, 0.5, 2.0])
        assert seq.time_source == "filename (date-time)"
        timed_field.save(str(tmp_path / "field.pyv"))
        np.testing.assert_allclose(gp.load(str(tmp_path / "field.pyv")).time, [0.0, 0.5, 2.0])


# ---------------------------------------------------------------------------
# Leapfrog / sequential increments and rates
# ---------------------------------------------------------------------------

class TestIncrements:
    def test_sequence_increment_displacements(self, timed_sequence):
        u1 = np.asarray(timed_sequence.mesh_solution_at(1).displacements)
        leap, dt = timed_sequence.increment_displacements(1, "leapfrog")
        np.testing.assert_array_equal(np.asarray(leap), u1)
        assert dt == 2.0
        seq_inc, dt = timed_sequence.increment_displacements(1, "sequential")
        u0 = np.asarray(timed_sequence.mesh_solution_at(0).displacements)
        np.testing.assert_allclose(np.asarray(seq_inc), u1 - u0, atol=1e-12)
        assert dt == 1.5
        # Identical frames 1 and 2: no movement between them (to within the
        # two solves' DIC noise -- a weak boundary subset can differ by more).
        assert np.median(np.abs(np.asarray(seq_inc))) < 0.05
        with pytest.raises(ValueError):
            timed_sequence.increment_displacements(0, "bogus")

    def test_field_increment_modes_and_rates(self, timed_field):
        u = np.array([np.asarray(p.warps)[:, 0] for p in timed_field.particles])
        seq = np.asarray(timed_field.contour_values("u", increment=2, mode="sequential"))
        leap = np.asarray(timed_field.contour_values("u", increment=2, mode="leapfrog"))
        np.testing.assert_allclose(seq, u[:, 2] - u[:, 1], atol=1e-12)
        np.testing.assert_allclose(leap, u[:, 2] - u[:, 0], atol=1e-12)
        rate = np.asarray(timed_field.contour_values("u", increment=2, mode="sequential", rate=True))
        np.testing.assert_allclose(rate, (u[:, 2] - u[:, 1]) / 1.5, atol=1e-12)
        total = np.asarray(timed_field.contour_values("u", rate=True))
        np.testing.assert_allclose(total, (u[:, 2] - u[:, 0]) / 2.0, atol=1e-12)
        assert timed_field.contour_range(increment=2, mode="sequential") == (1, 3)

    def test_principal_rate_is_of_the_rate_tensor(self, timed_field):
        p = timed_field.particle_at(0)
        d = np.asarray(p.strains)[2] - np.asarray(p.strains)[0]
        mean, radius = (d[0] + d[1]) / 2, np.hypot((d[0] - d[1]) / 2, d[5])
        got = np.asarray(timed_field.contour_values("ep1", rate=True))[0]
        np.testing.assert_allclose(got, (mean + radius) / 2.0, rtol=1e-10, atol=1e-15)

    def test_series_increments(self, timed_field):
        p = timed_field.particle_at(0)
        u = np.asarray(p.warps)[:, 0]
        np.testing.assert_allclose(p.series_increments("warps", 0, "sequential", False), np.diff(u))
        np.testing.assert_allclose(p.series_increments("warps", 0, "leapfrog", True),
                                   [(u[1] - u[0]) / 0.5, (u[2] - u[0]) / 2.0])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

class TestPlots:
    def test_contour_mesh_and_sequence_velocity(self, timed_sequence):
        fig, _ = contour_sequence(timed_sequence, 1, "u", mode="sequential", rate=True, show=False)
        assert "px/s" in fig.axes[-1].get_ylabel()
        fig, _ = contour_mesh(timed_sequence.mesh_solution_at(1), "R", rate=True, show=False)
        assert "px/s" in fig.axes[-1].get_ylabel()
        with pytest.raises(ValueError, match="rate only applies"):
            contour_mesh(timed_sequence.mesh_solution_at(1), "C_ZNCC", rate=True, show=False)

    def test_contour_field_labels(self, timed_field):
        fig, _ = contour_field(timed_field, "ep_xx", increment=2, mode="sequential", rate=True, show=False)
        assert fig.axes[-1].get_ylabel() == "ep_xx (/s)"
        fig, _ = contour_field(timed_field, "u", rate=True, show=False)
        assert fig.axes[-1].get_ylabel() == "u (px/s)"
        with pytest.raises(ValueError, match="not both"):
            contour_field(timed_field, "u", window=(0, 2), increment=1, show=False)

    def test_history_uses_time_axis(self, timed_field):
        p = timed_field.particle_at(0)
        _, ax = history_particle(p, "warps", components=[0], show=False)
        assert "Time" in ax.get_xlabel()
        np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.0, 0.5, 2.0])
        _, ax = history_particle(p, "strains", rate=True, mode="sequential", show=False)
        assert "Time" in ax.get_xlabel()

    @pytest.mark.parametrize("mode,rate", [("leapfrog", False), ("sequential", True), ("time", False)])
    def test_trace_modes(self, timed_field, mode, rate):
        fig, _ = trace_field(timed_field, "warps", 0, mode=mode, rate=rate, show=False)
        label = fig.axes[-1].get_ylabel()
        if mode == "time":
            assert "Time" in label
        elif rate:
            assert "px/s" in label

    def test_trace_rate_with_time_mode_raises(self, timed_field):
        with pytest.raises(ValueError):
            trace_field(timed_field, "warps", 0, mode="time", rate=True, show=False)
