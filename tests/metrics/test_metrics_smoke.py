import numpy as np
import pytest

from odil_wave.metrics import (
    relative_l2,
    relative_linfty,
    final_time_l2,
    l2_error_history,
    residual,
    residual_l2,
    residual_linfty,
    relative_pde_residual,
    trace_misfit,
    trace_misfit_norm,
    ErrorReport,
)


@pytest.fixture
def ref_field(grid):
    rng = np.random.default_rng(0)
    return rng.standard_normal((grid.nt, grid.nx * grid.ny))


# ====== relative field metrics ======


def test_relative_l2_zero_for_identical(ref_field):
    assert relative_l2(ref_field, ref_field) == pytest.approx(0.0, abs=1e-12)


def test_relative_l2_unit_for_double(ref_field):
    assert relative_l2(2 * ref_field, ref_field) == pytest.approx(1.0)


def test_relative_linfty_zero_and_unit(ref_field):
    assert relative_linfty(ref_field, ref_field) == pytest.approx(0.0, abs=1e-12)
    assert relative_linfty(2 * ref_field, ref_field) == pytest.approx(1.0)


def test_final_time_l2_zero_for_identical(ref_field):
    assert final_time_l2(ref_field, ref_field) == pytest.approx(0.0, abs=1e-12)


def test_l2_error_history_shape_and_zero(ref_field, grid):
    hist = l2_error_history(ref_field, ref_field)
    assert hist.shape == (grid.nt,)
    assert np.allclose(hist, 0.0)


def test_shape_mismatch_raises(ref_field):
    with pytest.raises(AssertionError):
        relative_l2(ref_field[:, :-1], ref_field)


def test_accepts_wavefield(wavefield, ref_field):
    wavefield.U = ref_field
    assert relative_l2(wavefield, wavefield) == pytest.approx(0.0, abs=1e-12)


# ====== residual metrics ======


def test_residual_shape(wave_eq, ref_field, sources, grid):
    r = residual(wave_eq, ref_field, sources)
    assert r.shape == (grid.nt * grid.nx * grid.ny,)


def test_residual_l2_matches_norm(wave_eq, ref_field, sources):
    r = residual(wave_eq, ref_field, sources)
    assert residual_l2(wave_eq, ref_field, sources) == pytest.approx(np.linalg.norm(r))


def test_residual_linfty_matches_max(wave_eq, ref_field, sources):
    r = residual(wave_eq, ref_field, sources)
    assert residual_linfty(wave_eq, ref_field, sources) == pytest.approx(
        np.abs(r).max()
    )


def test_relative_pde_residual_finite(wave_eq, ref_field, sources):
    val = relative_pde_residual(wave_eq, ref_field, sources)
    assert np.isfinite(val) and val >= 0.0


# ====== trace metrics ======


def test_trace_misfit_zero_for_identical(ref_field, receivers, grid):
    dm = trace_misfit(ref_field, ref_field, receivers)
    assert dm.ndim == 2 and dm.shape[0] == grid.nt
    assert np.allclose(dm, 0.0)


def test_trace_misfit_norm_zero_for_identical(ref_field, receivers):
    assert trace_misfit_norm(ref_field, ref_field, receivers) == pytest.approx(
        0.0, abs=1e-12
    )


# ====== ErrorReport aggregator ======


def test_error_report_full(wave_eq, ref_field, sources, receivers):
    rep = ErrorReport(
        u=ref_field, u_ref=ref_field, A=wave_eq, sources=sources, receivers=receivers
    ).generate()
    assert {"relative_l2_norm", "residual_l2_norm", "trace_misfit"} <= rep.keys()


def test_error_report_requires_something(ref_field):
    with pytest.raises(ValueError):
        ErrorReport(u=ref_field)


def test_error_report_warns_on_partial(ref_field):
    with pytest.warns(UserWarning):
        ErrorReport(u=ref_field, u_ref=ref_field)  # relative only; skips residual+trace
