import numpy as np

"""Tests for the reduced wave equation operator (BTTB by IC row treatment)"""


def test_reduced_consistent_with_matvec(wave_eq):
    """Acertain that reduced operator is equivalent (minus ICs) to full"""
    ntm2, ns = wave_eq.nt - 2, wave_eq.nx * wave_eq.ny

    # create random U
    rng = np.random.default_rng(0)
    w = rng.standard_normal(ntm2 * ns)
    U = np.zeros((wave_eq.nt, ns))
    U[2:] = w.reshape(ntm2, ns)

    # apply full and reduced ops
    full = wave_eq.matvec(U.ravel()).reshape(wave_eq.nt, ns)
    red = (wave_eq.reduced_operator() @ w).reshape(ntm2, ns)

    # asset close on inner
    assert np.allclose(full[2 : wave_eq.nt - 1], red[1:], rtol=1e-13, atol=1e-14)


def test_blocks(wave_eq):
    ntm2, ns = wave_eq.nt - 2, wave_eq.nx * wave_eq.ny
    blocks = wave_eq.reduced_blocks

    rng = np.random.default_rng(0)
    slice_idx, wq = ntm2 // 2, rng.standard_normal(ns)

    # make probe vector
    wp = np.zeros((ntm2, ns))
    wp[slice_idx] = wq

    # reduced operator applied to probe
    out = (wave_eq.reduced_operator() @ wp.ravel()).reshape(ntm2, ns)

    # assert block[l] (l=0, 1, 2) is equivalent to reduced operator at same time step
    for pert in range(3):
        assert np.allclose(
            out[slice_idx + pert], blocks[pert] @ wq, rtol=1e-12, atol=1e-13
        )

    # check all slices other than q, q+1, q+2 are zero
    quiet = np.delete(np.arange(ntm2), [slice_idx, slice_idx + 1, slice_idx + 2])
    assert np.abs(out[quiet]).max() == 0
