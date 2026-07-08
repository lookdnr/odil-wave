from odil_wave import AcquisitionGeometry
from odil_wave.geometry import Sources, Receivers


def test_sources_custom_instantiates(sources):
    assert sources.src_xy.shape == (sources.n_sources, 2)
    assert sources.source_matrix().shape[1] == sources.n_sources


def test_sources_ring_instantiates(grid):
    s = Sources(grid, n_sources=4, mode="ring")
    assert s.src_ij.shape == (4, 2)


def test_receivers_ring_instantiates(receivers):
    assert receivers.recv_ij.shape == (receivers.n_receivers, 2)


def test_receivers_custom_instantiates(grid):
    r = Receivers(grid, mode="custom", receiver_locs=((0.0, 0.0), (0.1, 0.1)))
    assert r.recv_ij.shape == (2, 2)


def test_acquisition_geometry_instantiates(geometry):
    assert isinstance(geometry, AcquisitionGeometry)
    assert geometry.source_matrix().shape[1] == geometry.sources.n_sources
