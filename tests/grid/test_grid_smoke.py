import pytest

from odil_wave import HomogeneousModel, OverDensityModel, SheppLoganModel

MODEL_FACTORIES = [
    lambda g: HomogeneousModel(g, background_c=1.5),
    lambda g: OverDensityModel(g, centre=(0.0, 0.0), radius=0.3),
    lambda g: SheppLoganModel(g, interior_fill=0.7),
]


@pytest.mark.parametrize("make", MODEL_FACTORIES)
def test_model_instantiates(grid, make):
    m = make(grid)
    assert m.c.shape == grid.shape
