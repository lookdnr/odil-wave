from odil_wave.optimisation.precond import AlphaCirculantPreconditioner

import pytest


@pytest.mark.parametrize("alpha", [1.0, 0.0, -1.0])
def test_invalid_alpha_raises(wave_eq, alpha):
    with pytest.raises(ValueError, match="alpha"):
        AlphaCirculantPreconditioner.from_wave_equation(wave_eq, alpha)
