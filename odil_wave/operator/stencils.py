# hardcoded central difference stencil coefficients (1st derivative)
# e.g. order : ([numerator coeffs], denom coeff)
STENCIL_COEFFS_1ST = {
    2: ([-1.0, 0.0, 1.0], 2.0),
    4: ([1.0, -8.0, 0.0, 8.0, -1.0], 12.0),
    6: ([-1.0, 9.0, -45.0, 0, 45.0, -9.0, 1.0], 60.0),
    8: ([3.0, -32.0, 168.0, -672.0, 0.0, 672.0, -168.0, 32.0, -3.0], 840.0),
}

# hardcoded central difference stencil coefficients (2nd derivative)
# e.g. order : ([numerator coeffs], denom coeff)
STENCIL_COEFFS_2ND = {
    2: ([1.0, -2.0, 1.0], 1.0),
    4: ([-1.0, 16.0, -30.0, 16.0, -1.0], 12.0),
    6: ([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0], 180.0),
    8: ([-9.0, 128.0, -1008.0, 8064.0, -14350.0, 8064.0, -1008.0, 128.0, -9.0], 5040.0),
}

# global offsets for constructing matrix operators
# e.g., order : offsets
STENCIL_OFFSETS = {
    2: [-1, 0, 1],
    4: [-2, -1, 0, 1, 2],
    6: [-3, -2, -1, 0, 1, 2, 3],
    8: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
}
