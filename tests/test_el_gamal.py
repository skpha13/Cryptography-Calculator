import pytest
from cryptography_calculator.calculator import CryptographicCalculator


class TestElGamal:
    @pytest.mark.parametrize(
        "p, g, k, y, m, operation, expected",
        [
            (11, 2, 9, 7, 8, "multiplicative", (6, (7, 9))),
            (100, 31, 17, 11, 72, "additive", (27, (41, 69))),
            (100, 11, 12, 13, 14, "additive", (32, (43, 30))),
        ],
    )
    def test_el_gamal(self, p, g, k, y, m, operation, expected):
        assert CryptographicCalculator.el_gamal(p, g, k, y, m, operation) == expected
