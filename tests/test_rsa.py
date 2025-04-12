import pytest
from cryptography_calculator.calculator import CryptographicCalculator


class TestRsa:
    @pytest.mark.parametrize(
        "N, e, m, modified, expected",
        [
            (119, 5, 11, True, (44, 29, 11)),
            (85, 3, 80, False, (45, 43, 80)),
            (33, 7, 2, False, (29, 3, 2)),
            (133, 29, 99, False, (92, 41, 99)),
        ],
    )
    def test_rsa(self, N, e, m, modified, expected):
        assert CryptographicCalculator.rsa(N, e, m, modified=modified) == expected
