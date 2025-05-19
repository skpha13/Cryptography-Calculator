import pytest
from cryptography_calculator.calculator import CryptographicCalculator


class TestRsa:
    @pytest.mark.parametrize(
        "N, e, m, modified, decrypt, expected",
        [
            (119, 5, 11, True, False, (44, 29, 11)),
            (85, 3, 80, False, False, (45, 43, 80)),
            (33, 7, 2, False, False, (29, 3, 2)),
            (133, 29, 99, False, False, (92, 41, 99)),
            (85, 9, 10, False, True, (10, 57, 75)),
            (85, 9, 10, True, True, (10, 9, 75)),
            (91, 5, 25, True, True, (25, 5, 51)),
        ],
    )
    def test_rsa(self, N, e, m, modified, expected, decrypt):
        assert CryptographicCalculator.rsa(N, e, m, modified=modified, decrypt=decrypt) == expected
