import pytest
from cryptography_calculator.calculator import CryptographicCalculator


class TestMultipartyComputation:
    @pytest.mark.parametrize(
        "sharing_initial, sharing_multiply, modulo, operations, expected",
        [
            (
                [[3, 1], [4, 2], [5, 3]],
                [[0, 4], [0, 5], [0, 6]],
                1_000_000,
                [("mul", "x1", "x2"), ("add", "temp0", "x3")],
                17,
            ),
            (
                [[6, 3], [11, 5], [13, 9]],
                [[0, 1], [0, 3], [6, 6]],
                1_000_000,
                [("add", "x1", "x3"), ("mul", "temp0", "x2")],
                209,
            ),
        ],
    )
    def test_multiparty_computation(self, sharing_initial, sharing_multiply, modulo, operations, expected):
        result = CryptographicCalculator.multiparty_computation(sharing_initial, sharing_multiply, modulo, operations)
        assert result == expected
