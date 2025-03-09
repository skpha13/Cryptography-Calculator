import pytest
from cryptography_calculator.calculator import CryptographicCalculator


class TestFastExponentiation:
    """Test suite for FastExponentiation class."""

    @pytest.mark.parametrize(
        "a, p, m, expected",
        [
            (2, 10, 1000, 24),  # 2^10 % 1000 = 1024 % 1000 = 24
            (5, 3, 13, 8),  # 5^3 % 13 = 125 % 13 = 8
            (7, 0, 9, 1),  # Any number^0 = 1
            (3, 4, 5, 1),  # 3^4 % 5 = 81 % 5 = 1
            (10, 100, 17, 4),  # 10^100 % 17
            (123, 456, 789, 699),  # Large numbers test
            (5, 117, 19, 1),
        ],
    )
    def test_fast_exponentiation(self, a, p, m, expected):
        """Tests the fast_exponentiation function with multiple test cases."""
        assert CryptographicCalculator.fast_exponentiation(a, p, m) == expected
