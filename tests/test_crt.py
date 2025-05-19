import pytest
from cryptography_calculator.calculator import CryptographicCalculator


class TestChineseRemainderTheorem:
    """Test suite for the Chinese Remainder Theorem function."""

    @pytest.mark.parametrize(
        "remainders, moduli, expected",
        [
            ([2, 3, 2], [3, 5, 7], 23),  # x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) → x = 23
            ([1, 4, 6], [3, 5, 7], 34),  # x ≡ 1 (mod 3), x ≡ 4 (mod 5), x ≡ 6 (mod 7) → x = 34
            ([0, 3, 4], [5, 7, 9], 220),  # x ≡ 0 (mod 5), x ≡ 3 (mod 7), x ≡ 4 (mod 9) → x = 220
            ([1, 3, 3], [3, 5, 7], 73),  # x ≡ 1 (mod 3), x ≡ 3 (mod 5), x ≡ 3 (mod 7) → x = 73
        ],
    )
    def test_chinese_remainder_theorem(self, remainders, moduli, expected):
        """Test valid cases for Chinese Remainder Theorem."""

        assert CryptographicCalculator.chinese_remainder_theorem(remainders, moduli) == expected

    def test_chinese_remainder_theorem_invalid(self):
        """Test CRT with mismatched list lengths, which should raise a ValueError."""

        with pytest.raises(ValueError, match="Lists 'a' and 'b' must have the same length"):
            CryptographicCalculator.chinese_remainder_theorem([1, 2], [3])
