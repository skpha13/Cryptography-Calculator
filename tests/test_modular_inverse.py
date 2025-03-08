import pytest
from cryptography_calculator.calculator import CryptographicCalculator


class TestModularInverse:
    """Test suite for the modular inverse function."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (3, 7, 5),  # 3 * 5 ≡ 1 (mod 7)
            (10, 17, 12),  # 10 * 12 ≡ 1 (mod 17)
            (7, 13, 2),  # 7 * 2 ≡ 1 (mod 13)
            (1, 2, 1),  # 1 is always its own modular inverse
        ],
    )
    def test_mod_inverse(self, a, b, expected):
        """Test valid cases for modular inverse."""

        assert CryptographicCalculator.modular_inverse(a, b) == expected

    def test_mod_inverse_invalid(self):
        """Test modular inverse for non-coprime numbers, which should raise a ValueError."""

        with pytest.raises(ValueError, match="Modular inverse does not exist"):
            CryptographicCalculator.modular_inverse(6, 9)  # gcd(6, 9) != 1, so no inverse
