from cryptography_calculator.calculator import CryptographicCalculator


class TestExtendedEuclid:
    def test_extended_euclid_basic(self):
        # Testing basic case: gcd(30, 12)

        a, b = 30, 12
        gcd, x, y = CryptographicCalculator.extended_euclid(a, b)
        assert gcd == 6
        assert 30 * x + 12 * y == gcd

    def test_extended_euclid_same_numbers(self):
        # Testing when a and b are the same

        a, b = 20, 20
        gcd, x, y = CryptographicCalculator.extended_euclid(a, b)
        assert gcd == 20
        assert 20 * x + 20 * y == gcd

    def test_extended_euclid_one_is_zero(self):
        # Testing case when one of the numbers is zero

        a, b = 0, 15
        gcd, x, y = CryptographicCalculator.extended_euclid(a, b)
        assert gcd == 15
        assert x == 0
        assert y == 1

    def test_extended_euclid_negative_numbers(self):
        # Testing with negative numbers

        a, b = -30, 12
        gcd, x, y = CryptographicCalculator.extended_euclid(a, b)
        assert gcd == 6
        assert -30 * x + 12 * y == gcd or -30 * x + 12 * y == -gcd

    def test_extended_euclid_reverse_order(self):
        # Testing with reversed order of a and b

        a, b = 12, 30
        gcd, x, y = CryptographicCalculator.extended_euclid(a, b)
        assert gcd == 6
        assert 12 * x + 30 * y == gcd

    def test_extended_euclid_large_numbers(self):
        # Testing with large numbers

        a, b = 123456789, 987654321
        gcd, x, y = CryptographicCalculator.extended_euclid(a, b)
        assert gcd == 9
        assert 123456789 * x + 987654321 * y == gcd

    def test_extended_euclid_co_prime(self):
        # Testing when a and b are co-prime (gcd should be 1)

        a, b = 17, 31
        gcd, x, y = CryptographicCalculator.extended_euclid(a, b)
        assert gcd == 1
        assert 17 * x + 31 * y == gcd
