import pytest
from cryptography_calculator.utils.primes import PrimeException, is_prime, two_prime_decomposition


class TestPrimes:
    """Unit tests for prime number utilities, including primality checks and two-prime decomposition."""

    @pytest.mark.parametrize(
        "n, expected",
        [
            (2, True),
            (3, True),
            (4, False),
            (5, True),
            (9, False),
            (11, True),
            (15, False),
            (1, False),
            (0, False),
            (-5, False),
        ],
    )
    def test_is_prime(self, n, expected):
        assert is_prime(n) == expected

    @pytest.mark.parametrize("n, expected", [(119, (7, 17)), (65, (5, 13)), (143, (11, 13))])
    def test_two_prime_decomposition(self, n, expected):
        pass

    @pytest.mark.parametrize("invalid_input", [1, -1, 4, 70, 17, 169])
    def test_two_prime_decomposition_invalid_inputs(self, invalid_input):
        with pytest.raises(PrimeException):
            two_prime_decomposition(invalid_input)
