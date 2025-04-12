from typing import List, Tuple


class PrimeException(Exception):
    """Exception raised for errors related to prime number operations.

    Parameters
    ----------
    message : str
        Explanation of the error.
    number : int
        The number that caused the exception.
    """

    def __init__(self, message, number):
        self.number = number
        super().__init__(f"{message} (N: {number})")


def is_prime(N: int) -> bool:
    if N < 2:
        return False

    for i in range(2, N):
        if i * i > N:
            return True

        if N % i == 0:
            return False

    return True


def two_prime_decomposition(N: int) -> Tuple[int, int]:
    """Decompose a number into two distinct prime factors.

    Parameters
    ----------
    N : int
        The number to decompose.

    Returns
    -------
    Tuple[int, int]
        A tuple containing two distinct prime factors of N.

    Raises
    ------
    PrimeException
        If N is less than 2, or if N is not a product of two distinct prime numbers.
    """

    if N < 2:
        raise PrimeException("N is lower than 2", N)

    for first_prime in range(2, N):
        if N % first_prime == 0:
            second_prime = N // first_prime

            if is_prime(first_prime) and is_prime(second_prime) and first_prime != second_prime:
                return first_prime, second_prime

            raise PrimeException("N is not a product of two primes", N)

    raise PrimeException("N is not a product of two primes", N)
