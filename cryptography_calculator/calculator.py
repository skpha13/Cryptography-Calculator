import logging
from functools import reduce
from typing import List, Tuple

from cryptography_calculator.utils.logger import LogStack

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logstack = LogStack(logger=logger)


class CryptographicCalculator:
    @staticmethod
    def euclid(a: int, b: int) -> List[Tuple[int, int, int]]:
        """Computes the greatest common divisor (GCD) of two numbers using the Euclidean Algorithm.
        It also logs and returns the intermediate steps of the algorithm.

        The Euclidean Algorithm repeatedly divides `a` by `b`, replacing `a` with `b`
        and `b` with the remainder until `b` becomes zero.

        Parameters
        ----------
        a : int
            The first integer.
        b : int
            The second integer.

        Returns
        -------
        List[Tuple[int, int, int]]
            A list of tuples, where each tuple contains:
            - The dividend (`a`).
            - The divisor (`b`).
            - The remainder (`a % b`).
        """
        steps = []

        while b:
            remainder = a % b
            steps.append((a, b, remainder))

            logstack.add_message(f"{a} = {a // b} * {b} + {remainder}")

            a, b = b, remainder

        logstack.add_message()
        return steps

    @staticmethod
    def __log_extended_euclid(a: int, b: int, x: int, y: int) -> None:
        x_new, a_new = (abs(x), abs(a)) if x < 0 and a < 0 else (x, a)
        y_new, b_new = (abs(y), abs(b)) if y < 0 and b < 0 else (y, b)

        logstack.add_message(f"1 = {x_new} * {a_new} + {y_new} * {b_new}")

    @staticmethod
    def extended_euclid(a: int, b: int) -> Tuple[int, int, int]:
        """Computes the greatest common divisor (GCD) of two numbers using the Extended Euclidean Algorithm.
        Also finds integers x and y such that `a * x + b * y = gcd(a, b)`.
        It also logs the intermediate steps of the algorithm.

        Parameters
        ----------
        a : int
            First integer.
        b : int
            Second integer.
        first_call : bool, Optional
            Default value True.

        Returns
        -------
        Tuple[int, int, int]
            A tuple (gcd, x, y) where:
            - `gcd` is the greatest common divisor of `a` and `b`.
            - `x` and `y` satisfy the equation `a * x + b * y = gcd`.
        """

        if a == 0:
            return b, 0, 1

        gcd, x1, y1 = CryptographicCalculator.extended_euclid(b % a, a)
        x = y1 - (b // a) * x1
        y = x1

        CryptographicCalculator.__log_extended_euclid(a, b, x, y)

        return abs(gcd), x, y

    @staticmethod
    def modular_inverse(a: int, b: int) -> int:
        """Computes the modular inverse of `a` under modulo `b` using the Extended Euclidean Algorithm.

        Parameters
        ----------
        a : int
            The number for which to compute the modular inverse.
        b : int
            The modulus.

        Returns
        -------
        int
            The modular inverse of `a` modulo `b`, if it exists.

        Raises
        ------
        ValueError
            If the modular inverse does not exist (i.e., `a` and `b` are not coprime).
        """
        gcd, x, _ = CryptographicCalculator.extended_euclid(a, b)

        if gcd != 1:
            raise ValueError(f"Modular inverse does not exist for {a} mod {b}")

        logstack.add_message(f"\nFinal Result:\n\t inv({a}) mod {b} = {x % b}\n")
        return x % b

    @staticmethod
    def chinese_remainder_theorem(a: List[int], b: List[int]) -> int:
        """Solves a system of modular congruences using the Chinese Remainder Theorem (CRT).

        The system of equations is given by:
            x ≡ a_1 (mod b_1)
            x ≡ a_2 (mod b_2)
            ...
            x ≡ a_n (mod b_n)

        where the moduli `b` must be pairwise coprime.

        Parameters
        ----------
        a : List[int]
            List of remainders.
        b : List[int]
            List of moduli (must be pairwise coprime).

        Returns
        -------
        int
            The smallest non-negative solution `x` that satisfies all the congruences.

        Raises
        ------
        ValueError
            If the input lists `a` and `b` are not of the same length.
        """

        if len(a) != len(b):
            raise ValueError("Lists 'a' and 'b' must have the same length.")
        N = reduce(lambda x, y: x * y, b)

        logstack.add_message(f"N = {N}")

        x = 0
        index = 1
        for ai, bi in zip(a, b):
            Ni = N // bi

            logstack.add_message(f"\n\t\t\t Step: {index}")
            logstack.add_message(f"Modular Inverse for: a = {Ni}, b = {bi}:")
            logstack.add_message(f"\tSteps:")

            bi_inv = CryptographicCalculator.modular_inverse(Ni, bi)
            x += ai * Ni * bi_inv

            logstack.add_message(f"\nResult:\nai = {ai}, bi = {Ni}, bi_inv = {bi_inv}, x = {x}\n")

            index += 1

        logstack.add_message(f"Final Result: {x % N}")
        return x % N


if __name__ == "__main__":
    logstack.add_message(" Euclid for: a = 113, b = 15")
    CryptographicCalculator.euclid(113, 15)
    logstack.display_logs()
    logstack.empty_messages()

    logstack.add_message(" Extended Euclid for: a = 113, b = 15")
    CryptographicCalculator.extended_euclid(113, 15)
    logstack.add_message()
    logstack.display_logs()
    logstack.empty_messages()

    logstack.add_message(" Modular Inverse for: a = 9, b = 26")
    CryptographicCalculator.modular_inverse(9, 26)
    logstack.display_logs()
    logstack.empty_messages()

    logstack.add_message(" CRT:")
    CryptographicCalculator.chinese_remainder_theorem([1, 3, 3], [3, 5, 7])
    logstack.display_logs()
