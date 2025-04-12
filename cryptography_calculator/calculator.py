import logging
from functools import reduce
from operator import mul
from typing import List, Tuple

from cryptography_calculator.utils.logger import LogStack
from cryptography_calculator.utils.primes import two_prime_decomposition
from cryptography_calculator.utils.rsa_functions import Lambda, Phi, RSAFunctions


class CryptographicCalculator:
    logstack: LogStack | None = None

    @staticmethod
    def initialise_logger(logstack: LogStack) -> None:
        CryptographicCalculator.logstack = logstack

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
        a_copy = a
        b_copy = b

        while b:
            remainder = a % b
            steps.append((a, b, remainder))

            CryptographicCalculator.logstack.add_message(f"{a} = {a // b} * {b} + {remainder}")

            a, b = b, remainder

        CryptographicCalculator.logstack.add_message(f"\nFinal Result:\n\tgcd({a_copy}, {b_copy}) = {a}")
        return steps

    @staticmethod
    def __log_extended_euclid(a: int, b: int, x: int, y: int) -> None:
        x_new, a_new = (abs(x), abs(a)) if x < 0 and a < 0 else (x, a)
        y_new, b_new = (abs(y), abs(b)) if y < 0 and b < 0 else (y, b)

        CryptographicCalculator.logstack.add_message(f"1 = {x_new} * {a_new} + {y_new} * {b_new}")

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
        CryptographicCalculator.logstack.add_message(f"\nEuclid for: a = {a}, b = {b}:")
        CryptographicCalculator.euclid(a, b)
        CryptographicCalculator.logstack.add_message(f"\nExtended Euclid for: a = {a}, b = {b}:")
        gcd, x, _ = CryptographicCalculator.extended_euclid(a, b)

        if gcd != 1:
            raise ValueError(f"Modular inverse does not exist for {a} mod {b}")

        CryptographicCalculator.logstack.add_message(f"\nFinal Result:\n\t inv({a}) mod {b} = {x % b}\n")
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

        CryptographicCalculator.logstack.add_message(f"N = {N}")

        x = 0
        index = 1
        for ai, bi in zip(a, b):
            Ni = N // bi

            CryptographicCalculator.logstack.add_message(f"\n\t\t\t Step: {index}")
            CryptographicCalculator.logstack.add_message(f"\nModular Inverse for: a = {Ni}, b = {bi}:")
            CryptographicCalculator.logstack.add_message(f"\tSteps:")

            bi_inv = CryptographicCalculator.modular_inverse(Ni, bi)
            x += ai * Ni * bi_inv

            CryptographicCalculator.logstack.add_message(
                f"\nResult:\nai = {ai}, bi = {Ni}, bi_inv = {bi_inv}, x = {x}\n"
            )

            index += 1

        CryptographicCalculator.logstack.add_message(f"Final Result: {x % N}")
        return x % N

    @staticmethod
    def fast_exponentiation(a: int, p: int, m: int) -> int:
        """Computes (a^p) % m using fast exponentiation (recursive method).

        Parameters
        ----------
        a : int
            The base integer.
        p : int
            The exponent.
        m : int
            The modulus.

        Returns
        -------
        int
            The result of (a^p) % m.
        """
        if p == 0:
            CryptographicCalculator.logstack.add_message(f"Power is 0, result is trivial = 1")
            return 1

        p_bin = bin(p)[2:]
        powers = [2**power for power, value in enumerate(p_bin[::-1]) if value == "1"]
        powers_str = " * ".join([f"{a}^{power}" for power in powers])

        CryptographicCalculator.logstack.add_message(f"Binary for {p} = {p_bin}\n")
        CryptographicCalculator.logstack.add_message(f"\t\t\tSteps:")
        CryptographicCalculator.logstack.add_message(f"{a}^{p} = {powers_str}")

        mem: dict[int, int] = {1: a}

        def mod_exp(power: int) -> int:
            if power in mem:
                return mem[power]

            result = mod_exp(power // 2)
            mem[power] = result * result % m

            CryptographicCalculator.logstack.add_message(
                f"{a}^{power} = {a}^{power // 2} * {a}^{power // 2} = {result} * {result} = {result * result % m}"
            )

            return result * result % m

        starting_power = 2 ** (len(p_bin) - 1)
        mod_exp(starting_power)

        result = reduce(mul, [mem[power] for power in powers]) % m
        results_str = f"{a} ^ {p} % {m} = " + " * ".join([str(mem[power]) for power in powers]) + f" % {m}"
        CryptographicCalculator.logstack.add_message(f"\nFinal Result:\n\ta^p % N = {results_str} = {result}")

        return result

    @staticmethod
    def rsa(N: int, e: int, m: int, modified: bool = False) -> Tuple[int, int, int]:
        """Perform RSA encryption and decryption, compute the private key.

        Parameters
        ----------
        N : int
            The modulus for the RSA algorithm, typically the product of two prime numbers (p and q).
        e : int
            The public exponent for the RSA algorithm, typically a small prime number like 3 or 65537.
        m : int
            The message to be encrypted and decrypted.
        modified : bool, optional
            If True, uses a modified version of the function for the RSA key generation;
            if False, the standard function is used. The default is False.

        Returns
        -------
        Tuple[int, int, int]
            A tuple containing:
            - The encrypted message.
            - The private key (d).
            - The decrypted message.
        """

        rsa_function: RSAFunctions = Lambda() if modified else Phi()
        rsa_function_name = type(rsa_function).__name__

        CryptographicCalculator.logstack.add_message(f"N = {N}\ne = {e}\nm = {m}")
        CryptographicCalculator.logstack.add_message(f"\t\tStep 1: {rsa_function_name}(N)")

        p, q = two_prime_decomposition(N)
        result = rsa_function(p=p, q=q)

        CryptographicCalculator.logstack.add_message(f"\nN = p * q\n{N} = {p} * {q}")
        CryptographicCalculator.logstack.add_message(
            f"{rsa_function_name}(N) = {type(rsa_function).__name__}({N}) = {result}"
        )

        CryptographicCalculator.logstack.add_message("\n\t\tStep 2: Encryption")
        encrypted_message = CryptographicCalculator.fast_exponentiation(m, e, N)
        CryptographicCalculator.logstack.add_message(f"\nCrypted Message: {encrypted_message}")

        CryptographicCalculator.logstack.add_message("\n\t\tStep 3: Private Key")
        d = CryptographicCalculator.modular_inverse(e, result)
        CryptographicCalculator.logstack.add_message(f"Private Key: {d}")

        CryptographicCalculator.logstack.add_message("\n\t\tStep 3: Decryption")
        decrypted_message = CryptographicCalculator.fast_exponentiation(encrypted_message, d, N)
        CryptographicCalculator.logstack.add_message(f"\nDecrypted Message: {decrypted_message}")

        return encrypted_message, d, decrypted_message


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logstack = LogStack(logger=logger)
    CryptographicCalculator.initialise_logger(logstack)

    CryptographicCalculator.logstack.add_message(" RSA:")
    CryptographicCalculator.rsa(119, 5, 11, modified=True)
    CryptographicCalculator.logstack.display_logs()
    CryptographicCalculator.logstack.empty_messages()
