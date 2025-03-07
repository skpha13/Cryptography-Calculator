import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CryptographicCalculator:
    @staticmethod
    def euclid(a: int, b: int) -> List[Tuple[int, int, int]]:
        steps = []

        while b:
            remainder = a % b
            steps.append((a, b, remainder))

            logger.info(f" {a} = {a // b} * {b} + {remainder}")

            a, b = b, remainder

        return steps

    @staticmethod
    def extended_euclid(a: int, b: int) -> Tuple[int, int, int]:
        """
        Returns a tuple (gcd, x, y) such that a*x + b*y == gcd, which is the greatest common divisor of a and b.
        """
        if a == 0:
            return b, 0, 1

        gcd, x1, y1 = CryptographicCalculator.extended_euclid(b % a, a)
        x = y1 - (b // a) * x1
        y = x1

        logger.info(f" 1 = {y} * {b} + {x} * {a}")

        return gcd, x, y


if __name__ == "__main__":
    print(CryptographicCalculator.euclid(113, 15))
    print(CryptographicCalculator.extended_euclid(113, 15))
