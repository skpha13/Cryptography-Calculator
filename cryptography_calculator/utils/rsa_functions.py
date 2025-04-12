import math
from abc import ABC, abstractmethod


class RSAFunctions(ABC):
    """Abstract base class for RSA-related functions, such as Euler's totient function
    (phi(n)) and Carmichael's lambda function (lambda(n)).
    """

    @abstractmethod
    def __call__(self, p: int, q: int) -> int:
        pass


class Phi(RSAFunctions):
    """Class for calculating Euler's Totient function phi(p, q) in RSA."""

    def __call__(self, p: int, q: int) -> int:
        return (p - 1) * (q - 1)


class Lambda(RSAFunctions):
    """Class for calculating Carmichael's lambda function lambda(p, q) in RSA."""

    def __call__(self, p: int, q: int) -> int:
        return math.lcm(p - 1, q - 1)
