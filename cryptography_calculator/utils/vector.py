from typing import List


def vector_to_str(vector: List[int]) -> str:
    return " ".join(f"{elem:3d}" for elem in vector)


def compute_polynomial_value(polynom: List[int], x: int) -> int:
    return sum([coefficient * (x**index) for index, coefficient in enumerate(polynom)])


def polynom_to_str(polynom: List[int]) -> str:
    def format_x(index: int) -> str:
        if index == 0:
            return ""

        if index == 1:
            return "x"

        return f"x^{index}"

    return " + ".join([f"{coefficient}{format_x(index)}" for index, coefficient in enumerate(polynom)])
