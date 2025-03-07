from typing import List, Tuple

from cryptography_calculator.calculator import CryptographicCalculator


class TestEuclid:
    def test_euclid_normal_case(self):
        # For inputs 1071 and 462, expected steps are:
        # Step 1: (1071, 462, 147) because 1071 % 462 = 147
        # Step 2: (462, 147, 21) because 462 % 147 = 21
        # Step 3: (147, 21, 0) because 147 % 21 = 0

        expected_steps: List[Tuple[int, int, int]] = [(1071, 462, 147), (462, 147, 21), (147, 21, 0)]
        result = CryptographicCalculator.euclid(1071, 462)
        assert result == expected_steps

    def test_euclid_first_operand_zero(self):
        # When a is 0 and b is nonzero.
        # Calculation: (0, 5, 0) because 0 % 5 = 0, then stops.

        expected_steps: List[Tuple[int, int, int]] = [(0, 5, 0)]
        result = CryptographicCalculator.euclid(0, 5)
        assert result == expected_steps

    def test_euclid_second_operand_zero(self):
        # When the second operand is 0, the algorithm returns an empty list.

        expected_steps: List[Tuple[int, int, int]] = []
        result = CryptographicCalculator.euclid(5, 0)
        assert result == expected_steps

    def test_euclid_case_with_smaller_first_operand(self):
        # When a < b, for example: (12, 15)
        # Step 1: (12, 15, 12) because 12 % 15 = 12
        # Step 2: (15, 12, 3) because 15 % 12 = 3
        # Step 3: (12, 3, 0) because 12 % 3 = 0

        expected_steps: List[Tuple[int, int, int]] = [(12, 15, 12), (15, 12, 3), (12, 3, 0)]
        result = CryptographicCalculator.euclid(12, 15)
        assert result == expected_steps

    def test_euclid_both_zero(self):
        # If both a and b are 0, the while loop never executes.

        expected_steps: List[Tuple[int, int, int]] = []
        result = CryptographicCalculator.euclid(0, 0)
        assert result == expected_steps
