from cryptography_calculator.calculator import CryptographicCalculator


class TestLogStack:
    def test_logging_euclid(self):
        CryptographicCalculator.logstack.empty_messages()

        CryptographicCalculator.euclid(54, 24)
        assert CryptographicCalculator.logstack.messages == [
            "54 = 2 * 24 + 6",
            "24 = 4 * 6 + 0",
            "\nFinal Result:\n\tgcd(54, 24) = 6",
        ]

        CryptographicCalculator.logstack.empty_messages()

        CryptographicCalculator.euclid(17, 3)
        assert CryptographicCalculator.logstack.messages == [
            "17 = 5 * 3 + 2",
            "3 = 1 * 2 + 1",
            "2 = 2 * 1 + 0",
            "\nFinal Result:\n\tgcd(17, 3) = 1",
        ]

    def test_logging_extended_euclid(self):
        CryptographicCalculator.logstack.empty_messages()
        CryptographicCalculator.extended_euclid(383, 74)

        assert CryptographicCalculator.logstack.messages == [
            "1 = 1 * 1 + 0 * 4",
            "1 = -2 * 4 + 1 * 9",
            "1 = 3 * 9 + -2 * 13",
            "1 = -17 * 13 + 3 * 74",
            "1 = 88 * 74 + -17 * 383",
            "1 = -17 * 383 + 88 * 74",
        ]

    def test_logging_modular_inverse(self):
        CryptographicCalculator.logstack.empty_messages()
        CryptographicCalculator.modular_inverse(9, 26)

        assert CryptographicCalculator.logstack.messages == [
            "1 = 1 * 1 + 0 * 8",
            "1 = -1 * 8 + 1 * 9",
            "1 = 3 * 9 + -1 * 26",
            "\nFinal Result:\n\t inv(9) mod 26 = 3\n",
        ]

    def test_logging_crt(self):
        CryptographicCalculator.logstack.empty_messages()
        CryptographicCalculator.chinese_remainder_theorem([2, 3, 10], [5, 7, 11])

        assert CryptographicCalculator.logstack.messages == [
            "N = 385",
            "\n\t\t\t Step: 1",
            "Euclid for: a = 77, b = 5:",
            "77 = 15 * 5 + 2",
            "5 = 2 * 2 + 1",
            "2 = 2 * 1 + 0",
            "\nFinal Result:\n\tgcd(77, 5) = 1",
            "\nModular Inverse for: a = 77, b = 5:",
            "\tSteps:",
            "1 = 1 * 1 + 0 * 2",
            "1 = -2 * 2 + 1 * 5",
            "1 = 31 * 5 + -2 * 77",
            "1 = -2 * 77 + 31 * 5",
            "\nFinal Result:\n\t inv(77) mod 5 = 3\n",
            "\nResult:\nai = 2, bi = 77, bi_inv = 3, x = 462\n",
            "\n\t\t\t Step: 2",
            "Euclid for: a = 55, b = 7:",
            "55 = 7 * 7 + 6",
            "7 = 1 * 6 + 1",
            "6 = 6 * 1 + 0",
            "\nFinal Result:\n\tgcd(55, 7) = 1",
            "\nModular Inverse for: a = 55, b = 7:",
            "\tSteps:",
            "1 = 1 * 1 + 0 * 6",
            "1 = -1 * 6 + 1 * 7",
            "1 = 8 * 7 + -1 * 55",
            "1 = -1 * 55 + 8 * 7",
            "\nFinal Result:\n\t inv(55) mod 7 = 6\n",
            "\nResult:\nai = 3, bi = 55, bi_inv = 6, x = 1452\n",
            "\n\t\t\t Step: 3",
            "Euclid for: a = 35, b = 11:",
            "35 = 3 * 11 + 2",
            "11 = 5 * 2 + 1",
            "2 = 2 * 1 + 0",
            "\nFinal Result:\n\tgcd(35, 11) = 1",
            "\nModular Inverse for: a = 35, b = 11:",
            "\tSteps:",
            "1 = 1 * 1 + 0 * 2",
            "1 = -5 * 2 + 1 * 11",
            "1 = 16 * 11 + -5 * 35",
            "1 = -5 * 35 + 16 * 11",
            "\nFinal Result:\n\t inv(35) mod 11 = 6\n",
            "\nResult:\nai = 10, bi = 35, bi_inv = 6, x = 3552\n",
            "Final Result: 87",
        ]

    def test_logging_fast_exponentiation(self):
        CryptographicCalculator.logstack.empty_messages()
        CryptographicCalculator.fast_exponentiation(23, 43, 77)

        assert CryptographicCalculator.logstack.messages == [
            "Binary for 43 = 101011\n",
            "\t\t\tSteps:",
            "23^43 = 23^1 * 23^2 * 23^8 * 23^32",
            "23^2 = 23^1 * 23^1 = 23 * 23 = 67",
            "23^4 = 23^2 * 23^2 = 67 * 67 = 23",
            "23^8 = 23^4 * 23^4 = 23 * 23 = 67",
            "23^16 = 23^8 * 23^8 = 67 * 67 = 23",
            "23^32 = 23^16 * 23^16 = 23 * 23 = 67",
            "\nFinal Result: 23 ^ 43 = 23 * 67 * 67 * 67 % 77 = 23",
        ]
