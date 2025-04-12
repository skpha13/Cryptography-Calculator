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
            "\nEuclid for: a = 9, b = 26:",
            "9 = 0 * 26 + 9",
            "26 = 2 * 9 + 8",
            "9 = 1 * 8 + 1",
            "8 = 8 * 1 + 0",
            "\nFinal Result:\n\tgcd(9, 26) = 1",
            "\nExtended Euclid for: a = 9, b = 26:",
            "1 = 1 * 1 + 0 * 8",
            "1 = -1 * 8 + 1 * 9",
            "1 = 3 * 9 + -1 * 26",
            "\nFinal Result:\n\t inv(9) mod 26 = 3\n",
        ]

    def test_logging_crt(self):
        CryptographicCalculator.logstack.empty_messages()
        CryptographicCalculator.chinese_remainder_theorem([2, 3, 10], [5, 7, 11])

        print(CryptographicCalculator.logstack.messages)

        assert CryptographicCalculator.logstack.messages == [
            "N = 385",
            "\n\t\t\t Step: 1",
            "\nModular Inverse for: a = 77, b = 5:",
            "\tSteps:",
            "\nEuclid for: a = 77, b = 5:",
            "77 = 15 * 5 + 2",
            "5 = 2 * 2 + 1",
            "2 = 2 * 1 + 0",
            "\nFinal Result:\n\tgcd(77, 5) = 1",
            "\nExtended Euclid for: a = 77, b = 5:",
            "1 = 1 * 1 + 0 * 2",
            "1 = -2 * 2 + 1 * 5",
            "1 = 31 * 5 + -2 * 77",
            "1 = -2 * 77 + 31 * 5",
            "\nFinal Result:\n\t inv(77) mod 5 = 3\n",
            "\nResult:\nai = 2, bi = 77, bi_inv = 3, x = 462\n",
            "\n\t\t\t Step: 2",
            "\nModular Inverse for: a = 55, b = 7:",
            "\tSteps:",
            "\nEuclid for: a = 55, b = 7:",
            "55 = 7 * 7 + 6",
            "7 = 1 * 6 + 1",
            "6 = 6 * 1 + 0",
            "\nFinal Result:\n\tgcd(55, 7) = 1",
            "\nExtended Euclid for: a = 55, b = 7:",
            "1 = 1 * 1 + 0 * 6",
            "1 = -1 * 6 + 1 * 7",
            "1 = 8 * 7 + -1 * 55",
            "1 = -1 * 55 + 8 * 7",
            "\nFinal Result:\n\t inv(55) mod 7 = 6\n",
            "\nResult:\nai = 3, bi = 55, bi_inv = 6, x = 1452\n",
            "\n\t\t\t Step: 3",
            "\nModular Inverse for: a = 35, b = 11:",
            "\tSteps:",
            "\nEuclid for: a = 35, b = 11:",
            "35 = 3 * 11 + 2",
            "11 = 5 * 2 + 1",
            "2 = 2 * 1 + 0",
            "\nFinal Result:\n\tgcd(35, 11) = 1",
            "\nExtended Euclid for: a = 35, b = 11:",
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
            "\nFinal Result:\n\ta^p % N = 23 ^ 43 % 77 = 23 * 67 * 67 * 67 % 77 = 23",
        ]

    def test_logging_rsa(self):
        CryptographicCalculator.logstack.empty_messages()
        CryptographicCalculator.rsa(119, 5, 11)

        assert CryptographicCalculator.logstack.messages == [
            "N = 119\ne = 5\nm = 11",
            "\t\tStep 1: Phi(N)",
            "\nN = p * q\n119 = 7 * 17",
            "Phi(N) = Phi(119) = 96",
            "\n\t\tStep 2: Encryption",
            "Binary for 5 = 101\n",
            "\t\t\tSteps:",
            "11^5 = 11^1 * 11^4",
            "11^2 = 11^1 * 11^1 = 11 * 11 = 2",
            "11^4 = 11^2 * 11^2 = 2 * 2 = 4",
            "\nFinal Result:\n\ta^p % N = 11 ^ 5 % 119 = 11 * 4 % 119 = 44",
            "\nCrypted Message: 44",
            "\n\t\tStep 3: Private Key",
            "\nEuclid for: a = 5, b = 96:",
            "5 = 0 * 96 + 5",
            "96 = 19 * 5 + 1",
            "5 = 5 * 1 + 0",
            "\nFinal Result:\n\tgcd(5, 96) = 1",
            "\nExtended Euclid for: a = 5, b = 96:",
            "1 = 1 * 1 + 0 * 5",
            "1 = -19 * 5 + 1 * 96",
            "\nFinal Result:\n\t inv(5) mod 96 = 77\n",
            "Private Key: 77",
            "\n\t\tStep 3: Decryption",
            "Binary for 77 = 1001101\n",
            "\t\t\tSteps:",
            "44^77 = 44^1 * 44^4 * 44^8 * 44^64",
            "44^2 = 44^1 * 44^1 = 44 * 44 = 32",
            "44^4 = 44^2 * 44^2 = 32 * 32 = 72",
            "44^8 = 44^4 * 44^4 = 72 * 72 = 67",
            "44^16 = 44^8 * 44^8 = 67 * 67 = 86",
            "44^32 = 44^16 * 44^16 = 86 * 86 = 18",
            "44^64 = 44^32 * 44^32 = 18 * 18 = 86",
            "\nFinal Result:\n\ta^p % N = 44 ^ 77 % 119 = 44 * 72 * 67 * 86 % 119 = 11",
            "\nDecrypted Message: 11",
        ]
