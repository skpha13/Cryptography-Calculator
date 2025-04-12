import argparse
import logging
from argparse import ArgumentParser

from cryptography_calculator.calculator import CryptographicCalculator
from cryptography_calculator.utils.logger import LogStack

# TODO: add rsa


def add_arguments(parser: ArgumentParser) -> ArgumentParser:
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Euclidean Algorithm
    parser_euclid = subparsers.add_parser("euclid", help="Compute the Euclidean Algorithm")
    parser_euclid.add_argument("a", type=int, help="First number")
    parser_euclid.add_argument("b", type=int, help="Second number")

    # Extended Euclidean Algorithm
    parser_ext_euclid = subparsers.add_parser("extended-euclid", help="Compute the Extended Euclidean Algorithm")
    parser_ext_euclid.add_argument("a", type=int, help="First number")
    parser_ext_euclid.add_argument("b", type=int, help="Second number")

    # Modular Inverse
    parser_mod_inv = subparsers.add_parser("modular-inverse", help="Compute the Modular Inverse")
    parser_mod_inv.add_argument("a", type=int, help="Number to find the modular inverse of")
    parser_mod_inv.add_argument("b", type=int, help="Modulo value")

    # Chinese Remainder Theorem
    parser_crt = subparsers.add_parser("crt", help="Compute the Chinese Remainder Theorem")
    parser_crt.add_argument("--a", nargs="+", type=int, help="List of remainders")
    parser_crt.add_argument("--b", nargs="+", type=int, help="List of moduli")

    # Fast Exponentiation
    parser_exp = subparsers.add_parser("fast-exponentiation", help="Compute Fast Exponentiation")
    parser_exp.add_argument("a", type=int, help="Base number")
    parser_exp.add_argument("p", type=int, help="Exponent")
    parser_exp.add_argument("m", type=int, help="Modulo value")

    return parser


def main():
    parser = argparse.ArgumentParser(description="Cryptographic Calculator CLI")
    parser = add_arguments(parser)
    args = parser.parse_args()

    command_map = {
        "euclid": lambda args: (
            CryptographicCalculator.logstack.add_message(f" Euclid for: a = {args.a}, b = {args.b}"),
            CryptographicCalculator.euclid(args.a, args.b),
        ),
        "extended-euclid": lambda args: (
            CryptographicCalculator.logstack.add_message(f" Euclid for: a = {args.a}, b = {args.b}"),
            CryptographicCalculator.euclid(args.a, args.b),
            CryptographicCalculator.logstack.add_message(f"\nExtended Euclid for: a = {args.a}, b = {args.b}"),
            CryptographicCalculator.extended_euclid(args.a, args.b),
        ),
        "modular-inverse": lambda args: (
            CryptographicCalculator.logstack.add_message(f" Modular Inverse for: a = {args.a}, b = {args.b}"),
            CryptographicCalculator.modular_inverse(args.a, args.b),
        ),
        "crt": lambda args: (
            CryptographicCalculator.logstack.add_message(" CRT:"),
            CryptographicCalculator.chinese_remainder_theorem(args.a, args.b),
        ),
        "fast-exponentiation": lambda args: (
            CryptographicCalculator.logstack.add_message(
                f" Fast Exponentiation for: a = {args.a}, p = {args.p}, m = {args.m}"
            ),
            CryptographicCalculator.fast_exponentiation(args.a, args.p, args.m),
        ),
    }

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logstack = LogStack(logger=logger)
    CryptographicCalculator.initialise_logger(logstack)

    command_map[args.command](args)

    logstack.display_logs()
    logstack.empty_messages()


if __name__ == "__main__":
    main()
