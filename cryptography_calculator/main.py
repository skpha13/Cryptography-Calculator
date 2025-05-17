import argparse
import logging
from argparse import ArgumentParser
from typing import get_args

from cryptography_calculator.calculator import CryptographicCalculator
from cryptography_calculator.utils.logger import LogStack
from cryptography_calculator.utils.types import ElGamalOperations


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

    # RSA
    parser_rsa = subparsers.add_parser("rsa", help="Compute RSA private key, encrypt/decrypt message")
    parser_rsa.add_argument("N", type=int, help="Modulo value")
    parser_rsa.add_argument("e", type=int, help="Public key")
    parser_rsa.add_argument("m", type=int, help="Message to encrypt")
    parser_rsa.add_argument(
        "--modified",
        type=bool,
        help="Whether or not to use the modified or classical method. Optional, default one to use is classical",
        default=False,
    )
    parser_rsa.add_argument(
        "--decrypt",
        type=bool,
        help="Whether or not to encrypt or decrypt the message. Optional, default one to use is False",
        default=False,
    )

    # El Gamal
    parser_el_gamal = subparsers.add_parser("el-gamal", help="Compute El Gamal public key and crypted/decrypt message")
    parser_el_gamal.add_argument("p", type=int, help="Modulo Value")
    parser_el_gamal.add_argument("g", type=int, help="Generator Value")
    parser_el_gamal.add_argument("k", type=int, help="Private Key of A")
    parser_el_gamal.add_argument("y", type=int, help="Private Key of B")
    parser_el_gamal.add_argument("m", type=int, help="Message to encrypt")
    parser_el_gamal.add_argument("operation", choices=get_args(ElGamalOperations), help="Message to encrypt")

    # Multiparty Computation
    parser_mpc = subparsers.add_parser(
        "multiparty-computation", help="Perform secure multiparty computation with secret-shared values"
    )
    parser_mpc.add_argument(
        "--sharing-initial",
        type=lambda s: eval(s),
        required=True,
        help="Initial sharing polynomials (list of lists of coefficients)",
    )
    parser_mpc.add_argument(
        "--sharing-multiply",
        type=lambda s: eval(s),
        required=True,
        help="Multiplication sharing polynomials (list of lists of coefficients)",
    )
    parser_mpc.add_argument("--modulo", type=int, required=True, help="Modulo for field arithmetic")
    parser_mpc.add_argument(
        "--operations",
        type=lambda s: eval(s),
        required=True,
        help="List of operations to perform in the format [(op, var1, var2), ...]",
    )

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
        "rsa": lambda args: (
            CryptographicCalculator.logstack.add_message(" RSA:"),
            CryptographicCalculator.rsa(args.N, args.e, args.m, args.modified, args.decrypt),
        ),
        "el-gamal": lambda args: (
            CryptographicCalculator.logstack.add_message(" El Gamal:"),
            CryptographicCalculator.el_gamal(args.p, args.g, args.k, args.y, args.m, args.operation),
        ),
        "multiparty-computation": lambda args: (
            CryptographicCalculator.logstack.add_message(" Multiparty Computation:"),
            CryptographicCalculator.multiparty_computation(
                args.sharing_initial, args.sharing_multiply, args.modulo, args.operations
            ),
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
