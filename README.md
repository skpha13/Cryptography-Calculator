# Cryptography-Calculator

A Python package offering a wide range of cryptographic algorithms.

## User Documentation

### Installation    

To get started, install the package by following these steps:

```bash
# clone repository
git clone https://github.com/skpha13/Cryptography-Calculator.git

# enter the directory 
cd cryptography-calculator

# install all required dependencies
pip install -e .
```

### Summary of CLI Commands

| Command                  | Parameters                                                                                                                         | Description                                                                                 |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `euclid`                 | `a` (int), `b` (int)                                                                                                               | Computes the greatest common divisor of two numbers.                                        |
| `extended-euclid`        | `a` (int), `b` (int)                                                                                                               | Computes the extended Euclidean algorithm, returning x, y such that ax + by = gcd(a, b).    |
| `modular-inverse`        | `a` (int), `b` (int)                                                                                                               | Computes the modular inverse of `a` modulo `b`.                                             |
| `crt`                    | `--a` (list\[int]), `--b` (list\[int])                                                                                             | Solves the system of congruences using the Chinese Remainder Theorem.                       |
| `fast-exponentiation`    | `a` (int), `p` (int), `m` (int)                                                                                                    | Computes `(a^p) % m` efficiently.                                                           |
| `rsa`                    | `N` (int), `e` (int), `m` (int), `--modified` (bool, optional), `--decrypt` (bool, optional)                                       | Performs RSA encryption or decryption based on flags.                                       |
| `el-gamal`               | `p` (int), `g` (int), `k` (int), `y` (int), `m` (int), `operation` (str)                                                           | Encrypts/decrypts using El Gamal scheme. Operation must be one of: additive/multiplicative. |
| `multiparty-computation` | `--sharing-initial` (list\[list\[int]]), `--sharing-multiply` (list\[list\[int]]), `--modulo` (int), `--operations` (list\[tuple]) | Secure computation using secret sharing.                                                    |

### Example Commands

```bash
# euclid
python ./cryptography_calculator/main.py euclid 240 46

# extended euclid
python ./cryptography_calculator/main.py extended-euclid 240 46

# modular inverse
python ./cryptography_calculator/main.py modular-inverse 3 11

# chinese remainder theorem
python ./cryptography_calculator/main.py crt --a 2 3 2 --b 3 5 7

# fast exponentiation
python ./cryptography_calculator/main.py fast-exponentiation 2 10 1000

# rsa encryption
python ./cryptography_calculator/main.py rsa 3233 17 65

# rsa decryption (with flags)
python ./cryptography_calculator/main.py rsa 3233 17 2790 --decrypt True

# rsa using lambda instead of phi
python ./cryptography_calculator/main.py rsa 3233 17 65 --modified True

# el gamal additive
python ./cryptography_calculator/main.py el-gamal 467 2 127 327 123 additive

# el gamal multiplicative
python ./cryptography_calculator/main.py el-gamal 467 2 127 327 123 multiplicative

# multiparty computation
python ./cryptography_calculator/main.py multiparty-computation --sharing-initial '[[3, 1], [4, 2], [5, 3]]' --sharing-multiply '[[0, 4], [0, 5], [0, 6]]' --modulo 1000000  --operations "[['mul', 'x1', 'x2'], ['add', 'temp0', 'x3']]"
```

## Developer Documentation

### Install Optional Packages

For development purposes, install the optional packages with:

```bash
pip install cryptography-calculator[dev]
```

This will install the following tools:

- **Black:**  A code formatter to ensure consistent style.
- **isort:**  A tool for sorting imports automatically.
- **Pytest:** A testing framework for running unit tests.

### Running Tests

Run the test suite to ensure everything is working:

```bash
python -m pytest
```

## Observations

- The Python docstrings for functions and classes were generated with the assistance of **ChatGPT**. 
- Portions of the **README**, particularly repetitive sections such as the command table, were also created with the help of **ChatGPT**.