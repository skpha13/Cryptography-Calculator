import logging

import pytest
from cryptography_calculator.calculator import CryptographicCalculator
from cryptography_calculator.utils.logger import LogStack


@pytest.fixture(autouse=True)
def initialize_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logstack = LogStack(logger=logger)

    CryptographicCalculator.initialise_logger(logstack)
