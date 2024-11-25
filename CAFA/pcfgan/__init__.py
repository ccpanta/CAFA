import sys

from . import logger
from .data import DataModule  # noqa: F401
from .PCFGAN import PCFGAN  # noqa: F401

logger.add(sink=sys.stderr, level="CRITICAL")