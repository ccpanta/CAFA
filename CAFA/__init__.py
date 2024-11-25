import sys

from pcfgan import logger
from pcfgan.data import DataModule  # noqa: F401
from pcfgan.PCFGAN import PCFGAN  # noqa: F401

logger.add(sink=sys.stderr, level="CRITICAL")
