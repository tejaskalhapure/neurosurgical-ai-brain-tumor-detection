"""Application logging configuration."""

import logging
import sys


def setup_logging():
    """Configures structured console logging for the FastAPI application."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


logger = logging.getLogger("neurosurgical_ai")
