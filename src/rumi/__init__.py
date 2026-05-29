"""rumi — a slim agentic library."""
__version__ = "0.1.0"

from rumi.api import *  # noqa: F401,F403,E402
from rumi.api import __all__  # noqa: F401,E402

__all__ = list(__all__) + ["__version__"]
