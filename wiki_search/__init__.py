"""Answer questions and check claims using Wikipedia and Claude."""

import warnings

# macOS system Python links LibreSSL; urllib3 warns on import but works fine for these requests.
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

from .agent import Answer, Clarification, WikiQA  # noqa: E402
from .render import Source  # noqa: E402

__all__ = ["Answer", "Clarification", "Source", "WikiQA"]
