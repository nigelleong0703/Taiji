"""Compatibility boundary for the browser decision handler.

Only this module knows where the vendored policy lives. Browser execution remains
in the registered MCP adapter; importing this handler must never connect CDP.
"""

import os
import sys
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parents[2] / "examples" / "browser" / "harness"
if _HARNESS_DIR.is_dir() and str(_HARNESS_DIR) not in sys.path:
    sys.path.append(str(_HARNESS_DIR))

try:
    from jev_ultrafast import model
    from jev_ultrafast.model import field_context
    from jev_ultrafast.questions import TEXT_VALUE
    AVAILABLE = True
except ImportError:
    model = field_context = None
    TEXT_VALUE = ""
    AVAILABLE = False


def choose(page, goal, history, cache_session=None):
    return model.choose(page, goal, history, cache_session, relative_tie=True)


def configure(base_url, key):
    model.TYPESAFE_URL = f"{base_url}/v1/systemone"
    os.environ["TYPESAFE_API_KEY"] = key
