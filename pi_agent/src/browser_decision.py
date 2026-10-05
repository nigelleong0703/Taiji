"""Former name of env_decision.py, kept for existing callers and tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_decision import *  # noqa: E402,F401,F403
from env_decision import browser_policy, main  # noqa: E402,F401

if __name__ == "__main__":
    main()
