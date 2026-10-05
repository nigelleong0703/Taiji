"""Jev chooses an observed action. Code owns execution."""

__all__ = ["Agent", "Browser"]


def __getattr__(name):
    # Importing the decision policy must not connect a CDP browser. Pilot and other
    # MCP adapters use the policy with their own transport.
    if name == "Agent":
        from .agent import Agent
        return Agent
    if name == "Browser":
        from .browser import Browser
        return Browser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
