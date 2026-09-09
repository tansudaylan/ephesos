"""Ephesos package.

Ephesos provides the supported forward-modeling workflow for astronomical light
curves. The package exposes the active scientific API from the main module in a
way that preserves compatibility with legacy import patterns and the repository
smoke tests.
"""

from .main import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith('_')]
