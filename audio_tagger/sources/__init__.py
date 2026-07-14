"""Source adapters. Import lazily so a missing optional dep or key never
breaks the whole package — the CLI decides which sources to activate.
"""

from .base import Source, query_string

__all__ = ["Source", "query_string"]
