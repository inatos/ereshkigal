"""Ereshkigal Python package: native Engine + stdio Client."""

from .client import Client

try:
    # Native extension built by maturin (ereshkigal.abi3.so).
    from .ereshkigal import Engine, schema  # type: ignore
except ImportError:  # pragma: no cover - editable without rebuild
    Engine = None  # type: ignore
    schema = None  # type: ignore

__all__ = ["Client", "Engine", "schema"]
