from collections.abc import Callable, Sequence
from typing import Any

from omegaconf import DictConfig

from llca.mappers.modules.column_ref import ColumnRef

Validator = Callable[[DictConfig], list[str]]


class Registry[T]:
    """Bind configuration names to modular builders, validators, and column contracts.

    Each pipeline extension registers construction and validation under the same stable
    name. Optional ``ColumnRef`` metadata lets generic validation and runtime checks derive
    dataset dependencies without embedding component-specific fields in central mappers.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._builders: dict[str, Callable[..., T]] = {}
        self._validators: dict[str, Validator] = {}
        self._column_refs: dict[str, tuple[ColumnRef, ...]] = {}

    def register(
        self, name: str, columns: Sequence[ColumnRef] = ()
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Register one builder and its declarative column dependencies."""

        def decorator(builder: Callable[..., T]) -> Callable[..., T]:
            if name in self._builders:
                raise ValueError(f"{self._kind} '{name}' is already registered")
            self._builders[name] = builder
            self._column_refs[name] = tuple(columns)
            return builder

        return decorator

    def register_validator(self, name: str) -> Callable[[Validator], Validator]:
        """Attach component-specific validation to the corresponding registry name."""

        def decorator(validator: Validator) -> Validator:
            self._validators[name] = validator
            return validator

        return decorator

    def build(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Build a registered component or report all available names."""
        if name not in self._builders:
            raise KeyError(f"unknown {self._kind} '{name}', available: {self.available()}")
        return self._builders[name](*args, **kwargs)

    def validate(self, name: str, cfg: DictConfig) -> list[str]:
        """Return component-specific configuration errors, if a validator is registered."""
        validator = self._validators.get(name)
        return validator(cfg) if validator is not None else []

    def column_refs(self, name: str) -> tuple[ColumnRef, ...]:
        return self._column_refs.get(name, ())

    def is_registered(self, name: str) -> bool:
        return name in self._builders

    def available(self) -> list[str]:
        return sorted(self._builders)
