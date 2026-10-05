from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


@dataclass
class Context:
    """
    Represents the current state available to the decision engine.

    Parameters
    ----------
    data : Mapping[str, Any], optional
        Context variables available to the decision engine.

    metadata : Mapping[str, Any], optional
        Additional information about the context itself, such as
        source, version, request ID, or environment.

    timestamp : datetime, optional
        Time at which the context was created. If omitted, the
        current UTC time is used.

    Examples
    --------
    >>> context = Context(
    ...     data={
    ...         "traffic": "heavy",
    ...         "eta": 35,
    ...         "sla": 25,
    ...     }
    ... )

    >>> context.get("eta")
    35
    """

    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        self.data = dict(self.data)
        self.metadata = dict(self.metadata)

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "Context timestamp must be timezone-aware."
            )

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a value from the context.

        Parameters
        ----------
        key : str
            Context variable name.

        default : Any, optional
            Value returned when the key does not exist.

        Returns
        -------
        Any
            Stored value or default.
        """
        return self.data.get(key, default)

    def require(self, key: str) -> Any:
        """
        Retrieve a required context variable.

        Raises
        ------
        KeyError
            If the variable does not exist.
        """
        if key not in self.data:
            raise KeyError(
                f"Required context variable '{key}' is missing."
            )

        return self.data[key]

    def set(self, key: str, value: Any) -> None:
        """
        Add or update a context variable.
        """
        self.data[key] = value

    def remove(self, key: str) -> Any:
        """
        Remove a context variable.

        Returns
        -------
        Any
            Removed value.

        Raises
        ------
        KeyError
            If the variable does not exist.
        """
        return self.data.pop(key)

    def has(self, key: str) -> bool:
        """
        Check whether a context variable exists.
        """
        return key in self.data

    def keys(self):
        """Return context variable names."""
        return self.data.keys()

    def values(self):
        """Return context variable values."""
        return self.data.values()

    def items(self):
        """Return context variable key-value pairs."""
        return self.data.items()

    def update(self, values: Mapping[str, Any]) -> None:
        """
        Add or update multiple context variables.
        """
        self.data.update(values)

    def copy(self) -> "Context":
        """
        Create a copy of the context.
        """
        return Context(
            data=self.data.copy(),
            metadata=self.metadata.copy(),
            timestamp=self.timestamp,
        )

    def __getitem__(self, key: str) -> Any:
        return self.require(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"Context("
            f"data={self.data!r}, "
            f"metadata={self.metadata!r}, "
            f"timestamp={self.timestamp!r}"
            f")"
        )