"""
Action representation for the OpenDecision engine.

An Action represents a candidate operation that the decision
engine may choose to execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


ActionExecutor = Callable[["Action", Any], Any]


@dataclass
class Action:
    """
    Represents a candidate action.

    Parameters
    ----------
    name : str
        Unique name identifying the action.

    parameters : Mapping[str, Any], optional
        Parameters required to execute the action.

    metadata : Mapping[str, Any], optional
        Additional information associated with the action.

    executor : Callable, optional
        Optional callable responsible for executing the action.

        The executor receives:

            executor(action, context)

        Execution is intentionally optional because the decision
        engine should be able to evaluate actions without executing
        them.

    Examples
    --------
    >>> action = Action(
    ...     name="reassign_rider",
    ...     parameters={
    ...         "rider_id": "R123",
    ...     },
    ... )

    >>> action.name
    'reassign_rider'
    """

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    executor: ActionExecutor | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Action name cannot be empty."
            )

        self.parameters = dict(self.parameters)
        self.metadata = dict(self.metadata)

        if self.executor is not None and not callable(self.executor):
            raise TypeError(
                "executor must be callable or None."
            )

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve an action parameter.
        """
        return self.parameters.get(key, default)

    def require(self, key: str) -> Any:
        """
        Retrieve a required action parameter.

        Raises
        ------
        KeyError
            If the parameter does not exist.
        """
        if key not in self.parameters:
            raise KeyError(
                f"Required action parameter '{key}' is missing."
            )

        return self.parameters[key]

    def set(self, key: str, value: Any) -> None:
        """
        Add or update an action parameter.
        """
        self.parameters[key] = value

    def has(self, key: str) -> bool:
        """
        Check whether an action parameter exists.
        """
        return key in self.parameters

    def update(self, values: Mapping[str, Any]) -> None:
        """
        Add or update multiple action parameters.
        """
        self.parameters.update(values)

    def execute(self, context: Any = None) -> Any:
        """
        Execute the action using its registered executor.

        Parameters
        ----------
        context : Any, optional
            Context in which the action is being executed.

        Returns
        -------
        Any
            Result returned by the executor.

        Raises
        ------
        RuntimeError
            If the action has no executor.
        """
        if self.executor is None:
            raise RuntimeError(
                f"Action '{self.name}' has no executor."
            )

        return self.executor(self, context)

    def copy(self) -> "Action":
        """
        Create a copy of the action.
        """
        return Action(
            name=self.name,
            parameters=self.parameters.copy(),
            metadata=self.metadata.copy(),
            executor=self.executor,
        )

    def __getitem__(self, key: str) -> Any:
        return self.require(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __repr__(self) -> str:
        return (
            f"Action("
            f"name={self.name!r}, "
            f"parameters={self.parameters!r}"
            f")"
        )