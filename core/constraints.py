from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


ConstraintType = Literal["hard", "soft"]

ConstraintEvaluator = Callable[[Any, Any], bool]


@dataclass
class Constraint:
    """
    A single constraint applied to a candidate action.

    Parameters
    ----------
    name : str
        Unique name identifying the constraint.

    evaluator : Callable[[Any, Any], bool]
        Function that receives (action, context) and returns True
        when the constraint is satisfied.

    description : str, optional
        Human-readable explanation of what the constraint checks.

    constraint_type : {"hard", "soft"}, default="hard"
        Hard constraints must be satisfied.
        Soft constraints are preferences and may be violated.

    weight : float, default=1.0
        Importance of a soft constraint. For hard constraints this
        value is currently informational.

    enabled : bool, default=True
        Whether the constraint participates in evaluation.

    metadata : dict, optional
        Additional information associated with the constraint.
    """

    name: str
    evaluator: ConstraintEvaluator
    description: str = ""
    constraint_type: ConstraintType = "hard"
    weight: float = 1.0
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # certain important checks
        if not self.name.strip():
            raise ValueError("Constraint name cannot be empty.")

        if self.constraint_type not in ("hard", "soft"):
            raise ValueError(
                "constraint_type must be either 'hard' or 'soft'."
            )

        if self.weight < 0:
            raise ValueError("Constraint weight cannot be negative.")

        if not callable(self.evaluator):
            raise TypeError("evaluator must be callable.")

    def evaluate(self, action: Any, context: Any = None) -> bool:
        """
        Evaluate this constraint against an action and context.

        Parameters
        ----------
        action : Any
            Candidate action being evaluated.

        context : Any, optional
            Current decision context.

        Returns
        -------
        bool
            True if the constraint is satisfied or disabled.
        """
        if not self.enabled:
            return True

        return bool(self.evaluator(action, context))


@dataclass
class ConstraintViolation:
    """
    Represents a failed constraint evaluation.
    """

    constraint: Constraint
    reason: str | None = None

    @property
    def name(self) -> str:
        """Return the name of the violated constraint."""
        return self.constraint.name

    @property
    def constraint_type(self) -> ConstraintType:
        """Return whether the constraint is hard or soft."""
        return self.constraint.constraint_type

    @property
    def weight(self) -> float:
        """Return the constraint weight."""
        return self.constraint.weight


@dataclass
class ConstraintResult:
    """
    Result of evaluating all constraints for an action.
    """

    valid: bool
    satisfied: list[Constraint] = field(default_factory=list)
    violations: list[ConstraintViolation] = field(default_factory=list)

    @property
    def hard_violations(self) -> list[ConstraintViolation]:
        """Return only violated hard constraints."""
        return [
            violation
            for violation in self.violations
            if violation.constraint_type == "hard"
        ]

    @property
    def soft_violations(self) -> list[ConstraintViolation]:
        """Return only violated soft constraints."""
        return [
            violation
            for violation in self.violations
            if violation.constraint_type == "soft"
        ]

    @property
    def is_fully_satisfied(self) -> bool:
        """
        Return True when every enabled constraint is satisfied.
        """
        return len(self.violations) == 0


class Constraints:
    """
    Manage and evaluate a collection of constraints.

    Parameters
    ----------
    constraints : list[Constraint], optional
        Initial collection of constraints.

    Examples
    --------
    >>> constraints = Constraints()

    >>> constraints.add(
    ...     Constraint(
    ...         name="max_cost",
    ...         evaluator=lambda action, context:
    ...             action["cost"] <= context["max_cost"],
    ...         description="Action must not exceed the maximum cost.",
    ...         constraint_type="hard",
    ...     )
    ... )

    >>> result = constraints.evaluate(
    ...     action={"cost": 50},
    ...     context={"max_cost": 100},
    ... )

    >>> result.valid
    True
    """

    def __init__(
        self,
        constraints: list[Constraint] | None = None,
    ) -> None:
        self._constraints: list[Constraint] = []

        if constraints:
            for constraint in constraints:
                self.add(constraint)

    def evaluate(
        self,
        action: Any,
        context: Any = None,
    ) -> ConstraintResult:
        """
        Evaluate all enabled constraints against an action.

        All constraints are evaluated so that the caller receives
        the complete set of violations.

        Parameters
        ----------
        action : Any
            Candidate action being evaluated.

        context : Any, optional
            Current decision context.

        Returns
        -------
        ConstraintResult
            Structured result containing satisfied constraints
            and violations.

        Notes
        -----
        An action is considered valid when it does not violate
        any hard constraint.

        Soft constraint violations do not make the action invalid.
        """
        satisfied: list[Constraint] = []
        violations: list[ConstraintViolation] = []

        for constraint in self._constraints:

            if not constraint.enabled:
                continue

            try:
                passed = constraint.evaluate(action, context)

            except Exception as exc:
                # A constraint that cannot be evaluated should not
                # silently pass.
                violations.append(
                    ConstraintViolation(
                        constraint=constraint,
                        reason=(
                            f"Constraint evaluation failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                )
                continue

            if passed:
                satisfied.append(constraint)

            else:
                violations.append(
                    ConstraintViolation(
                        constraint=constraint,
                        reason=constraint.description or None,
                    )
                )

        hard_violations = [
            violation
            for violation in violations
            if violation.constraint_type == "hard"
        ]

        return ConstraintResult(
            valid=len(hard_violations) == 0,
            satisfied=satisfied,
            violations=violations,
        )

    def check(
        self,
        action: Any,
        context: Any = None,
    ) -> bool:
        """
        Convenience method that returns only whether an action
        satisfies all hard constraints.

        Parameters
        ----------
        action : Any
            Candidate action.

        context : Any, optional
            Current decision context.

        Returns
        -------
        bool
            True if no hard constraints are violated.
        """
        return self.evaluate(action, context).valid

    def add(self, constraint: Constraint) -> None:
        """
        Add a constraint.

        Parameters
        ----------
        constraint : Constraint
            Constraint to add.

        Raises
        ------
        TypeError
            If the supplied object is not a Constraint.

        ValueError
            If another constraint with the same name already exists.
        """
        if not isinstance(constraint, Constraint):
            raise TypeError(
                "constraint must be an instance of Constraint."
            )

        if self.get(constraint.name) is not None:
            raise ValueError(
                f"Constraint '{constraint.name}' already exists."
            )

        self._constraints.append(constraint)

    def remove(self, name: str) -> None:
        """
        Remove a constraint by name.

        Parameters
        ----------
        name : str
            Name of the constraint to remove.

        Raises
        ------
        KeyError
            If the constraint does not exist.
        """
        constraint = self.get(name)

        if constraint is None:
            raise KeyError(
                f"Constraint '{name}' does not exist."
            )

        self._constraints.remove(constraint)

    def get(self, name: str) -> Constraint | None:
        """
        Retrieve a constraint by name.

        Parameters
        ----------
        name : str
            Constraint name.

        Returns
        -------
        Constraint or None
            Matching constraint, if found.
        """
        for constraint in self._constraints:
            if constraint.name == name:
                return constraint

        return None

    def enable(self, name: str) -> None:
        """
        Enable a constraint.
        """
        constraint = self._require(name)
        constraint.enabled = True

    def disable(self, name: str) -> None:
        """
        Disable a constraint without removing it.
        """
        constraint = self._require(name)
        constraint.enabled = False

    def list(self) -> list[Constraint]:
        """
        Return all registered constraints.

        Returns
        -------
        list[Constraint]
            A copy of the constraint collection.
        """
        return list(self._constraints)

    def clear(self) -> None:
        """
        Remove all constraints.
        """
        self._constraints.clear()

    def _require(self, name: str) -> Constraint:
        """
        Retrieve a constraint or raise KeyError.
        """
        constraint = self.get(name)

        if constraint is None:
            raise KeyError(
                f"Constraint '{name}' does not exist."
            )

        return constraint

    def __len__(self) -> int:
        """Return the number of registered constraints."""
        return len(self._constraints)

    def __iter__(self):
        """Iterate over registered constraints."""
        return iter(self._constraints)

    def __contains__(self, name: str) -> bool:
        """Check whether a constraint with the given name exists."""
        return self.get(name) is not None