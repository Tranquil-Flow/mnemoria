"""
Rule-based observers for Mnemoria v0.2.0.

Each observer watches events and produces PendingFacts for downstream processing.
"""
from __future__ import annotations

from typing import Protocol

from dataclasses import dataclass


@dataclass
class PendingFact:
    content: str
    type: str           # 'C' | 'D' | 'V'
    target: str
    source: str         # 'observed' | 'user_stated' | 'agent_inference'
    provenance: dict
    retract: bool = False

    @property
    def is_retraction(self) -> bool:
        """True when this PendingFact signals a retraction of a prior provisional fact."""
        return self.retract


class Observer(Protocol):
    """Protocol for rule-based event observers."""
    name: str

    def observe(self, event: dict) -> list[PendingFact]:
        """
        Examine an event and return zero or more PendingFacts.

        Parameters
        ----------
        event : dict
            Structured event with at minimum:
            - kind: 'tool_call' | 'tool_result' | 'user_message' | 'agent_message'
            - session_id: str
            - timestamp: float
            - payload: dict (kind-specific)

        Returns
        -------
        list[PendingFact]
            Zero or more facts extracted from the event. Empty list means no match.
        """
        ...


from .user_statement import UserStatementObserver
from .user_content import UserContentObserver
from .tool_output import PytestObserver, GitObserver, FileObserver
from .error_context import ErrorContextObserver
from .memory_write import MemoryWriteObserver
from .delegation import DelegationObserver

__all__ = [
    "PendingFact", "Observer",
    "UserStatementObserver", "UserContentObserver",
    "PytestObserver", "GitObserver", "FileObserver",
    "ErrorContextObserver", "MemoryWriteObserver", "DelegationObserver",
    "all_observers",
]


def all_observers() -> list:
    """Return every built-in observer instance."""
    return [
        UserStatementObserver(),
        UserContentObserver(),
        PytestObserver(),
        GitObserver(),
        FileObserver(),
        ErrorContextObserver(),
        MemoryWriteObserver(),
        DelegationObserver(),
    ]