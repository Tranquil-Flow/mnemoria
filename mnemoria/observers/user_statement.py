"""
User-statement observer for Mnemoria v0.2.0.

Rule-based pattern matching on user messages to extract explicit preferences,
identity statements, and memory requests.
"""
from __future__ import annotations

import re
from typing import Optional

from . import PendingFact


_WH_WORDS = frozenset([
    "what", "when", "where", "why", "who", "how", "which", "whose", "whom",
])


class UserStatementObserver:
    """Extracts user-stated facts from natural-language messages."""

    name: str = "user_statement"

    def observe(self, event: dict) -> list[PendingFact]:
        if event.get("kind") != "user_message":
            return []

        payload = event.get("payload", {})
        content: str = payload.get("content", "").strip()
        if not content:
            return []

        # --- False positive mitigation: skip questions ---
        if self._is_question(content):
            return []

        session_id = event.get("session_id", "")
        target = self._infer_target(content)

        facts: list[PendingFact] = []

        # "I always / never / prefer (to) ..." → V (value/approach)
        m = re.search(
            r"\bI\s+(always|never|prefer\s+to|prefer|usually|typically)\s+(.*)",
            content, re.IGNORECASE
        )
        if m:
            preference = m.group(2).rstrip(".").strip()
            facts.append(
                PendingFact(
                    content=f"user prefers: {preference}",
                    type="V",
                    target=target,
                    source="user_stated",
                    provenance={
                        "extractor": self.name,
                        "trigger": m.group(1),
                        "sentence": content,
                        "session_id": session_id,
                    },
                )
            )
            return facts

        # "my (email|name|handle|phone|slack) is ..." → C, target = 'identity'
        m = re.search(
            r"\bmy\s+(email|name|handle|phone|slack|telegram|github\s+username)\s+is\s+(\S+)",
            content, re.IGNORECASE
        )
        if m:
            facts.append(
                PendingFact(
                    content=f"user's {m.group(1)} is {m.group(2)}",
                    type="C",
                    target="identity",
                    source="user_stated",
                    provenance={
                        "extractor": self.name,
                        "identity_field": m.group(1),
                        "value": m.group(2),
                        "sentence": content,
                        "session_id": session_id,
                    },
                )
            )
            return facts

        # "(use|don't use|never use|avoid) X for ..." → C
        m = re.search(
            r"\b(?:use|don't\s+use|never\s+use|avoid|prefer|like)\s+(\S+)\s+(?:for|in|as|when)\s+(.*)",
            content, re.IGNORECASE
        )
        if m:
            tool_or_thing = m.group(1).rstrip(".,")
            context = m.group(2).rstrip(".").strip()
            facts.append(
                PendingFact(
                    content=f"{tool_or_thing} for {context}",
                    type="C",
                    target=self._target_from_mention(tool_or_thing, content),
                    source="user_stated",
                    provenance={
                        "extractor": self.name,
                        "thing": tool_or_thing,
                        "context": context,
                        "sentence": content,
                        "session_id": session_id,
                    },
                )
            )
            return facts

        # "don't use X" — standalone negation
        m = re.search(
            r"\b(?:don't|do\s+not|please\s+don't|never|stop)\s+(?:use\s+)?(\S+)",
            content, re.IGNORECASE
        )
        if m:
            avoided = m.group(1).rstrip(".,")
            facts.append(
                PendingFact(
                    content=f"do not use {avoided}",
                    type="C",
                    target=self._target_from_mention(avoided, content),
                    source="user_stated",
                    provenance={
                        "extractor": self.name,
                        "avoided": avoided,
                        "sentence": content,
                        "session_id": session_id,
                    },
                )
            )
            return facts

        # "remember that ..." → V, explicit memory request, target = 'general'
        m = re.search(r"\bremember\s+that\s+(.*)", content, re.IGNORECASE)
        if m:
            recall = m.group(1).rstrip(".").strip()
            facts.append(
                PendingFact(
                    content=f"user wants remembered: {recall}",
                    type="V",
                    target="general",
                    source="user_stated",
                    provenance={
                        "extractor": self.name,
                        "explicit_request": True,
                        "sentence": content,
                        "session_id": session_id,
                    },
                )
            )
            return facts

        return facts

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _is_question(self, content: str) -> bool:
        if content.endswith("?"):
            return True
        first_word = content.split()[0].lower().rstrip("?") if content else ""
        return first_word in _WH_WORDS

    def _infer_target(self, content: str) -> str:
        # If a tool name is mentioned, use it
        tool = self._extract_tool_name(content)
        if tool:
            return tool
        # If a filename is mentioned, use its parent directory
        path = self._extract_file_path(content)
        if path:
            return path.rstrip("/").rsplit("/", 1)[-1] or path
        return "general"

    TOOL_RE = re.compile(
        r"\b(pytest|git|pip|npm|yarn|docker|kubectl|terraform|ansible|"
        r"make|curl|wget|ffmpeg|rg|fd|exa|bat|delta|lazygit|vim?)\b",
        re.IGNORECASE
    )

    def _extract_tool_name(self, content: str) -> Optional[str]:
        m = self.TOOL_RE.search(content)
        return m.group(1).lower() if m else None

    FILE_RE = re.compile(r"[^\s]+(?:\.py|\.toml|\.yaml|\.yml|\.json|\.sh|\.txt)\b")

    def _extract_file_path(self, content: str) -> Optional[str]:
        m = self.FILE_RE.search(content)
        return m.group(0) if m else None

    def _target_from_mention(self, mention: str, content: str) -> str:
        # Check if mention is a tool
        tool = self._extract_tool_name(mention)
        if tool:
            return tool
        # Otherwise use file parent if it's a path
        path = self._extract_file_path(mention)
        if path:
            return path.rstrip("/").rsplit("/", 1)[-1]
        return "general"