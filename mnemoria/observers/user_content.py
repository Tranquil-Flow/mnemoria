"""UserContentObserver — extract URLs and file paths from user messages.

Complements UserStatementObserver which only fires on preference/identity
patterns. This observer unconditionally extracts URLs and file paths from
any user message.
"""
import re

from . import PendingFact

_URL_RE = re.compile(r"https?://\S+")
_FILE_PATH_RE = re.compile(r"(/[\w./_+-]+\.\w+)")


class UserContentObserver:
    """Extract URLs and file paths from user messages."""

    name: str = "user_content"

    def observe(self, event: dict) -> list[PendingFact]:
        if event.get("kind") != "user_message":
            return []

        content = event.get("payload", {}).get("content", "")
        if not content:
            return []

        session_id = event.get("session_id", "")
        facts: list[PendingFact] = []

        # URLs
        for match in _URL_RE.finditer(content):
            url = match.group(0).rstrip(".,;:)\"'")
            facts.append(PendingFact(
                content=f"V[url]: {url}",
                type="V",
                target="url",
                source="user_stated",
                provenance={"extractor": self.name, "session_id": session_id},
            ))

        # File paths
        for match in _FILE_PATH_RE.finditer(content):
            path = match.group(1)
            if len(path) < 5 or path.count("/") < 1:
                continue
            facts.append(PendingFact(
                content=f"V[file]: {path}",
                type="V",
                target="file",
                source="user_stated",
                provenance={"extractor": self.name, "session_id": session_id},
            ))

        return facts
