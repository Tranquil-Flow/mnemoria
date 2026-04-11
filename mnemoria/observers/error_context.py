"""ErrorContextObserver — extract generic errors, URLs and paths near errors.

Handles tool_result events. Covers extraction that PytestObserver/GitObserver
don't handle: generic error lines, and URLs/file paths within 3 lines of errors.
"""
import re
from typing import List

from . import PendingFact

_ERROR_RE = re.compile(
    r"(?:error|failed|exception|traceback|assert(?:ion)?error|FAILED)",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+")
_FILE_PATH_RE = re.compile(r"(/[\w./_+-]+\.\w+)")


class ErrorContextObserver:
    """Extract error lines, URLs near errors, and file paths near errors from tool output."""

    name: str = "error_context"

    def observe(self, event: dict) -> list[PendingFact]:
        if event.get("kind") != "tool_result":
            return []

        payload = event.get("payload", {})
        text = payload.get("stdout", "") or payload.get("output", "")
        if not text:
            return []

        lines = text.splitlines()
        error_line_indices = {i for i, line in enumerate(lines) if _ERROR_RE.search(line)}
        if not error_line_indices:
            return []

        session_id = event.get("session_id", "")
        facts: list[PendingFact] = []

        # 1. First error line as ?[error] fact
        first_error_idx = min(error_line_indices)
        summary = lines[first_error_idx].strip()[:200]
        facts.append(PendingFact(
            content=f"?[error]: {summary}",
            type="?",
            target="error",
            source="observed",
            provenance={"extractor": self.name, "session_id": session_id},
        ))

        # 2. URLs within 3 lines of any error
        seen_urls: set = set()
        for i, line in enumerate(lines):
            if not any(abs(i - e) <= 3 for e in error_line_indices):
                continue
            for match in _URL_RE.finditer(line):
                url = match.group(0).rstrip(".,;:)\"'")
                if url not in seen_urls:
                    seen_urls.add(url)
                    facts.append(PendingFact(
                        content=f"V[url]: {url}",
                        type="V",
                        target="url",
                        source="observed",
                        provenance={"extractor": self.name, "near_error": True,
                                    "session_id": session_id},
                    ))

        # 3. File paths within 3 lines of any error
        seen_paths: set = set()
        for i, line in enumerate(lines):
            if not any(abs(i - e) <= 3 for e in error_line_indices):
                continue
            for match in _FILE_PATH_RE.finditer(line):
                path = match.group(1)
                if len(path) < 5 or path.count("/") < 1:
                    continue
                if path not in seen_paths:
                    seen_paths.add(path)
                    facts.append(PendingFact(
                        content=f"V[file]: {path}",
                        type="V",
                        target="file",
                        source="observed",
                        provenance={"extractor": self.name, "near_error": True,
                                    "session_id": session_id},
                    ))

        return facts
