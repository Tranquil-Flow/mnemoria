"""DelegationObserver — store delegation outcomes and extract from tool traces."""
from . import PendingFact

class DelegationObserver:
    """Extract facts from delegation (subagent) outcomes."""
    name: str = "delegation"

    def observe(self, event: dict) -> list[PendingFact]:
        if event.get("kind") != "delegation":
            return []
        payload = event.get("payload", {})
        task = (payload.get("task", "") or "")[:200].strip()
        result = (payload.get("result", "") or "")[:200].strip()
        if not task and not result:
            return []
        session_id = event.get("session_id", "")
        child_session_id = payload.get("child_session_id", "")
        facts: list[PendingFact] = []
        facts.append(PendingFact(
            content=f"D[delegation]: {task} -> {result}",
            type="D", target="delegation", source="observed",
            provenance={"extractor": self.name, "child_session_id": child_session_id,
                        "session_id": session_id},
        ))
        tool_trace = payload.get("tool_trace")
        if tool_trace and isinstance(tool_trace, list):
            for entry in tool_trace:
                if not isinstance(entry, dict):
                    continue
                tool_name = entry.get("tool", "unknown")
                success = entry.get("success")
                action = entry.get("action", "")
                status = "success" if success else "failed" if success is False else "unknown"
                desc = f"{tool_name} {action}".strip() if action else tool_name
                facts.append(PendingFact(
                    content=f"subagent used {desc} ({status})",
                    type="V", target="delegation", source="observed",
                    provenance={"extractor": self.name, "tool": tool_name,
                                "success": success, "session_id": session_id},
                ))
        return facts
