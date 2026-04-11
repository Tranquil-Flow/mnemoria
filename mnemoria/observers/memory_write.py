"""MemoryWriteObserver — mirror built-in memory writes as typed Mnemoria facts."""
import re
from . import PendingFact

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "that",
    "this", "it", "its", "and", "or", "but", "not", "no", "so",
})

def content_slug(text: str, max_words: int = 3) -> str:
    """Generate a short slug from content for target discrimination."""
    if not text:
        return "general"
    words = re.findall(r"[\w.]+", text.lower())
    meaningful = [w for w in words if w not in _STOP_WORDS]
    if not meaningful:
        return "general"
    return "-".join(meaningful[:max_words])

class MemoryWriteObserver:
    """Mirror built-in memory writes (MEMORY.md/USER.md) as typed Mnemoria facts."""
    name: str = "memory_write"

    def observe(self, event: dict) -> list[PendingFact]:
        if event.get("kind") != "memory_write":
            return []
        payload = event.get("payload", {})
        action = payload.get("action", "")
        content = payload.get("content", "").strip()
        target_ns = payload.get("target", "memory")
        if action not in ("add", "replace") or not content:
            return []
        session_id = event.get("session_id", "")
        slug = content_slug(content)
        if target_ns == "user":
            spec = f"V[user.{slug}]: {content}"
        else:
            spec = f"V[memory.{slug}]: {content}"
        return [PendingFact(
            content=spec, type="V", target=f"{target_ns}.{slug}",
            source="observed",
            provenance={"extractor": self.name, "action": action,
                        "original_target": target_ns, "session_id": session_id},
        )]
