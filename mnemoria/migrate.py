from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from mnemoria.config import MnemoriaConfig
from mnemoria.store import MnemoriaStore


DEFAULT_HONCHO_LOCAL_CONFIG = Path.home() / ".hermes" / "honcho.json"
DEFAULT_HONCHO_GLOBAL_CONFIG = Path.home() / ".honcho" / "config.json"


@dataclass
class HonchoSourceConfig:
    host: str = "hermes"
    workspace_id: str = "hermes"
    api_key: str | None = None
    environment: str = "production"
    base_url: str | None = None
    user_peer: str | None = None
    ai_peer: str = "hermes"
    session_strategy: str = "per-directory"
    session_peer_prefix: bool = False


def resolve_honcho_config_path(config_path: str | None = None) -> Path:
    if config_path:
        return Path(config_path).expanduser()
    if DEFAULT_HONCHO_LOCAL_CONFIG.exists():
        return DEFAULT_HONCHO_LOCAL_CONFIG
    return DEFAULT_HONCHO_GLOBAL_CONFIG


def load_honcho_source_config(
    config_path: str | None = None,
    host: str = "hermes",
) -> HonchoSourceConfig:
    path = resolve_honcho_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Honcho config not found at {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    host_block = (raw.get("hosts") or {}).get(host, {})

    workspace_id = host_block.get("workspace") or raw.get("workspace") or host
    api_key = host_block.get("apiKey") or raw.get("apiKey") or os.environ.get("HONCHO_API_KEY")
    environment = host_block.get("environment") or raw.get("environment") or "production"
    base_url = raw.get("baseUrl") or os.environ.get("HONCHO_BASE_URL") or None
    user_peer = host_block.get("peerName") or raw.get("peerName")
    ai_peer = host_block.get("aiPeer") or raw.get("aiPeer") or host
    session_strategy = host_block.get("sessionStrategy") or raw.get("sessionStrategy") or "per-directory"
    session_peer_prefix = bool(host_block.get("sessionPeerPrefix") if host_block.get("sessionPeerPrefix") is not None else raw.get("sessionPeerPrefix", False))

    return HonchoSourceConfig(
        host=host,
        workspace_id=workspace_id,
        api_key=api_key,
        environment=environment,
        base_url=base_url,
        user_peer=user_peer,
        ai_peer=ai_peer,
        session_strategy=session_strategy,
        session_peer_prefix=session_peer_prefix,
    )


def build_honcho_client(source: HonchoSourceConfig):
    try:
        from honcho import Honcho
    except ImportError as e:
        raise ImportError(
            "honcho-ai is required for Honcho migration. Install with: pip install 'honcho-ai>=2.0.1,<3'"
        ) from e

    return Honcho(
        api_key=source.api_key,
        environment=source.environment,
        base_url=source.base_url,
        workspace_id=source.workspace_id,
    )


def _iter_conclusions(scope) -> Iterable[Any]:
    for conclusion in scope.list(size=100):
        yield conclusion


def _iter_session_messages(peer) -> Iterable[Any]:
    for session in peer.sessions(size=100):
        for msg in session.messages(size=100):
            yield msg


def _safe_session_scope(session_id: str | None) -> str:
    if not session_id:
        return "honcho-import"
    sanitized = "".join(ch if ch.isalnum() or ch in "-_:." else "-" for ch in session_id)
    return f"honcho:{sanitized}"


def migrate_honcho(
    store: MnemoriaStore,
    honcho_client,
    *,
    user_peer: str,
    ai_peer: str,
    include_messages: bool = False,
    include_assistant_messages: bool = False,
    max_conclusions: int = 2000,
    max_messages: int = 2000,
) -> dict[str, int]:
    """Migrate high-signal Honcho memory into Mnemoria.

    Default behavior imports only Honcho conclusions about the user from the AI
    peer's perspective. This is intentionally conservative: conclusions are the
    distilled, higher-signal form of Honcho memory.

    Optional message import can also bring over raw session messages, but is off
    by default to avoid polluting Mnemoria with low-signal chat residue.
    """
    migrated_conclusions = 0
    migrated_messages = 0

    observer_peer = honcho_client.peer(ai_peer)
    user_peer_obj = honcho_client.peer(user_peer)

    # Main high-signal path: conclusions the AI peer formed about the user.
    for idx, conclusion in enumerate(_iter_conclusions(observer_peer.conclusions_of(user_peer)), start=1):
        if idx > max_conclusions:
            break
        content = (getattr(conclusion, "content", "") or "").strip()
        if not content:
            continue
        scope = _safe_session_scope(getattr(conclusion, "session_id", None))
        store.store(content, scope=scope, category="factual", importance=0.8)
        migrated_conclusions += 1

    if include_messages:
        seen: set[tuple[str, str, str | None]] = set()
        for idx, msg in enumerate(_iter_session_messages(user_peer_obj), start=1):
            if idx > max_messages:
                break
            content = (getattr(msg, "content", "") or "").strip()
            if not content:
                continue
            peer_id = getattr(msg, "peer_id", None)
            if peer_id != user_peer and not include_assistant_messages:
                continue
            session_id = getattr(msg, "session_id", None)
            dedup_key = (peer_id or "", content, session_id)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            scope = _safe_session_scope(session_id)
            if peer_id == user_peer:
                import_text = content
            else:
                import_text = f"Assistant memory from Honcho session: {content}"
            store.store(import_text, scope=scope, category="factual", importance=0.45)
            migrated_messages += 1

    return {
        "conclusions": migrated_conclusions,
        "messages": migrated_messages,
        "total": migrated_conclusions + migrated_messages,
    }


def migrate_from_honcho_config(
    *,
    db_path: str | None = None,
    honcho_config_path: str | None = None,
    honcho_host: str = "hermes",
    user_peer: str | None = None,
    ai_peer: str | None = None,
    include_messages: bool = False,
    include_assistant_messages: bool = False,
    max_conclusions: int = 2000,
    max_messages: int = 2000,
) -> dict[str, Any]:
    source = load_honcho_source_config(honcho_config_path, host=honcho_host)
    if not source.user_peer and not user_peer:
        raise ValueError(
            "Could not determine Honcho user peer from config. Pass --user-peer explicitly or set peerName in honcho config."
        )

    cfg = MnemoriaConfig.balanced()
    if db_path:
        cfg.db_path = db_path
    store = MnemoriaStore(cfg)
    honcho_client = build_honcho_client(source)

    result = migrate_honcho(
        store,
        honcho_client,
        user_peer=user_peer or source.user_peer,  # type: ignore[arg-type]
        ai_peer=ai_peer or source.ai_peer,
        include_messages=include_messages,
        include_assistant_messages=include_assistant_messages,
        max_conclusions=max_conclusions,
        max_messages=max_messages,
    )
    return {
        "db_path": cfg.db_path,
        "workspace_id": source.workspace_id,
        "user_peer": user_peer or source.user_peer,
        "ai_peer": ai_peer or source.ai_peer,
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Honcho memory into Mnemoria")
    parser.add_argument("--db", dest="db_path", default=None, help="Target Mnemoria SQLite DB path")
    parser.add_argument("--honcho-config", dest="honcho_config_path", default=None, help="Path to honcho.json/config.json")
    parser.add_argument("--honcho-host", default="hermes", help="Honcho host key inside config (default: hermes)")
    parser.add_argument("--user-peer", default=None, help="Override Honcho user peer id / name")
    parser.add_argument("--ai-peer", default=None, help="Override Honcho AI peer id / name")
    parser.add_argument("--include-messages", action="store_true", help="Also import raw user-authored session messages")
    parser.add_argument("--include-assistant-messages", action="store_true", help="When importing messages, include assistant-authored messages too")
    parser.add_argument("--max-conclusions", type=int, default=2000)
    parser.add_argument("--max-messages", type=int, default=2000)
    args = parser.parse_args()

    try:
        result = migrate_from_honcho_config(
            db_path=args.db_path,
            honcho_config_path=args.honcho_config_path,
            honcho_host=args.honcho_host,
            user_peer=args.user_peer,
            ai_peer=args.ai_peer,
            include_messages=args.include_messages,
            include_assistant_messages=args.include_assistant_messages,
            max_conclusions=args.max_conclusions,
            max_messages=args.max_messages,
        )
    except Exception as e:
        raise SystemExit(
            "Honcho migration failed: "
            f"{e}\n"
            "Check your Honcho config/API access, and pass --user-peer explicitly if peerName is not configured."
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
