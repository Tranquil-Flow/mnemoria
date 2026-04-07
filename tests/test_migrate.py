import json
from pathlib import Path

from mnemoria.config import MnemoriaConfig
from mnemoria.migrate import load_honcho_source_config, migrate_honcho
from mnemoria.store import MnemoriaStore


class FakeConclusion:
    def __init__(self, content, session_id=None):
        self.content = content
        self.session_id = session_id


class FakeMessage:
    def __init__(self, content, peer_id, session_id=None):
        self.content = content
        self.peer_id = peer_id
        self.session_id = session_id


class FakeSession:
    def __init__(self, messages):
        self._messages = messages

    def messages(self, size=100):
        return list(self._messages)


class FakeConclusionScope:
    def __init__(self, conclusions):
        self._conclusions = conclusions

    def list(self, size=100):
        return list(self._conclusions)


class FakePeer:
    def __init__(self, peer_id, conclusions=None, sessions=None):
        self.id = peer_id
        self._conclusions = conclusions or []
        self._sessions = sessions or []

    def conclusions_of(self, target):
        return FakeConclusionScope(self._conclusions)

    def sessions(self, size=100):
        return list(self._sessions)


class FakeHoncho:
    def __init__(self, peers):
        self._peers = peers

    def peer(self, peer_id):
        return self._peers[peer_id]


def make_store(tmp_path):
    cfg = MnemoriaConfig.balanced()
    cfg.db_path = str(tmp_path / "mnemoria.db")
    return MnemoriaStore(cfg)


def test_load_honcho_source_config_reads_host_block(tmp_path, monkeypatch):
    cfg_path = tmp_path / "honcho.json"
    cfg_path.write_text(
        json.dumps(
            {
                "workspace": "default-workspace",
                "peerName": "default-user",
                "aiPeer": "default-ai",
                "hosts": {
                    "hermes": {
                        "workspace": "workspace-1",
                        "peerName": "moon-user",
                        "aiPeer": "moon-ai",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_honcho_source_config(str(cfg_path), host="hermes")
    assert cfg.workspace_id == "workspace-1"
    assert cfg.user_peer == "moon-user"
    assert cfg.ai_peer == "moon-ai"


def test_migrate_honcho_imports_conclusions_only_by_default(tmp_path):
    store = make_store(tmp_path)
    honcho = FakeHoncho(
        {
            "moon-ai": FakePeer(
                "moon-ai",
                conclusions=[
                    FakeConclusion("User prefers dark mode", session_id="sess-1"),
                    FakeConclusion("User wants tested code only", session_id=None),
                ],
            ),
            "moon-user": FakePeer(
                "moon-user",
                sessions=[
                    FakeSession([FakeMessage("hello", peer_id="moon-user", session_id="sess-1")])
                ],
            ),
        }
    )

    result = migrate_honcho(store, honcho, user_peer="moon-user", ai_peer="moon-ai")
    assert result["conclusions"] == 2
    assert result["messages"] == 0

    recalled = store.recall("dark mode", top_k=5)
    assert any("dark mode" in r.fact.content for r in recalled)


def test_migrate_honcho_can_include_user_messages(tmp_path):
    store = make_store(tmp_path)
    honcho = FakeHoncho(
        {
            "moon-ai": FakePeer("moon-ai", conclusions=[]),
            "moon-user": FakePeer(
                "moon-user",
                sessions=[
                    FakeSession(
                        [
                            FakeMessage("I use PostgreSQL in production", peer_id="moon-user", session_id="sess-1"),
                            FakeMessage("Assistant reply", peer_id="moon-ai", session_id="sess-1"),
                        ]
                    )
                ],
            ),
        }
    )

    result = migrate_honcho(
        store,
        honcho,
        user_peer="moon-user",
        ai_peer="moon-ai",
        include_messages=True,
    )
    assert result["messages"] == 1

    recalled = store.recall("PostgreSQL", top_k=5)
    assert any("PostgreSQL" in r.fact.content for r in recalled)
