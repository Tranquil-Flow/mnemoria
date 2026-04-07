import os
import tempfile

from mnemoria import MnemoriaConfig, MnemoriaStore


def make_store():
    tmpdir = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmpdir.name, "test.db")
    cfg = MnemoriaConfig.balanced()
    cfg.db_path = db_path
    cfg.enable_pressure = False
    store = MnemoriaStore(cfg)
    store.enable_virtual_clock()
    return tmpdir, store


def benchmark_store(store, text):
    store.store(text, category="factual", importance=0.5)
    store.advance_time(0.0001)


def test_reset_prevents_cross_scenario_delegation_leakage():
    tmpdir, store = make_store()
    try:
        # Two prior scenarios on the same store, matching benchmark behavior.
        for task, result, query, expected in [
            (
                "Investigate where production should be deployed.",
                "Recommendation: production region is us-east-1 because latency and tooling are best there.",
                "What region did the delegated deployment investigation recommend?",
                "us-east-1",
            ),
            (
                "Compare database options for the new service.",
                "Conclusion: use PostgreSQL 16 as the primary database and keep Redis only for caching.",
                "What primary database did the delegated comparison choose?",
                "PostgreSQL 16",
            ),
        ]:
            store.reset()
            benchmark_store(store, task)
            benchmark_store(store, result)
            assert expected in store.recall(query, top_k=1)[0].fact.content

        # Regression: this used to return the task prompt instead of the result.
        store.reset()
        benchmark_store(store, "Review admin account security policy.")
        benchmark_store(
            store,
            "Policy check result: all admin accounts must require MFA before production access is granted.",
        )
        top = store.recall(
            "What did the delegated security review say about admin accounts?",
            top_k=1,
        )[0].fact.content
        assert "require MFA" in top
    finally:
        tmpdir.cleanup()



def test_named_answer_beats_unrelated_newer_fact():
    tmpdir, store = make_store()
    try:
        benchmark_store(store, "The founding engineer who designed the auth system is Priya Nair")
        store.simulate_time(29)
        benchmark_store(store, "The on-call rotation was updated to include the new hire last week")
        store.simulate_time(1)

        top = store.recall("Who designed the original auth system?", top_k=1)[0].fact.content
        assert "Priya Nair" in top
    finally:
        tmpdir.cleanup()

