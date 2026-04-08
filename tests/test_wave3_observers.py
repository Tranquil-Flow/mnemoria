"""Verification test for Wave 3 observers."""
from mnemoria.observers import Observer, PendingFact
from mnemoria.observers.tool_output import PytestObserver, GitObserver, FileObserver
from mnemoria.observers.user_statement import UserStatementObserver


def test_pytest_exit_zero():
    pyo = PytestObserver()
    result = pyo.observe({
        'kind': 'tool_result',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'tool': 'pytest', 'exit_code': 0, 'stdout': 'passed'}
    })
    assert result == [], f"Exit 0 should produce no fact: {result}"


def test_pytest_exit_nonzero_no_failed():
    pyo = PytestObserver()
    result = pyo.observe({
        'kind': 'tool_result',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'tool': 'pytest', 'exit_code': 1, 'stdout': 'ERROR'}
    })
    assert result == [], f"No FAILED keyword → no fact: {result}"


def test_pytest_failing():
    pyo = PytestObserver()
    result = pyo.observe({
        'kind': 'tool_result',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'tool': 'pytest', 'exit_code': 1, 'stdout': 'FAILED test_foo.py::test_bar - AssertionError: expected 1 got 2'}
    })
    # First result is the failing-test fact, second is the retraction of any "tests passing" fact
    assert len(result) == 2, f"Failing test should produce fact + retraction: {result}"
    assert result[0].source == 'observed'
    assert 'tests failing' in result[0].content
    # Second result should be a retraction type D
    assert result[1].type == 'D'


def test_pytest_retraction_emitted():
    pyo = PytestObserver()
    result = pyo.observe({
        'kind': 'tool_result',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'tool': 'pytest', 'exit_code': 1, 'stdout': 'FAILED test_foo.py::test_bar'}
    })
    # Should be two: one fact + one retraction
    assert len(result) == 2, f"Expected 2 facts (fact + retraction): {result}"
    assert result[1].type == 'D', f"Second should be retraction: {result[1]}"


def test_git_observer_rejected_push():
    gio = GitObserver()
    result = gio.observe({
        'kind': 'tool_result',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'tool': 'git', 'exit_code': 1, 'stderr': 'push rejected', 'command': 'git push origin main'}
    })
    assert len(result) == 1, f"Rejected push should produce fact: {result}"
    assert 'push to main rejected' in result[0].content


def test_git_observer_no_noise():
    gio = GitObserver()
    result = gio.observe({
        'kind': 'tool_result',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'tool': 'git', 'exit_code': 0, 'stdout': '?? nothing', 'command': 'git status'}
    })
    assert result == [], f"git status should produce no fact: {result}"


def test_user_statement_preference():
    uso = UserStatementObserver()
    result = uso.observe({
        'kind': 'user_message',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'content': 'I prefer using pip for package management'}
    })
    assert len(result) == 1, f"Preference should produce fact: {result}"
    assert result[0].source == 'user_stated'


def test_user_statement_identity():
    uso = UserStatementObserver()
    result = uso.observe({
        'kind': 'user_message',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'content': 'my email is alice@example.com'}
    })
    assert len(result) == 1
    assert result[0].target == 'identity'
    assert result[0].type == 'C'


def test_user_statement_remember():
    uso = UserStatementObserver()
    result = uso.observe({
        'kind': 'user_message',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'content': 'remember that I always use dark mode'}
    })
    assert len(result) == 1
    assert result[0].source == 'user_stated'


def test_user_statement_question_skipped():
    uso = UserStatementObserver()
    result = uso.observe({
        'kind': 'user_message',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'content': 'do you prefer pip or conda?'}
    })
    assert result == [], f"Question should produce no fact: {result}"


def test_user_statement_wh_question_skipped():
    uso = UserStatementObserver()
    result = uso.observe({
        'kind': 'user_message',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'content': 'What package manager should I use?'}
    })
    assert result == [], f"wh-word question should produce no fact: {result}"


def test_user_statement_dont_use():
    uso = UserStatementObserver()
    result = uso.observe({
        'kind': 'user_message',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'content': "don't use black for formatting"}
    })
    assert len(result) == 1
    assert result[0].type == 'C'


def test_file_observer_noise_paths():
    fo = FileObserver()
    result = fo.observe({
        'kind': 'tool_result',
        'session_id': 's1',
        'timestamp': 1.0,
        'payload': {'tool': 'read_file', 'path': '/home/user/project/.venv/lib/site-packages/foo.py'}
    })
    assert result == [], f"Noise path should produce no fact: {result}"


def test_pending_fact_dataclass():
    pf = PendingFact(
        content="test content",
        type="V",
        target="test_target",
        source="observed",
        provenance={"key": "value"}
    )
    assert pf.content == "test content"
    assert pf.type == "V"
    assert pf.target == "test_target"
    assert pf.source == "observed"
    assert pf.is_retraction is False

    retraction = PendingFact(
        content="retract test",
        type="D",
        target="test_target",
        source="observed",
        provenance={}
    )
    assert retraction.is_retraction is True


def test_observer_protocol():
    # Verify our observers satisfy the Observer protocol
    observers = [PytestObserver(), GitObserver(), FileObserver(), UserStatementObserver()]
    for obs in observers:
        assert hasattr(obs, 'name')
        assert hasattr(obs, 'observe')
        assert callable(obs.observe)


if __name__ == "__main__":
    test_pytest_exit_zero()
    test_pytest_exit_nonzero_no_failed()
    test_pytest_failing()
    test_pytest_retraction_emitted()
    test_git_observer_rejected_push()
    test_git_observer_no_noise()
    test_user_statement_preference()
    test_user_statement_identity()
    test_user_statement_remember()
    test_user_statement_question_skipped()
    test_user_statement_wh_question_skipped()
    test_user_statement_dont_use()
    test_file_observer_noise_paths()
    test_pending_fact_dataclass()
    test_observer_protocol()
    print("Wave 3 verification: PASSED")