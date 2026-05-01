"""Tests for the relative-date helpers in run_slice.py.

Tested in isolation because the slice runner is loaded by file path (it
isn't a real package) — see scripts/run_full_locomo_with_dates.py for the
same import pattern.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SLICE_PY = _HERE / "run_slice.py"
_spec = importlib.util.spec_from_file_location("eval_slice_run", _SLICE_PY)
_rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rs)


# ── _parse_session_date ──────────────────────────────────────────────────


def test_parse_pm_date():
    assert _rs._parse_session_date("1:56 pm on 8 May, 2023") == date(2023, 5, 8)


def test_parse_am_date():
    assert _rs._parse_session_date("10:37 am on 27 June, 2023") == date(2023, 6, 27)


def test_parse_no_time():
    assert _rs._parse_session_date("8 May, 2023") == date(2023, 5, 8)


def test_parse_garbage_returns_none():
    assert _rs._parse_session_date("not a date") is None
    assert _rs._parse_session_date("") is None


# ── _resolve_relative_dates ──────────────────────────────────────────────


def test_resolve_yesterday():
    out = _rs._resolve_relative_dates(
        "I went to the support group yesterday.",
        date(2023, 5, 8),
    )
    assert "7 May 2023" in out


def test_resolve_last_year():
    out = _rs._resolve_relative_dates(
        "I painted that lake sunrise last year.",
        date(2023, 5, 25),
    )
    assert "2022" in out


def test_resolve_x_years_ago():
    out = _rs._resolve_relative_dates(
        "It was made for my 18th birthday ten years ago.",
        date(2023, 6, 27),
    )
    assert "2013" in out
    assert "10 years ago" in out


def test_resolve_two_days_ago():
    out = _rs._resolve_relative_dates(
        "I went to a conference two days ago.",
        date(2023, 6, 9),
    )
    assert "7 June 2023" in out


def test_resolve_last_sunday():
    # 25 May 2023 is a Thursday, last Sunday = 21 May 2023
    out = _rs._resolve_relative_dates(
        "We met last Sunday.",
        date(2023, 5, 25),
    )
    assert "21 May 2023" in out


def test_resolve_this_month():
    out = _rs._resolve_relative_dates(
        "I'm going to a transgender conference this month.",
        date(2023, 7, 3),
    )
    assert "July 2023" in out


def test_resolve_no_marker_returns_empty():
    out = _rs._resolve_relative_dates(
        "I have known these friends for 4 years, since I moved.",
        date(2023, 5, 25),
    )
    # "for 4 years" without "ago" is not a relative-time anchor.
    assert out == []


def test_resolve_handles_none_session():
    out = _rs._resolve_relative_dates("yesterday", None)
    assert out == []


def test_resolve_dedupes():
    out = _rs._resolve_relative_dates(
        "Last year I went there. It was last year, what a year.",
        date(2023, 5, 25),
    )
    # "last year" appears multiple times but the resolver should output
    # "2022" only once.
    assert out.count("2022") == 1
