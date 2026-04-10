from __future__ import annotations

from peagle_q.cli import _expand_config_value


def test_expand_config_value_recurses_through_nested_values(monkeypatch) -> None:
    monkeypatch.setenv("PEAGLE_TEST_PATH", "/tmp/peagle")
    payload = {
        "path": "$PEAGLE_TEST_PATH/run",
        "nested": {
            "items": ["$PEAGLE_TEST_PATH/a", "$PEAGLE_TEST_PATH/b"],
        },
    }

    expanded = _expand_config_value(payload)

    assert expanded == {
        "path": "/tmp/peagle/run",
        "nested": {
            "items": ["/tmp/peagle/a", "/tmp/peagle/b"],
        },
    }
