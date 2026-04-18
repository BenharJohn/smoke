from __future__ import annotations

import json
from pathlib import Path

import pytest

from peagle_q.cli import _expand_config_value, _load_config_file


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


def test_load_config_file_fails_clearly_on_unresolved_env_var(monkeypatch) -> None:
    # Use a sentinel name unlikely to exist in the environment, and defensively
    # delete it if a prior run/CI leaked it. On Sol real jobs export PEAGLE_*
    # variables, which would spuriously resolve the placeholder in the original
    # test and cause a false negative.
    monkeypatch.delenv("PEAGLE_UNSET_FOR_TEST_XYZ", raising=False)
    config_path = Path("tests/_config_unresolved_env.json")
    config_path.write_text(
        json.dumps({"dataset": "$PEAGLE_UNSET_FOR_TEST_XYZ", "output_dir": "runs/out"}),
        encoding="utf-8",
    )
    try:
        with pytest.raises(SystemExit, match="unresolved environment variables"):
            _load_config_file(str(config_path))
    finally:
        if config_path.exists():
            config_path.unlink()
