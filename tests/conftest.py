import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_data import build  # noqa: E402


@pytest.fixture
def db(tmp_path):
    """A fresh, seeded database for every test."""
    return build(tmp_path / "test.db")
