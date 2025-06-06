import re

from queensgym import __version__


def test_version_format() -> None:
    assert re.match("\d+.\d+.\d+", __version__) is not None
