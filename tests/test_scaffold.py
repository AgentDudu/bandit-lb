"""Baseline scaffold test verifying package structure and importability."""

import bandit_lb
import bandit_lb.algorithms
import bandit_lb.proxy
import bandit_lb.simulator
import bandit_lb.telemetry


def test_package_version() -> None:
    """Verify package version is defined."""
    assert bandit_lb.__version__ == "0.1.0"


def test_subpackages_importable() -> None:
    """Verify all core submodules can be imported cleanly."""
    assert bandit_lb.simulator is not None
    assert bandit_lb.algorithms is not None
    assert bandit_lb.proxy is not None
    assert bandit_lb.telemetry is not None
