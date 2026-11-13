"""`swarm_routing` is deprecated in 0.8.0 and removed in 0.9.0 (decision 8)."""

from __future__ import annotations

import importlib
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)  # the tests below re-import and assert on it
    import agentflow.swarm_routing as swarm_routing


def test_importing_swarm_routing_warns_with_the_removal_version_and_the_replacement():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(swarm_routing)

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "importing a module scheduled for removal must say so"
    message = str(deprecations[0].message)
    assert "0.9.0" in message, f"the warning must name the removal version: {message}"
    assert "SupervisorAgent" in message, f"the warning must name the replacement: {message}"


def test_the_deprecated_module_still_works_for_one_minor_release():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = importlib.reload(swarm_routing)

    assert hasattr(module, "DynamicSupervisorAgent"), "deprecation keeps the name working until removal"
