"""Sandbox hardening (0.8.0): images that exist, no host secrets, bounded processes, no root.

The mocked tests pin the container settings. The ``real_docker`` tests run the same
settings against a Docker daemon and skip when none is reachable.
"""

from __future__ import annotations

from unittest import mock

import pytest

from agentflow import sandbox as sandbox_module
from agentflow.sandbox import DockerSandbox, SubprocessSandbox

SECRET = "AGENTFLOW_TEST_SECRET"


def _docker_daemon_reachable() -> bool:
    if not sandbox_module._DOCKER_AVAILABLE:
        return False
    try:
        client = sandbox_module.docker.from_env(timeout=3)
        client.ping()
        client.close()
        return True
    except Exception:  # noqa: BLE001 - any failure means "no daemon to test against"
        return False


real_docker = pytest.mark.skipif(not _docker_daemon_reachable(), reason="no reachable Docker daemon")


def test_the_default_cpp_image_is_a_tag_that_exists():
    assert DockerSandbox.DEFAULT_CPP_IMAGE == "gcc:14", (
        "gcc:14-alpine does not exist on Docker Hub (HTTP 404), so every C++ call failed (SBX-1)"
    )


def test_the_module_docstring_imports_from_where_the_names_live():
    assert "from agentflow import sandboxed_tool" not in (sandbox_module.__doc__ or ""), (
        "sandboxed_tool is not exported from agentflow; the documented import failed (SBX-2)"
    )


@pytest.mark.asyncio
async def test_the_subprocess_sandbox_does_not_hand_host_secrets_to_the_code(monkeypatch):
    monkeypatch.setenv(SECRET, "sk-live-do-not-leak")

    result = await SubprocessSandbox().execute_code(
        "python", f"import os; print(os.environ.get({SECRET!r}))", timeout=20
    )

    assert result["stdout"].strip() == "None", (
        "code written by a model must not read the host's API keys from the environment"
    )


@pytest.mark.asyncio
async def test_the_subprocess_sandbox_still_runs_python():
    result = await SubprocessSandbox().execute_code("python", "print(6 * 7)", timeout=20)

    assert result["stdout"].strip() == "42", f"a stripped environment must still run code: {result}"


@pytest.mark.skipif(not sandbox_module._DOCKER_AVAILABLE, reason="Docker SDK not installed")
@pytest.mark.asyncio
async def test_containers_run_unprivileged_with_a_process_cap_and_an_executable_tmp():
    client = mock.MagicMock()
    container = mock.MagicMock()
    container.wait.return_value = {"StatusCode": 0}
    container.logs.side_effect = [b"", b""]
    client.containers.run.return_value = container
    sandbox = DockerSandbox(auto_pull=False)
    sandbox._client = client

    await sandbox.execute_code("cpp", "int main(){}", timeout=5)

    kwargs = client.containers.run.call_args.kwargs
    assert kwargs["user"] == "65534:65534", "code must not run as root inside the container"
    assert kwargs["pids_limit"] == 64, "a fork bomb must hit a process cap"
    assert "exec" in kwargs["tmpfs"]["/tmp"].split(","), (
        "Docker mounts tmpfs noexec by default, so the compiled C++ binary could never run"
    )
    assert kwargs.get("environment") in (None, {}), "no host variables are passed in"


@real_docker
@pytest.mark.asyncio
async def test_a_real_container_runs_as_nobody_with_a_process_cap():
    code = (
        "import os, pathlib\n"
        "print(os.getuid())\n"
        "print(pathlib.Path('/sys/fs/cgroup/pids.max').read_text().strip())\n"
    )
    async with DockerSandbox(auto_pull=False) as sandbox:  # python image only; gcc:14 is large
        result = await sandbox.execute_code("python", code, timeout=60)

    uid, pids = result["stdout"].split()
    assert uid == "65534", f"expected nobody, got uid {uid}: {result}"
    assert pids == "64", f"expected a 64-process cap, got {pids}"


@real_docker
@pytest.mark.asyncio
async def test_a_real_container_can_execute_a_binary_written_to_tmp():
    code = (
        "import shutil, subprocess\n"
        "shutil.copy('/bin/busybox', '/tmp/busybox')\n"
        "print(subprocess.run(['/tmp/busybox', 'echo', 'ran'], capture_output=True, text=True).stdout)\n"
    )
    async with DockerSandbox(auto_pull=False) as sandbox:  # python image only; gcc:14 is large
        result = await sandbox.execute_code("python", code, timeout=60)

    assert result["stdout"].strip() == "ran", (
        f"the C++ path compiles to /tmp and runs from there; it needs an exec tmpfs: {result}"
    )


@real_docker
@pytest.mark.asyncio
async def test_a_real_container_sees_no_host_secrets_and_no_network(monkeypatch):
    monkeypatch.setenv(SECRET, "sk-live-do-not-leak")
    code = (
        "import os, socket\n"
        f"print(os.environ.get({SECRET!r}))\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    print('network')\n"
        "except OSError:\n"
        "    print('no-network')\n"
    )
    async with DockerSandbox(auto_pull=False) as sandbox:  # python image only; gcc:14 is large
        result = await sandbox.execute_code("python", code, timeout=60)

    assert result["stdout"].split() == ["None", "no-network"], result
