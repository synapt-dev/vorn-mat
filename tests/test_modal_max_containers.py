"""Modal function binding declares max_containers=10.

Layne directive 2026-05-23: account supports 10 concurrent GPUs per profile;
the canonical Modal function decoration should declare that ceiling so the
server-side scheduler fans out up to 10 invocations concurrently when
.spawn() is used per cell. This test pins the contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import build_vanilla_entrypoint, build_vorn_entrypoint


class _FakeApp:
    def __init__(self, name: str):
        self.name = name
        self.last_kwargs: dict | None = None

    def function(self, **kwargs):
        self.last_kwargs = kwargs

        def decorator(fn):
            fn._modal_kwargs = kwargs
            return fn

        return decorator


class _FakeImage:
    def __init__(self, python_version: str):
        self.python_version = python_version
        self.packages: tuple[str, ...] = ()

    def pip_install(self, *packages: str):
        self.packages = packages
        return self


class _FakeDockerfileImage:
    def __init__(self, dockerfile_path: str, context_dir: str):
        self.dockerfile_path = dockerfile_path
        self.context_dir = context_dir
        self.env_vars: dict[str, str] = {}

    def env(self, env_dict):
        self.env_vars = dict(env_dict)
        return self


class _FakeImageFactory:
    @staticmethod
    def debian_slim(*, python_version: str):
        return _FakeImage(python_version)

    @staticmethod
    def from_dockerfile(path: str, *, context_dir: str):
        return _FakeDockerfileImage(path, context_dir)


class _FakeVolumeFactory:
    @staticmethod
    def from_name(name: str, create_if_missing: bool):
        return {"name": name, "create_if_missing": create_if_missing}


class _FakeSecretFactory:
    @staticmethod
    def from_name(name: str):
        return {"secret_name": name}


class _FakeModal:
    App = _FakeApp
    Image = _FakeImageFactory
    Volume = _FakeVolumeFactory
    Secret = _FakeSecretFactory


def test_vanilla_entrypoint_function_declares_max_containers_10():
    binding = build_vanilla_entrypoint(lambda r: r, modal_module=_FakeModal)

    assert binding.app.last_kwargs is not None
    assert binding.app.last_kwargs["max_containers"] == 10


def test_vorn_entrypoint_function_declares_max_containers_10():
    binding = build_vorn_entrypoint(lambda r: r, modal_module=_FakeModal)

    assert binding.app.last_kwargs is not None
    assert binding.app.last_kwargs["max_containers"] == 10
