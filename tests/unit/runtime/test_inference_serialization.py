"""A request cache cannot be shared by concurrent calls on one facade."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from helm.runtime.inference import HelmInference, HelmInferenceConfig


@pytest.mark.parametrize("operation", ["generate", "teardown"])
def test_lifecycle_waits_for_inflight_request(monkeypatch, operation):
    monkeypatch.setattr("helm.runtime.inference._free_gpu", lambda: None)
    facade = HelmInference(HelmInferenceConfig(model_id="local"))
    entered, release, attempted, next_entered = (Event() for _ in range(4))

    def generate(prompts, count):
        if prompts == ["first"]:
            entered.set()
            assert release.wait(5), "test failed to release first request"
        else:
            next_entered.set()
        return prompts

    facade._handle = SimpleNamespace(generate=generate, teardown=next_entered.set)

    def second():
        attempted.set()
        if operation == "generate":
            return facade.generate(["second"])
        facade.teardown()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(facade.generate, ["first"])
        try:
            assert entered.wait(5)
            later = pool.submit(second)
            assert attempted.wait(5)
            assert not next_entered.wait(0.1)
        finally:
            release.set()
        assert first.result(timeout=5) == ["first"]
        later.result(timeout=5)
        assert next_entered.is_set()
    facade.teardown()
