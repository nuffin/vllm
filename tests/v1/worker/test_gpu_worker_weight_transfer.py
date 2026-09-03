# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPUWorker weight-transfer and private-provider lifecycles.

The worker no longer contains transport, layerwise, or sparse logic: it only
delegates to the configured weight transfer engine and tracks whether an update
session is active. These tests verify that delegation and the session guard.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.model_executor.layers.fused_moe.expert_residency import Phase4UnsupportedError
from vllm.model_executor.layers.fused_moe.private_wna16_provider import (
    bind_private_wna16_generation_view_provider,
)
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_model_runner import _get_parameter_for_reload
from vllm.v1.worker.gpu_worker import Worker


class _RecordingEngine:
    """Minimal stand-in for a weight transfer engine."""

    def __init__(self, raise_on_update: bool = False):
        self.raise_on_update = raise_on_update
        self.started = False
        self.finished = False
        self.reset_count = 0
        self.supports_draft_weight_update = False
        self.update_calls: list[dict] = []
        self.seen_configs: list[VllmConfig] = []

    def _record_config(self) -> None:
        self.seen_configs.append(get_current_vllm_config())

    def start_weight_update(self) -> None:
        self._record_config()
        self.started = True

    def update_weights(self, update_info: dict) -> None:
        self._record_config()
        self.update_calls.append(update_info)
        if self.raise_on_update:
            raise ValueError("boom")

    def finish_weight_update(self) -> None:
        self._record_config()
        self.finished = True

    def reset_weight_update_target(self) -> None:
        self.reset_count += 1


class _RecordingModelRunner:
    def __init__(self) -> None:
        self.seen_config: VllmConfig | None = None
        self.reset_lora_calls = 0

    def reload_weights(self) -> None:
        self.seen_config = get_current_vllm_config()

    def reset_lora_state(self) -> None:
        self.reset_lora_calls += 1


def _make_worker(engine: _RecordingEngine | None) -> Worker:
    worker = object.__new__(Worker)
    worker.vllm_config = VllmConfig()
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    worker._weight_update_is_draft = False
    worker.model_runner = _RecordingModelRunner()
    return worker


def _make_lifecycle_worker(
    layer_id: int | None, *, weight_transfer_config: object | None = None
) -> Worker:
    worker = object.__new__(Worker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(private_wna16_residency_layer=layer_id),
        weight_transfer_config=weight_transfer_config,
    )
    worker.device = object()
    worker.model_runner = MagicMock()
    worker.weight_transfer_engine = None
    worker.profiler = None
    worker.elastic_ep_executor = MagicMock()
    worker._private_wna16_residency_controller = None
    worker._private_wna16_provider_registration = None
    worker._maybe_get_memory_pool_context = lambda *, tag: nullcontext()
    worker._scoped_allocator_max_split = lambda *, max_split_size_mb: nullcontext()
    return worker


def _assert_private_provider_registry_absent() -> None:
    with pytest.raises(
        Phase4UnsupportedError, match="registered controller provider factory"
    ):
        bind_private_wna16_generation_view_provider(
            layer_id=3, routed_experts=MagicMock()
        )


def test_load_model_without_private_wna16_leaves_provider_unconfigured():
    worker = _make_lifecycle_worker(None)

    with patch.object(
        gpu_worker, "set_current_vllm_config", return_value=nullcontext()
    ):
        Worker.load_model(worker, load_dummy_weights=True)

    worker.model_runner.load_model.assert_called_once_with(load_dummy_weights=True)
    assert worker._private_wna16_residency_controller is None
    assert worker._private_wna16_provider_registration is None
    _assert_private_provider_registry_absent()


def test_load_model_closes_private_provider_after_runner_base_exception():
    worker = _make_lifecycle_worker(3)
    worker.model_runner.load_model.side_effect = SystemExit("runner interrupted")

    with (
        patch.object(gpu_worker, "set_current_vllm_config", return_value=nullcontext()),
        pytest.raises(SystemExit, match="runner interrupted"),
    ):
        Worker.load_model(worker)

    assert worker._private_wna16_residency_controller is None
    assert worker._private_wna16_provider_registration is None
    _assert_private_provider_registry_absent()


def test_load_model_closes_private_provider_after_weight_transfer_failure():
    worker = _make_lifecycle_worker(3, weight_transfer_config=object())
    failure = RuntimeError("weight transfer setup failed")

    with (
        patch.object(gpu_worker, "set_current_vllm_config", return_value=nullcontext()),
        patch.object(
            gpu_worker.WeightTransferEngineFactory,
            "create_engine",
            side_effect=failure,
        ),
        pytest.raises(RuntimeError, match="weight transfer setup failed"),
    ):
        Worker.load_model(worker)

    worker.model_runner.load_model.assert_called_once_with(load_dummy_weights=False)
    assert worker._private_wna16_residency_controller is None
    assert worker._private_wna16_provider_registration is None
    assert worker.weight_transfer_engine is None
    _assert_private_provider_registry_absent()


def test_repeated_private_wna16_load_keeps_first_registration_until_shutdown():
    worker = _make_lifecycle_worker(3)

    with patch.object(
        gpu_worker, "set_current_vllm_config", return_value=nullcontext()
    ):
        Worker.load_model(worker)
        registration = worker._private_wna16_provider_registration
        controller = worker._private_wna16_residency_controller
        with pytest.raises(RuntimeError, match="already registered for this worker"):
            Worker.load_model(worker)

    assert worker._private_wna16_provider_registration is registration
    assert worker._private_wna16_residency_controller is controller
    worker.model_runner.load_model.assert_called_once_with(load_dummy_weights=False)

    with (
        patch.object(gpu_worker, "ensure_kv_transfer_shutdown", None),
        patch.object(gpu_worker, "ensure_ec_transfer_shutdown", None),
        patch.object(
            gpu_worker,
            "current_platform",
            SimpleNamespace(is_cuda_alike=lambda: False),
        ),
    ):
        Worker.shutdown(worker)
    _assert_private_provider_registry_absent()


def test_shutdown_is_idempotent_after_private_wna16_load():
    worker = _make_lifecycle_worker(3)
    profiler = MagicMock()
    elastic_executor = MagicMock()
    model_runner = MagicMock()
    worker.profiler = profiler
    worker.elastic_ep_executor = elastic_executor
    worker.model_runner = model_runner

    with patch.object(
        gpu_worker, "set_current_vllm_config", return_value=nullcontext()
    ):
        Worker.load_model(worker)

    with (
        patch.object(gpu_worker, "ensure_kv_transfer_shutdown", None),
        patch.object(gpu_worker, "ensure_ec_transfer_shutdown", None),
        patch.object(
            gpu_worker,
            "current_platform",
            SimpleNamespace(is_cuda_alike=lambda: False),
        ),
    ):
        Worker.shutdown(worker)
        Worker.shutdown(worker)

    profiler.shutdown.assert_called_once_with()
    elastic_executor.shutdown.assert_called_once_with()
    model_runner.shutdown.assert_called_once_with()
    assert worker._private_wna16_residency_controller is None
    assert worker._private_wna16_provider_registration is None
    _assert_private_provider_registry_absent()


def test_shutdown_retries_only_the_resource_that_failed_teardown():
    worker = _make_lifecycle_worker(3)
    profiler = MagicMock()
    weight_transfer_engine = MagicMock()
    elastic_executor = MagicMock()
    model_runner = MagicMock()
    worker.profiler = profiler
    worker.weight_transfer_engine = weight_transfer_engine
    worker.elastic_ep_executor = elastic_executor
    worker.model_runner = model_runner
    elastic_executor.shutdown.side_effect = RuntimeError("elastic teardown failed")

    with patch.object(
        gpu_worker, "set_current_vllm_config", return_value=nullcontext()
    ):
        Worker.load_model(worker)

    with (
        patch.object(gpu_worker, "ensure_kv_transfer_shutdown", None),
        patch.object(gpu_worker, "ensure_ec_transfer_shutdown", None),
        patch.object(
            gpu_worker,
            "current_platform",
            SimpleNamespace(is_cuda_alike=lambda: False),
        ),
        pytest.raises(RuntimeError, match="elastic teardown failed"),
    ):
        Worker.shutdown(worker)

    assert getattr(worker, "_shutdown_complete", False) is False
    _assert_private_provider_registry_absent()
    profiler.shutdown.assert_called_once_with()
    weight_transfer_engine.shutdown.assert_called_once_with()
    elastic_executor.shutdown.assert_called_once_with()
    model_runner.shutdown.assert_not_called()

    elastic_executor.shutdown.side_effect = None
    with (
        patch.object(gpu_worker, "ensure_kv_transfer_shutdown", None),
        patch.object(gpu_worker, "ensure_ec_transfer_shutdown", None),
        patch.object(
            gpu_worker,
            "current_platform",
            SimpleNamespace(is_cuda_alike=lambda: False),
        ),
    ):
        Worker.shutdown(worker)
        Worker.shutdown(worker)

    profiler.shutdown.assert_called_once_with()
    weight_transfer_engine.shutdown.assert_called_once_with()
    assert elastic_executor.shutdown.call_count == 2
    model_runner.shutdown.assert_called_once_with()
    assert worker._shutdown_complete is True


def test_reload_weights_sets_current_config():
    worker = _make_worker(None)
    model_runner = _RecordingModelRunner()
    worker.model_runner = model_runner  # type: ignore[assignment]

    Worker.reload_weights(worker)

    assert model_runner.seen_config is worker.vllm_config


def test_reload_parameter_lookup_preserves_lora_module_names():
    base_layer = nn.Module()
    qweight = nn.Parameter(torch.ones(1))
    base_layer.register_parameter("qweight", qweight)
    wrapper = BaseLayerWithLoRA()
    wrapper.base_layer = base_layer
    model = nn.Module()
    model.proj = wrapper

    named_parameters = dict(model.named_parameters())
    assert set(named_parameters) == {"proj.base_layer.qweight"}
    assert named_parameters["proj.base_layer.qweight"] is qweight
    assert model.get_parameter("proj.base_layer.qweight") is qweight
    assert _get_parameter_for_reload(model, "proj.qweight") is qweight


def test_start_update_finish_delegates_to_engine():
    engine = _RecordingEngine()
    worker = _make_worker(engine)

    Worker.start_weight_update(worker)
    assert engine.started is True
    assert worker._weight_update_active is True

    Worker.update_weights(worker, {"names": ["w"]})
    assert engine.update_calls == [{"names": ["w"]}]
    assert worker._weight_update_active is True

    Worker.finish_weight_update(worker)
    assert engine.finished is True
    assert engine.reset_count == 1
    assert worker._weight_update_active is False
    assert engine.seen_configs == [worker.vllm_config] * 3
    assert worker.model_runner.reset_lora_calls == 1


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(1, {"names": ["rank-1"]}), (2, {"names": []})],
)
def test_rank_local_update_selects_worker_payload(rank, expected):
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    worker.rank = rank
    Worker.start_weight_update(worker)

    Worker.update_weights(
        worker, [{"names": ["rank-0"]}, {"names": ["rank-1"]}, {"names": []}]
    )

    assert engine.update_calls == [expected]
    assert worker._weight_update_active is True


def test_rank_local_update_includes_data_parallel_rank():
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    worker.rank = 0
    worker.vllm_config.parallel_config.data_parallel_size = 4
    worker.vllm_config.parallel_config.data_parallel_rank = 2
    Worker.start_weight_update(worker)

    Worker.update_weights(
        worker,
        [
            {"names": ["dp-0"]},
            {"names": ["dp-1"]},
            {"names": ["dp-2"]},
            {"names": ["dp-3"]},
        ],
    )

    assert engine.update_calls == [{"names": ["dp-2"]}]
    assert worker._weight_update_active is True


def test_finish_draft_session_keeps_lora_state():
    engine = _RecordingEngine()
    engine.supports_draft_weight_update = True
    worker = _make_worker(engine)
    worker._set_draft_weight_update_target = lambda: None

    Worker.start_draft_weight_update(worker)
    Worker.finish_weight_update(worker)

    assert worker.model_runner.reset_lora_calls == 0


def test_double_start_raises():
    worker = _make_worker(_RecordingEngine())
    Worker.start_weight_update(worker)
    with pytest.raises(RuntimeError, match="already"):
        Worker.start_weight_update(worker)


def test_update_without_start_raises():
    worker = _make_worker(_RecordingEngine())
    with pytest.raises(RuntimeError, match="start_weight_update must be called"):
        Worker.update_weights(worker, {"names": ["w"]})


def test_finish_without_start_raises():
    worker = _make_worker(_RecordingEngine())
    with pytest.raises(RuntimeError, match="without a matching"):
        Worker.finish_weight_update(worker)


def test_update_resets_active_on_error():
    engine = _RecordingEngine(raise_on_update=True)
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)

    with pytest.raises(ValueError, match="boom"):
        Worker.update_weights(worker, {"names": ["w"]})

    # A failed update ends the session so the next start is clean.
    assert engine.reset_count == 1
    assert worker._weight_update_active is False


def test_missing_engine_raises():
    worker = _make_worker(None)
    with pytest.raises(RuntimeError, match="Weight transfer not configured"):
        Worker.start_weight_update(worker)
