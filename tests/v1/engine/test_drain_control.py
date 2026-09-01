# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.engine.core import (
    DrainAdmission,
    DrainState,
    EngineCore,
    EngineCoreProc,
)
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.outputs import ModelRunnerOutput


def _bare_engine_core() -> EngineCore:
    core = object.__new__(EngineCore)
    core._drain_state = DrainState.ACCEPTING
    core.log_stats = False
    core.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_logging_iteration_details=False)
    )
    core.scheduler = MagicMock()
    core.scheduler.get_kv_connector.return_value = None
    core.scheduler.get_ec_connector.return_value = None
    core.aborts_queue = MagicMock()
    core.aborts_queue.empty.return_value = True
    return core


def _request(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        pooling_params=None,
        kv_transfer_params=None,
        ec_transfer_params=None,
        abort_immediately=False,
    )


def test_drain_fences_new_admission_and_preserves_running_output_path():
    core = _bare_engine_core()
    running_request = _request("running")
    assert EngineCore.add_request(core, running_request) == DrainAdmission.ACCEPTED
    core.scheduler.add_request.assert_called_once_with(running_request)

    EngineCore.begin_drain(core)
    core.scheduler.set_pause_state.assert_called_once_with(PauseState.PAUSED_NEW)

    waiting_request = _request("waiting")
    assert EngineCore.add_request(core, waiting_request) == DrainAdmission.DRAINING
    core.scheduler.add_request.assert_called_once_with(running_request)

    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    model_output = object()
    model_future: Future[object] = Future()
    model_future.set_result(model_output)
    core.scheduler.has_requests.return_value = True
    core.scheduler.schedule.return_value = scheduler_output
    core.scheduler.get_grammar_bitmask.return_value = None
    core.scheduler.update_from_output.return_value = {0: object()}
    core.model_executor = MagicMock()
    core.model_executor.execute_model.return_value = model_future

    outputs, model_executed = EngineCore.step(core)

    assert outputs == {0: core.scheduler.update_from_output.return_value[0]}
    assert model_executed
    core.scheduler.update_from_output.assert_called_once_with(
        scheduler_output, model_output
    )


def test_drain_keeps_an_existing_real_scheduler_request_running():
    scheduler = create_scheduler(max_num_batched_tokens=16, max_model_len=32)
    existing_request = create_requests(1, num_tokens=32, req_ids=["existing"])[0]
    scheduler.add_request(existing_request)
    assert [request.req_id for request in scheduler.schedule().scheduled_new_reqs] == [
        "existing"
    ]

    core = object.__new__(EngineCore)
    core._drain_state = DrainState.ACCEPTING
    core.scheduler = scheduler
    EngineCore.begin_drain(core)

    assert EngineCore.add_request(core, _request("new")) == DrainAdmission.DRAINING
    continued = scheduler.schedule()

    assert [request.request_id for request in scheduler.running] == ["existing"]
    assert continued.num_scheduled_tokens["existing"] > 0
    assert "new" not in scheduler.requests


@pytest.mark.parametrize(
    (
        "state",
        "counts",
        "unfinished",
        "has_work",
        "quiescent",
        "pending_work",
    ),
    [
        (DrainState.ACCEPTING, (0, 0), 0, False, False, False),
        (DrainState.DRAINING, (1, 0), 1, True, False, False),
        (DrainState.DRAINING, (0, 1), 1, True, False, False),
        (DrainState.DRAINING, (0, 0), 0, True, False, True),
        (DrainState.DRAINING, (0, 0), 0, False, True, False),
    ],
)
def test_drain_snapshot_is_read_only_and_requires_complete_quiescence(
    state: DrainState,
    counts: tuple[int, int],
    unfinished: int,
    has_work: bool,
    quiescent: bool,
    pending_work: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    core = _bare_engine_core()
    core._drain_state = state
    core.scheduler.get_request_counts.return_value = counts
    core.scheduler.get_num_unfinished_requests.return_value = unfinished
    core.scheduler.has_requests.return_value = has_work
    core.scheduler.get_kv_cache_usage.return_value = 0.25
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", MagicMock(return_value=(128, 256))
    )

    snapshot = EngineCore.get_drain_snapshot(core)

    assert snapshot.state == state
    assert (snapshot.running_requests, snapshot.waiting_requests) == counts
    assert snapshot.unfinished_requests == unfinished
    assert snapshot.scheduler_has_work is has_work
    assert snapshot.pending_async_or_connector_work is pending_work
    assert snapshot.kv_cache_usage == 0.25
    assert snapshot.device_free_memory_bytes == 128
    assert snapshot.device_total_memory_bytes == 256
    assert snapshot.quiescent is quiescent
    core.scheduler.add_request.assert_not_called()
    core.scheduler.finish_requests.assert_not_called()
    core.scheduler.set_pause_state.assert_not_called()
    with pytest.raises(AttributeError):
        snapshot.state = DrainState.ACCEPTING


def test_drain_snapshot_reads_live_device_capacity_without_mutating_scheduler(
    monkeypatch: pytest.MonkeyPatch,
):
    core = _bare_engine_core()
    core.scheduler.get_request_counts.return_value = (0, 0)
    core.scheduler.get_num_unfinished_requests.return_value = 0
    core.scheduler.has_requests.return_value = False
    core.scheduler.get_kv_cache_usage.return_value = 0.0
    get_memory_info = MagicMock(return_value=(1024, 4096))
    monkeypatch.setattr(torch.accelerator, "get_memory_info", get_memory_info)

    snapshot = EngineCore.get_drain_snapshot(core)

    assert (snapshot.device_free_memory_bytes, snapshot.device_total_memory_bytes) == (
        1024,
        4096,
    )
    get_memory_info.assert_called_once_with()
    assert core.scheduler.method_calls == [
        call.get_request_counts(),
        call.get_num_unfinished_requests(),
        call.has_requests(),
        call.get_kv_cache_usage(),
    ]


def test_resume_admission_restores_identity_admission_path():
    core = _bare_engine_core()
    EngineCore.begin_drain(core)
    EngineCore.resume_admission(core)

    request = _request("after-resume")
    assert EngineCore.add_request(core, request) == DrainAdmission.ACCEPTED
    core.scheduler.set_pause_state.assert_has_calls(
        [
            call(PauseState.PAUSED_NEW),
            call(PauseState.UNPAUSED),
        ]
    )
    core.scheduler.add_request.assert_called_once_with(request)


def test_inproc_client_returns_drain_non_admission():
    client = object.__new__(InprocClient)
    request = _request("draining")
    prepared_request = object()
    client.engine_core = MagicMock()
    client.engine_core.is_draining.return_value = False
    client.engine_core.preprocess_add_request.return_value = (prepared_request, 0)
    client.engine_core.add_request.return_value = DrainAdmission.DRAINING

    assert client.add_request(request) == DrainAdmission.DRAINING
    client.engine_core.add_request.assert_called_once_with(prepared_request, 0)


def test_inproc_client_drain_does_not_preprocess_request():
    client = object.__new__(InprocClient)
    client.engine_core = MagicMock()
    client.engine_core.is_draining.return_value = True

    assert client.add_request(_request("draining")) == DrainAdmission.DRAINING
    client.engine_core.preprocess_add_request.assert_not_called()
    client.engine_core.add_request.assert_not_called()


def test_multiprocess_drain_rejection_finishes_registered_client_request():
    core = object.__new__(EngineCoreProc)
    core._reject_add_in_shutdown = MagicMock(return_value=False)
    core.add_request = MagicMock(return_value=DrainAdmission.DRAINING)
    core._send_abort_outputs_to_client = MagicMock()
    request = SimpleNamespace(request_id="draining", client_index=3)

    EngineCoreProc._handle_client_request(core, EngineCoreRequestType.ADD, (request, 0))

    core._send_abort_outputs_to_client.assert_called_once_with(["draining"], 3)


def test_llm_engine_drain_rejection_removes_local_output_state():
    engine = object.__new__(LLMEngine)
    engine.input_processor = MagicMock()
    engine.engine_core = MagicMock()
    engine.engine_core.add_request.return_value = DrainAdmission.DRAINING
    engine.output_processor = MagicMock()
    request = EngineCoreRequest(
        request_id="draining",
        prompt_token_ids=[0],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    with pytest.raises(RuntimeError, match="not admitted"):
        engine.add_request("draining", request, SamplingParams(max_tokens=1))

    engine.output_processor.abort_requests.assert_called_once_with(
        ["draining"], internal=True
    )


def test_llm_engine_parallel_sampling_drain_rolls_back_all_children():
    engine = object.__new__(LLMEngine)
    engine.input_processor = MagicMock()
    engine.input_processor.assign_request_id.side_effect = lambda request: setattr(
        request, "external_req_id", request.request_id
    )
    engine.engine_core = MagicMock()
    engine.engine_core.add_request.side_effect = [
        DrainAdmission.ACCEPTED,
        DrainAdmission.DRAINING,
    ]
    engine.output_processor = MagicMock()
    engine.output_processor.abort_requests.return_value = ["0_logical", "1_logical"]
    request = EngineCoreRequest(
        request_id="logical",
        prompt_token_ids=[0],
        mm_features=None,
        sampling_params=SamplingParams(n=2, max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    with pytest.raises(RuntimeError, match="not admitted"):
        engine.add_request("logical", request, SamplingParams(n=2, max_tokens=1))

    assert [
        call.args[0].request_id
        for call in engine.engine_core.add_request.call_args_list
    ] == [
        "0_logical",
        "1_logical",
    ]
    engine.output_processor.abort_requests.assert_called_once_with(
        ["logical"],
        internal=True,
    )
    engine.engine_core.abort_requests.assert_called_once_with(
        ["0_logical", "1_logical"]
    )


def _mixed_waiting_running_scheduler():
    scheduler = create_scheduler(max_num_batched_tokens=16, max_model_len=64)
    running, waiting = create_requests(2, num_tokens=8, req_ids=["running", "waiting"])
    scheduler.add_request(running)
    initial_output = scheduler.schedule()
    scheduler.update_from_output(
        initial_output,
        ModelRunnerOutput(
            req_ids=["running"],
            req_id_to_index={"running": 0},
            sampled_token_ids=[[1]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.add_request(waiting)
    return scheduler


def _scheduler_output_and_kv_signature(scheduler_output):
    return (
        scheduler_output.num_scheduled_tokens,
        [
            (request.req_id, request.block_ids, request.num_computed_tokens)
            for request in scheduler_output.scheduled_new_reqs
        ],
        (
            scheduler_output.scheduled_cached_reqs.req_ids,
            scheduler_output.scheduled_cached_reqs.new_block_ids,
            scheduler_output.scheduled_cached_reqs.num_computed_tokens,
            scheduler_output.scheduled_cached_reqs.num_output_tokens,
        ),
    )


def test_inactive_drain_observation_preserves_mixed_scheduler_and_kv_allocations(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline_scheduler = _mixed_waiting_running_scheduler()
    observed_scheduler = _mixed_waiting_running_scheduler()
    observed_core = object.__new__(EngineCore)
    observed_core._drain_state = DrainState.ACCEPTING
    observed_core.scheduler = observed_scheduler
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", MagicMock(return_value=(128, 256))
    )

    snapshot = EngineCore.get_drain_snapshot(observed_core)
    baseline_output = baseline_scheduler.schedule()
    observed_output = observed_scheduler.schedule()

    assert (snapshot.running_requests, snapshot.waiting_requests) == (1, 1)
    assert snapshot.state == DrainState.ACCEPTING
    assert _scheduler_output_and_kv_signature(
        observed_output
    ) == _scheduler_output_and_kv_signature(baseline_output)
    assert _scheduler_output_and_kv_signature(observed_output) == (
        {"running": 1, "waiting": 8},
        [("waiting", ([2],), 0)],
        (["running"], [None], [8], [1]),
    )
