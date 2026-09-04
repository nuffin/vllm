# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from threading import Event, Thread
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_residency import (
    Phase4FailureCategory,
    Phase4GpuResidencyAdapter,
    Phase4UnsupportedError,
    PostConversionExpertBundle,
    TorchCpuTransferReference,
    WNA16ExpertBundle,
    WNA16GenerationView,
    WNA16UseLease,
    _bind_controller_wna16_slot_storage,
    _new_controller_wna16_stable_slot_capability,
    _private_wna16_dispatch_operands,
    validate_wna16_generation_view,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelModularImpl,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend
from vllm.model_executor.layers.fused_moe.private_wna16_provider import (
    PrivateWNA16ProviderBindingRequest,
    PrivateWNA16ResidencyController,
    _WNA16RequestUseState,
    _WNA16SlotPlanStatus,
    _WNA16StagedCudaSlot,
    acquire_private_wna16_request_use,
    begin_private_wna16_request_enqueue,
    bind_private_wna16_generation_view_provider,
    capture_private_wna16_post_conversion,
    mark_private_wna16_request_dispatched,
    register_private_wna16_provider_factory,
    release_private_wna16_request_use_pending,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna3 import (  # noqa: E501
    CompressedTensorsWNA16RDNA3MoEMethod,
)


def complete_bundle() -> PostConversionExpertBundle:
    tensors = {
        "gate_proj": torch.ones(2, 2),
        "up_proj": torch.ones(2, 2),
        "down_proj": torch.ones(2, 2),
    }
    return PostConversionExpertBundle(
        schema_version=1,
        tensors=tensors,
        metadata={
            "weight_shapes": {name: tuple(t.shape) for name, t in tensors.items()},
            "quantization": "WNA16",
        },
    )


def mocked_routed_experts(adapter=None):
    layer = object.__new__(RoutedExperts)
    layer._phase4_residency = adapter
    layer._private_wna16_generation_view_provider = None
    layer.quant_method = MagicMock(is_monolithic=False)
    layer.quant_method.apply.return_value = torch.tensor([1.0])
    layer.expert_map_manager = MagicMock()
    layer.global_num_experts = 4
    return layer


def private_view() -> WNA16GenerationView:
    bundle = WNA16ExpertBundle(
        schema_version=1,
        backend="triton",
        layer_id=3,
        global_num_experts=4,
        slot_count=2,
        generation=9,
        quant_type="W4A16",
        num_bits=4,
        symmetric=True,
        group_size=32,
        act_order=False,
        w13=torch.zeros((2, 64, 16), dtype=torch.uint8),
        w2=torch.zeros((2, 32, 16), dtype=torch.uint8),
        w13_scale=torch.ones((2, 64, 1)),
        w2_scale=torch.ones((2, 32, 1)),
    )
    lease = WNA16UseLease(layer_id=3, generation=9, bundle=bundle, token=1)
    view = WNA16GenerationView(
        bundle,
        torch.tensor([0, -1, 1, -1], dtype=torch.int32),
        9,
        lease,
    )
    return _bind_controller_wna16_slot_storage(
        view, _new_controller_wna16_stable_slot_capability()
    )


def canonical_cpu_bank_layer(*, asymmetric: bool = False):
    layer = SimpleNamespace(
        global_num_experts=4,
        use_ep=False,
        w13_weight_packed=torch.arange(8, dtype=torch.uint8).reshape(2, 4),
        w2_weight_packed=torch.arange(8, 16, dtype=torch.uint8).reshape(2, 4),
        w13_weight_scale=torch.ones(2, 1),
        w2_weight_scale=torch.full((2, 1), 2.0),
        w13_weight_zero_point=torch.zeros(2, 1) if asymmetric else None,
        w2_weight_zero_point=torch.ones(2, 1) if asymmetric else None,
    )
    layer.w13_weight = layer.w13_weight_packed
    layer.w2_weight = layer.w2_weight_packed
    return layer


def _capture_canonical_cpu_bank(controller, layer, *, symmetric=True):
    controller(PrivateWNA16ProviderBindingRequest(layer_id=3, routed_experts=layer))
    capture_private_wna16_post_conversion(
        layer=layer,
        backend=WNA16MoEBackend.TRITON,
        num_bits=4,
        symmetric=symmetric,
        group_size=32,
        act_order=False,
    )


def staged_cpu_slot(controller, *, expert_id=0, map_device="cpu"):
    """Complete CPU fake for the private CUDA-slot staging contract."""
    bank = controller._canonical_cpu_bank
    assert bank is not None
    slots = len(controller._slots)
    slot_map = torch.full(
        (bank.global_num_experts,), -1, dtype=torch.int32, device=map_device
    )
    if map_device == "cpu":
        slot_map[expert_id] = 0

    def destination(source):
        return torch.empty((slots, *source.shape[1:]), dtype=source.dtype)

    return _WNA16StagedCudaSlot(
        destination(bank.w13_weight_packed),
        destination(bank.w2_weight_packed),
        destination(bank.w13_weight_scale),
        destination(bank.w2_weight_scale),
        destination(bank.w13_weight_zero_point)
        if bank.w13_weight_zero_point is not None
        else None,
        destination(bank.w2_weight_zero_point)
        if bank.w2_weight_zero_point is not None
        else None,
        slot_map,
    )


def unbound_private_view() -> WNA16GenerationView:
    """A public ABI view has snapshots but cannot issue private storage."""
    view = private_view()
    object.__setattr__(view, "_stable_slot_storage", None)
    return view


def _request_use_dispatch_target(kernel: MagicMock, factory: Callable | None = None):
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    routed_experts = mocked_routed_experts(adapter)
    if factory is None:
        factory = cast(Callable, lambda _: lambda **_: unbound_private_view())
    registration = register_private_wna16_provider_factory(factory)
    bind_private_wna16_generation_view_provider(
        layer_id=3, routed_experts=routed_experts
    )
    view = routed_experts.get_private_wna16_generation_view(
        topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
        topk_weights=torch.tensor([[0.25, 0.75]]),
    )
    assert view is not None
    method = SimpleNamespace(
        is_monolithic=False,
        moe_kernel=kernel,
        wna16_backend=WNA16MoEBackend.TRITON,
        num_bits=4,
        symmetric=True,
        group_size=32,
        actorder=None,
    )
    layer = SimpleNamespace(
        use_ep=False,
        activation=MagicMock(),
        global_num_experts=4,
        apply_router_weight_on_input=False,
    )
    return registration, view, method, layer


class _VerifiedShutdownDrainFactory:
    """CPU-only registered controller double with exact-lease verification."""

    def __init__(self) -> None:
        self.expected_lease: object | None = None
        self.verifier_calls: list[object] = []

    def __call__(self, _):
        return lambda **_: unbound_private_view()

    def verify_shutdown_drain(self, lease) -> bool:
        self.verifier_calls.append(lease)
        return lease is self.expected_lease


def _quarantine_request_use(registration, view):
    lease = acquire_private_wna16_request_use(view)
    assert lease is not None
    mark_private_wna16_request_dispatched(lease)
    begin_private_wna16_request_enqueue(lease)
    lease.manager.quarantine_unfenced(lease)
    with pytest.raises(Phase4UnsupportedError) as error:
        registration.close()
    assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
    return lease


def _apply_private_request_use(method, layer, view) -> object:
    return CompressedTensorsWNA16MoEMethod.apply(
        method,
        layer=layer,
        x=torch.ones(1, 2),
        topk_weights=torch.ones(1, 2),
        topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=None,
        generation_view=view,
    )


def test_canonical_cpu_bank_is_detached_private_and_one_shot():
    controller = PrivateWNA16ResidencyController(layer_id=3)
    registration = register_private_wna16_provider_factory(controller)
    layer = canonical_cpu_bank_layer()
    try:
        _capture_canonical_cpu_bank(controller, layer)

        bank = controller._canonical_cpu_bank
        assert bank is not None
        assert bank.layer_id == 3
        assert bank.global_num_experts == 4
        assert bank.w13_weight_packed.equal(layer.w13_weight_packed)
        assert bank.w13_weight_packed.data_ptr() != layer.w13_weight_packed.data_ptr()
        assert bank.w13_weight_zero_point is None
        layer.w13_weight_packed.zero_()
        assert bank.w13_weight_packed.sum().item() > 0

        with pytest.raises(Phase4UnsupportedError, match="unsupported"):
            _capture_canonical_cpu_bank(controller, layer)
        assert controller._canonical_cpu_bank is bank
    finally:
        registration.close()


def test_canonical_cpu_bank_ignores_non_target_layers_and_copies_zero_points():
    controller = PrivateWNA16ResidencyController(layer_id=3)
    registration = register_private_wna16_provider_factory(controller)
    layer = canonical_cpu_bank_layer(asymmetric=True)
    wrong_layer = canonical_cpu_bank_layer(asymmetric=True)
    try:
        capture_private_wna16_post_conversion(
            layer=cast(RoutedExperts, wrong_layer),
            backend=WNA16MoEBackend.TRITON,
            num_bits=4,
            symmetric=False,
            group_size=32,
            act_order=False,
        )
        assert controller._canonical_cpu_bank is None

        _capture_canonical_cpu_bank(controller, layer, symmetric=False)

        bank = controller._canonical_cpu_bank
        assert bank is not None
        assert bank.w13_weight_zero_point is not None
        assert bank.w2_weight_zero_point is not None
        assert (
            bank.w13_weight_zero_point.data_ptr()
            != layer.w13_weight_zero_point.data_ptr()
        )
        assert controller._canonical_cpu_bank is bank
    finally:
        registration.close()


def test_canonical_cpu_bank_capture_is_noop_without_registration():
    layer = canonical_cpu_bank_layer()
    original = layer.w13_weight_packed.clone()

    capture_private_wna16_post_conversion(
        layer=layer,
        backend=WNA16MoEBackend.TRITON,
        num_bits=4,
        symmetric=True,
        group_size=32,
        act_order=False,
    )

    assert layer.w13_weight_packed.equal(original)
    assert layer.w13_weight is layer.w13_weight_packed


def test_canonical_cpu_bank_close_drops_private_references():
    controller = PrivateWNA16ResidencyController(layer_id=3)
    registration = register_private_wna16_provider_factory(controller)
    layer = canonical_cpu_bank_layer()
    _capture_canonical_cpu_bank(controller, layer)

    registration.close()

    assert controller._canonical_cpu_bank is None
    with pytest.raises(Phase4UnsupportedError, match="stale"):
        controller.capture_post_conversion(
            layer=layer,
            backend=WNA16MoEBackend.TRITON,
            num_bits=4,
            symmetric=True,
            group_size=32,
            act_order=False,
        )


def test_canonical_cpu_bank_close_waits_for_capture_and_clears_publication():
    controller = PrivateWNA16ResidencyController(layer_id=3)
    registration = register_private_wna16_provider_factory(controller)
    layer = canonical_cpu_bank_layer()
    entered = Event()
    release = Event()
    failures = []

    def blocked_copy(tensor):
        entered.set()
        assert release.wait(timeout=3)
        return tensor.detach().clone().contiguous()

    def capture():
        try:
            _capture_canonical_cpu_bank(controller, layer)
        except BaseException as error:
            failures.append(error)

    with patch(
        "vllm.model_executor.layers.fused_moe.private_wna16_provider."
        "_copy_canonical_wna16_cpu_operand",
        side_effect=blocked_copy,
    ):
        capture_thread = Thread(target=capture)
        capture_thread.start()
        assert entered.wait(timeout=3)
        close_thread = Thread(target=registration.close)
        close_thread.start()
        assert close_thread.is_alive()
        release.set()
        capture_thread.join(timeout=3)
        close_thread.join(timeout=3)

    assert not failures
    assert not capture_thread.is_alive()
    assert not close_thread.is_alive()
    assert controller._canonical_cpu_bank is None


def test_canonical_cpu_bank_copy_failure_never_publishes_partial_bank():
    controller = PrivateWNA16ResidencyController(layer_id=3)
    registration = register_private_wna16_provider_factory(controller)
    layer = canonical_cpu_bank_layer()
    try:
        with (
            patch(
                "vllm.model_executor.layers.fused_moe.private_wna16_provider."
                "_copy_canonical_wna16_cpu_operand",
                side_effect=RuntimeError("copy failed"),
            ),
            pytest.raises(Phase4UnsupportedError, match="copy failed"),
        ):
            _capture_canonical_cpu_bank(controller, layer)
        assert controller._canonical_cpu_bank is None
    finally:
        registration.close()


def test_generic_provider_factory_close_is_not_a_lifecycle_callback():
    class Factory:
        closed = False

        def __call__(self, _):
            return lambda **_: unbound_private_view()

        def close(self):
            self.closed = True

    factory = Factory()
    registration = register_private_wna16_provider_factory(cast(Callable, factory))

    registration.close()

    assert factory.closed is False


def test_cpu_slot_planner_reserves_rolls_back_and_rejects_stale_leases():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)
    layer = canonical_cpu_bank_layer()
    try:
        _capture_canonical_cpu_bank(controller, layer)

        plan = controller._plan_slot_transaction(2)
        assert plan.status is _WNA16SlotPlanStatus.RESERVED
        assert plan.slot == 0
        assert plan.reservation is not None
        blocked = controller._plan_slot_transaction(1)
        assert blocked.status is _WNA16SlotPlanStatus.FALLBACK
        assert blocked.slot is None
        controller._rollback_slot_transaction(plan.reservation)
        assert controller._slots[0].state.name == "ABSENT"
        with pytest.raises(Phase4UnsupportedError, match="stale") as error:
            controller._rollback_slot_transaction(plan.reservation)
        assert error.value.category is Phase4FailureCategory.STALE_LEASE
    finally:
        registration.close()


def test_cpu_slot_planner_is_disabled_without_private_capacity_or_after_close():
    controller = PrivateWNA16ResidencyController(layer_id=3)
    registration = register_private_wna16_provider_factory(controller)
    layer = canonical_cpu_bank_layer()
    try:
        _capture_canonical_cpu_bank(controller, layer)
        with pytest.raises(Phase4UnsupportedError, match="unavailable"):
            controller._plan_slot_transaction(0)
    finally:
        registration.close()
    with pytest.raises(Phase4UnsupportedError, match="unavailable"):
        controller._plan_slot_transaction(0)


def test_cpu_slot_planner_hits_and_deterministically_rolls_back_lru_victim():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=2)
    registration = register_private_wna16_provider_factory(controller)
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        controller._seed_logical_resident_for_test(0, 1, last_used=10)
        controller._seed_logical_resident_for_test(1, 2, last_used=4)
        before_hit = controller._slots[0].last_used

        hit = controller._plan_slot_transaction(1)
        assert hit.status is _WNA16SlotPlanStatus.HIT
        assert hit.slot == 0
        assert hit.reservation is None
        assert controller._slots[0].expert_id == 1
        assert controller._slots[0].pins == 0
        assert controller._slots[0].last_used > before_hit

        plan = controller._plan_slot_transaction(3)
        assert plan.status is _WNA16SlotPlanStatus.RESERVED
        assert plan.slot == 1
        assert plan.reservation is not None
        assert controller._slots[1].state.name == "STAGED_REPLACEMENT"
        assert controller._slots[1].expert_id == 2
        controller._rollback_slot_transaction(plan.reservation)
        assert (
            controller._slots[1].state.name,
            controller._slots[1].expert_id,
            controller._slots[1].pins,
            controller._slots[1].last_used,
        ) == ("RESIDENT", 2, 0, 4)
    finally:
        registration.close()


def test_cpu_slot_planner_tie_breaks_equal_recency_by_lowest_slot_index():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=2)
    registration = register_private_wna16_provider_factory(controller)
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        controller._seed_logical_resident_for_test(0, 1, last_used=4)
        controller._seed_logical_resident_for_test(1, 2, last_used=4)

        plan = controller._plan_slot_transaction(3)

        assert plan.status is _WNA16SlotPlanStatus.RESERVED
        assert plan.slot == 0
        assert plan.reservation is not None
        controller._rollback_slot_transaction(plan.reservation)
    finally:
        registration.close()


def test_cpu_slot_planner_pinned_or_staged_slots_fall_back_without_mutation():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        controller._seed_logical_resident_for_test(0, 1, last_used=7)
        controller._acquire_logical_resident_pin_for_test(1)
        before = replace(controller._slots[0])
        before_clock = controller._slot_clock

        plan = controller._plan_slot_transaction(2)
        assert plan.status is _WNA16SlotPlanStatus.FALLBACK
        assert plan.slot is None
        assert plan.reservation is None
        assert controller._slots[0] == before
        assert controller._slot_clock == before_clock
    finally:
        registration.close()


def test_cpu_slot_planner_staged_fallback_preserves_all_private_state():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        controller._seed_logical_resident_for_test(0, 1, last_used=7)
        staged = controller._plan_slot_transaction(2)
        assert staged.reservation is not None
        slots_before = [replace(slot) for slot in controller._slots]
        clock_before = controller._slot_clock
        reservations_before = dict(controller._reservations)

        fallback = controller._plan_slot_transaction(3)

        assert fallback.status is _WNA16SlotPlanStatus.FALLBACK
        assert fallback.slot is None
        assert fallback.reservation is None
        assert controller._slots == slots_before
        assert controller._slot_clock == clock_before
        assert controller._reservations == reservations_before
        controller._rollback_slot_transaction(staged.reservation)
    finally:
        registration.close()


def test_cpu_slot_planner_rejects_forged_reservation_without_consuming_it():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        plan = controller._plan_slot_transaction(2)
        assert plan.reservation is not None
        forged = replace(plan.reservation)
        with pytest.raises(Phase4UnsupportedError) as error:
            controller._rollback_slot_transaction(forged)
        assert error.value.category is Phase4FailureCategory.STALE_LEASE
        assert len(controller._reservations) == 1
        controller._rollback_slot_transaction(plan.reservation)
    finally:
        registration.close()


def test_cpu_slot_planner_late_pin_restores_victim_without_eviction():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        controller._seed_logical_resident_for_test(0, 1, last_used=4)
        plan = controller._plan_slot_transaction(2)
        assert plan.reservation is not None
        controller._acquire_logical_resident_pin_for_test(1)
        controller._rollback_slot_transaction(plan.reservation)
        assert (
            controller._slots[0].state.name,
            controller._slots[0].expert_id,
            controller._slots[0].pins,
            controller._slots[0].last_used,
        ) == ("RESIDENT", 1, 1, 4)
        fallback = controller._plan_slot_transaction(2)
        assert fallback.status is _WNA16SlotPlanStatus.FALLBACK
        assert fallback.slot is None
    finally:
        registration.close()


def test_cpu_slot_planner_close_invalidates_outstanding_reservation():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)
    _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
    plan = controller._plan_slot_transaction(2)
    assert plan.reservation is not None
    registration.close()

    assert controller._canonical_cpu_bank is None
    assert controller._slots == []
    assert controller._reservations == {}
    with pytest.raises(Phase4UnsupportedError) as error:
        controller._rollback_slot_transaction(plan.reservation)
    assert error.value.category is Phase4FailureCategory.STALE_LEASE
    with pytest.raises(Phase4UnsupportedError, match="unavailable"):
        controller._plan_slot_transaction(2)


def test_cpu_source_slot_retention_blocks_reuse_until_exact_true_close_retry():
    """CPU-only ownership remains held; this is not a CUDA completion test."""
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)

    class Completion:
        result: object = False

        def query(self):
            return self.result

    completion = Completion()
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        source = controller._canonical_cpu_bank
        plan = controller._plan_slot_transaction(2)
        assert source is not None
        assert plan.reservation is not None
        retention = controller._retain_cpu_source_slot_until_completion(
            plan.reservation, completion
        )

        assert retention.source is source
        assert retention.reservation is plan.reservation
        controller._finalize_completed_cpu_source_slot_retentions()
        assert controller._canonical_cpu_bank is source
        assert controller._reservations[plan.reservation.token] is plan.reservation
        assert controller._retained_cpu_source_slots[retention.token] is retention
        assert (
            controller._plan_slot_transaction(1).status is _WNA16SlotPlanStatus.FALLBACK
        )

        with pytest.raises(Phase4UnsupportedError) as error:
            registration.close()
        assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
        assert controller._canonical_cpu_bank is source
        assert controller._reservations[plan.reservation.token] is plan.reservation

        completion.result = True
        registration.close()
        assert controller._canonical_cpu_bank is None
        assert controller._slots == []
        assert controller._reservations == {}
        assert controller._retained_cpu_source_slots == {}
    finally:
        completion.result = True
        registration.close()


@pytest.mark.parametrize("result", [1, 0, "complete", None])
def test_cpu_source_slot_retention_rejects_non_bool_completion_without_freeing(result):
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)

    class Completion:
        value: object = result

        def query(self):
            return self.value

    completion = Completion()
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        source = controller._canonical_cpu_bank
        plan = controller._plan_slot_transaction(2)
        assert source is not None
        assert plan.reservation is not None
        retention = controller._retain_cpu_source_slot_until_completion(
            plan.reservation, completion
        )

        with pytest.raises(Phase4UnsupportedError) as error:
            controller._finalize_completed_cpu_source_slot_retentions()
        assert error.value.category is Phase4FailureCategory.EVENT_QUERY
        assert controller._canonical_cpu_bank is source
        assert controller._reservations[plan.reservation.token] is plan.reservation
        assert controller._retained_cpu_source_slots[retention.token] is retention
        assert (
            controller._plan_slot_transaction(1).status is _WNA16SlotPlanStatus.FALLBACK
        )

        completion.value = True
        controller._finalize_completed_cpu_source_slot_retentions()
        assert controller._retained_cpu_source_slots == {}
        assert controller._reservations == {}
    finally:
        completion.value = True
        registration.close()


def test_cpu_source_slot_retention_query_is_outside_lifecycle_lock():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        plan = controller._plan_slot_transaction(2)
        assert plan.reservation is not None

        class Completion:
            def query(self) -> bool:
                acquired: list[bool] = []

                def acquire_lifecycle_lock() -> None:
                    locked = controller._lifecycle_lock.acquire(blocking=False)
                    acquired.append(locked)
                    if locked:
                        controller._lifecycle_lock.release()

                thread = Thread(target=acquire_lifecycle_lock)
                thread.start()
                thread.join(timeout=5)
                assert not thread.is_alive()
                assert acquired == [True]
                return True

        retention = controller._retain_cpu_source_slot_until_completion(
            plan.reservation, Completion()
        )
        controller._finalize_completed_cpu_source_slot_retentions()
        assert retention.token not in controller._retained_cpu_source_slots
        assert plan.reservation.token not in controller._reservations
    finally:
        registration.close()


def test_cpu_source_slot_retention_query_error_retains_until_close_retry():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)

    class Completion:
        complete = False

        def query(self) -> bool:
            if not self.complete:
                raise RuntimeError("query failed")
            return True

    completion = Completion()
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        source = controller._canonical_cpu_bank
        plan = controller._plan_slot_transaction(2)
        assert source is not None
        assert plan.reservation is not None
        retention = controller._retain_cpu_source_slot_until_completion(
            plan.reservation, completion
        )

        with pytest.raises(Phase4UnsupportedError) as error:
            registration.close()
        assert error.value.category is Phase4FailureCategory.EVENT_QUERY
        assert isinstance(error.value.__cause__, RuntimeError)
        assert controller._canonical_cpu_bank is source
        assert controller._reservations[plan.reservation.token] is plan.reservation
        assert controller._retained_cpu_source_slots[retention.token] is retention

        completion.complete = True
        registration.close()
        assert controller._canonical_cpu_bank is None
        assert controller._retained_cpu_source_slots == {}
    finally:
        completion.complete = True
        registration.close()


def test_cpu_source_slot_retention_requires_exact_reservation_identity():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)
    retention = None
    completion = SimpleNamespace(complete=False)
    completion.query = lambda: completion.complete
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        plan = controller._plan_slot_transaction(2)
        assert plan.reservation is not None
        forged = replace(plan.reservation)
        with pytest.raises(Phase4UnsupportedError) as error:
            controller._retain_cpu_source_slot_until_completion(forged, object())
        assert error.value.category is Phase4FailureCategory.STALE_LEASE
        retention = controller._retain_cpu_source_slot_until_completion(
            plan.reservation, completion
        )
        with pytest.raises(Phase4UnsupportedError) as error:
            controller._retain_cpu_source_slot_until_completion(
                plan.reservation, object()
            )
        assert error.value.category is Phase4FailureCategory.STALE_LEASE
        assert controller._retained_cpu_source_slots[retention.token] is retention
    finally:
        completion.complete = True
        registration.close()


def test_cpu_source_slot_retention_binding_is_immutable_and_forged_cannot_free():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)

    class Completion:
        complete = False

        def query(self) -> bool:
            return self.complete

    completion = Completion()
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        source = controller._canonical_cpu_bank
        plan = controller._plan_slot_transaction(2)
        assert source is not None
        assert plan.reservation is not None
        retention = controller._retain_cpu_source_slot_until_completion(
            plan.reservation, completion
        )
        with pytest.raises(FrozenInstanceError):
            retention.completion = SimpleNamespace(query=lambda: True)
        forged = replace(retention)
        controller._finalize_completed_cpu_source_slot_retention(forged)
        assert controller._canonical_cpu_bank is source
        assert controller._reservations[plan.reservation.token] is plan.reservation
        assert controller._retained_cpu_source_slots[retention.token] is retention
        assert controller._slots[plan.reservation.slot].state.name == "RESERVED_EMPTY"
        completion.complete = True
        controller._finalize_completed_cpu_source_slot_retentions()
    finally:
        completion.complete = True
        registration.close()


def test_cpu_source_slot_retention_rejects_rollback_until_completion():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)
    completion = SimpleNamespace(complete=False)
    completion.query = lambda: completion.complete
    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        plan = controller._plan_slot_transaction(2)
        assert plan.reservation is not None
        retention = controller._retain_cpu_source_slot_until_completion(
            plan.reservation, completion
        )
        with pytest.raises(Phase4UnsupportedError) as error:
            controller._rollback_slot_transaction(plan.reservation)
        assert error.value.category is Phase4FailureCategory.STALE_LEASE
        assert controller._reservations[plan.reservation.token] is plan.reservation
        assert controller._retained_cpu_source_slots[retention.token] is retention
        assert (
            controller._plan_slot_transaction(1).status is _WNA16SlotPlanStatus.FALLBACK
        )
        completion.complete = True
        controller._finalize_completed_cpu_source_slot_retentions()
    finally:
        completion.complete = True
        registration.close()


def test_cpu_source_slot_retention_close_race_retries_after_inflight_query():
    controller = PrivateWNA16ResidencyController(layer_id=3, slot_count=1)
    registration = register_private_wna16_provider_factory(controller)
    entered = Event()
    allow_return = Event()
    errors: list[BaseException] = []
    thread: Thread | None = None

    class Completion:
        def query(self) -> bool:
            entered.set()
            assert allow_return.wait(timeout=5)
            return True

    try:
        _capture_canonical_cpu_bank(controller, canonical_cpu_bank_layer())
        source = controller._canonical_cpu_bank
        plan = controller._plan_slot_transaction(2)
        assert source is not None
        assert plan.reservation is not None
        retention = controller._retain_cpu_source_slot_until_completion(
            plan.reservation, Completion()
        )

        def finalize() -> None:
            try:
                controller._finalize_completed_cpu_source_slot_retentions()
            except BaseException as error:
                errors.append(error)

        thread = Thread(target=finalize)
        thread.start()
        assert entered.wait(timeout=5)
        with pytest.raises(Phase4UnsupportedError) as error:
            registration.close()
        assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
        assert controller._canonical_cpu_bank is source
        assert controller._reservations[plan.reservation.token] is plan.reservation
        assert controller._retained_cpu_source_slots[retention.token] is retention
        with pytest.raises(Phase4UnsupportedError, match="unavailable"):
            controller._plan_slot_transaction(1)
        allow_return.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert errors == []
        registration.close()
        assert controller._canonical_cpu_bank is None
        assert controller._slots == []
        assert controller._reservations == {}
        assert controller._retained_cpu_source_slots == {}
    finally:
        allow_return.set()
        if thread is not None:
            thread.join(timeout=5)
        registration.close()


def test_default_off_is_explicit_and_fail_closed():
    adapter = Phase4GpuResidencyAdapter()

    capability = adapter.capability()

    assert capability.supported is False
    assert capability.category is Phase4FailureCategory.UNSUPPORTED_DYNAMIC_MAP
    assert capability.reason == "disabled by default"
    with pytest.raises(Phase4UnsupportedError, match="disabled by default") as error:
        adapter.validate_request(torch.tensor([[0, 1]]), torch.ones(1, 2))
    assert error.value.category is Phase4FailureCategory.UNSUPPORTED_DYNAMIC_MAP
    assert adapter.unsupported_requests == 1


def test_capability_rejects_unsupported_matrix_with_diagnostic():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="other",
        quantization="WNA16",
        execution_mode="graph",
        modular=False,
    )

    capability = adapter.capability()
    assert capability.category is Phase4FailureCategory.VALIDATION
    assert capability.reason == "only Qwen3-30B-A3B is supported"


def test_dynamic_map_claim_still_fails_closed_until_controller_exists():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="AWQ",
        backend_supports_dynamic_map=True,
    )

    assert adapter.capability().category is Phase4FailureCategory.VALIDATION
    with pytest.raises(Phase4UnsupportedError, match="only WNA16 is supported"):
        adapter.validate_request(torch.tensor([[2, 0]]), torch.ones(1, 2))


def test_cpu_reference_transfers_complete_post_conversion_bundle():
    source = complete_bundle()

    transferred = TorchCpuTransferReference.transfer(source)

    assert all(t.device.type == "cpu" for t in transferred.tensors.values())
    assert all(
        source.tensors[name].equal(transferred.tensors[name]) for name in source.tensors
    )
    assert all(
        source.tensors[name].data_ptr() != transferred.tensors[name].data_ptr()
        for name in source.tensors
    )
    assert transferred.metadata == source.metadata


def test_cpu_reference_rejects_partial_unknown_and_inconsistent_bundles():
    source = complete_bundle()
    cases = [
        ("partial", {"gate_proj": torch.ones(2, 2)}),
        (
            "unknown",
            {**source.tensors, "other": torch.ones(2, 2)},
        ),
    ]
    for _, tensors in cases:
        bundle = PostConversionExpertBundle(
            schema_version=1,
            tensors=tensors,
            metadata=source.metadata,
        )
        with pytest.raises(ValueError, match="exactly all WNA16 tensors"):
            TorchCpuTransferReference.transfer(bundle)

    inconsistent = PostConversionExpertBundle(
        schema_version=1,
        tensors=source.tensors,
        metadata={
            "weight_shapes": {
                **source.metadata["weight_shapes"],
                "gate_proj": (3, 3),
            },
            "quantization": "WNA16",
        },
    )
    with pytest.raises(ValueError, match="inconsistent"):
        TorchCpuTransferReference.transfer(inconsistent)


def test_cpu_reference_rejects_noncontiguous_converted_tensor():
    source = complete_bundle()
    source.tensors["gate_proj"] = torch.ones(2, 2).t()

    with pytest.raises(ValueError, match="contiguous"):
        TorchCpuTransferReference.transfer(source)


def test_cpu_reference_is_not_a_cuda_residency_proof():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = complete_bundle()
    cuda_source = PostConversionExpertBundle(
        schema_version=source.schema_version,
        tensors={name: tensor.cuda() for name, tensor in source.tensors.items()},
        metadata=source.metadata,
    )

    with pytest.raises(ValueError, match="CPU tensors only"):
        TorchCpuTransferReference.transfer(cuda_source)


def test_forward_modular_default_path_preserves_router_outputs():
    layer = mocked_routed_experts()
    x = torch.ones(1, 2)
    weights = torch.tensor([[0.25, 0.75]])
    ids = torch.tensor([[3, 1]], dtype=torch.int32)

    result = layer.forward_modular(x, weights, ids)

    assert result.equal(torch.tensor([1.0]))
    layer.quant_method.apply.assert_called_once()
    call = layer.quant_method.apply.call_args.kwargs
    assert call["topk_ids"] is ids
    assert call["topk_weights"] is weights


def test_forward_modular_adapter_does_not_block_ordinary_dispatch():
    layer = mocked_routed_experts(Phase4GpuResidencyAdapter())
    x = torch.ones(1, 2)
    weights = torch.tensor([[0.25, 0.75]])
    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    ids_before, weights_before = ids.clone(), weights.clone()
    manager_before = layer.expert_map_manager

    result = layer.forward_modular(x, weights, ids)

    assert result.equal(torch.tensor([1.0]))
    assert torch.equal(ids, ids_before)
    assert torch.equal(weights, weights_before)
    assert layer.expert_map_manager is manager_before
    layer.quant_method.apply.assert_called_once()


def test_forward_modular_transient_map_fails_closed_without_adapter():
    layer = mocked_routed_experts()
    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]])

    with pytest.raises(Phase4UnsupportedError, match="WNA16 transient map contract"):
        layer.forward_modular(
            torch.ones(1, 2),
            weights,
            ids,
            transient_expert_map=torch.tensor([-1, 0, 1, -1]),
        )

    layer.quant_method.apply.assert_not_called()


def test_forward_modular_rejects_non_wna16_before_adapter_validation():
    adapter = MagicMock()
    layer = mocked_routed_experts(adapter)
    layer.quant_method.supports_transient_expert_map = False
    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]])

    with pytest.raises(Phase4UnsupportedError, match="WNA16 transient map contract"):
        layer.forward_modular(
            torch.ones(1, 2),
            weights,
            ids,
            transient_expert_map=torch.tensor([-1, 0, 1, -1]),
        )

    adapter.validate_request.assert_not_called()
    layer.quant_method.apply.assert_not_called()


def test_forward_modular_passes_transient_map_only_after_adapter_validation():
    adapter = MagicMock()
    layer = mocked_routed_experts(adapter)
    layer.quant_method.supports_transient_expert_map = True
    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]])
    transient_map = torch.tensor([-1, 0, 1, -1])

    layer.forward_modular(
        torch.ones(1, 2),
        weights,
        ids,
        transient_expert_map=transient_map,
    )

    adapter.validate_request.assert_called_once_with(ids, weights, transient_map)
    assert (
        layer.quant_method.apply.call_args.kwargs["transient_expert_map"]
        is transient_map
    )


def test_wna16_marker_allows_gate_then_apply_remains_fail_closed():
    class MarkedWNA16Method:
        is_monolithic = False
        supports_transient_expert_map = True

        def apply(self, **kwargs):
            assert kwargs["transient_expert_map"] is transient_map
            raise RuntimeError(
                "WNA16 transient expert map requires a proven backend contract"
            )

    adapter = MagicMock()
    layer = mocked_routed_experts(adapter)
    layer.quant_method = MarkedWNA16Method()
    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]])
    transient_map = torch.tensor([-1, 0, 1, -1])

    with pytest.raises(RuntimeError, match="proven backend contract"):
        layer.forward_modular(
            torch.ones(1, 2),
            weights,
            ids,
            transient_expert_map=transient_map,
        )

    adapter.validate_request.assert_called_once_with(ids, weights, transient_map)


def test_wna16_methods_advertise_transient_map_contract():
    assert CompressedTensorsWNA16MoEMethod.supports_transient_expert_map is True
    assert CompressedTensorsWNA16RDNA3MoEMethod.supports_transient_expert_map is True


def test_wna16_apply_rejects_transient_map_before_kernel_dispatch():
    method = SimpleNamespace(is_monolithic=False, moe_kernel=MagicMock())
    transient_map = torch.tensor([-1, 0, 1, -1])

    with pytest.raises(RuntimeError, match="proven backend contract"):
        CompressedTensorsWNA16MoEMethod.apply(
            method,
            layer=MagicMock(),
            x=torch.ones(1, 2),
            topk_weights=torch.ones(1, 2),
            topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            shared_experts=None,
            shared_experts_input=None,
            transient_expert_map=transient_map,
        )

    method.moe_kernel.apply.assert_not_called()


def test_rdna3_apply_rejects_transient_map_before_kernel_dispatch():
    method = object.__new__(CompressedTensorsWNA16RDNA3MoEMethod)
    transient_map = torch.tensor([-1, 0, 1, -1])

    with pytest.raises(RuntimeError, match="proven backend contract"):
        method.apply(
            layer=MagicMock(),
            x=torch.ones(1, 2),
            topk_weights=torch.ones(1, 2),
            topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            shared_experts=None,
            shared_experts_input=None,
            transient_expert_map=transient_map,
        )


def test_forward_modular_rejects_transient_map_for_non_wna16_method():
    class UnsupportedQuantMethod:
        is_monolithic = False

        def apply(self, **kwargs):
            raise AssertionError("unsupported quantization method was dispatched")

    layer = mocked_routed_experts()
    layer.quant_method = UnsupportedQuantMethod()
    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]])
    ids_before, weights_before = ids.clone(), weights.clone()
    manager_before = layer.expert_map_manager

    with pytest.raises(Phase4UnsupportedError) as error:
        layer.forward_modular(
            torch.ones(1, 2),
            weights,
            ids,
            transient_expert_map=torch.tensor([-1, 0, 1, -1]),
        )

    assert error.value.category is Phase4FailureCategory.UNSUPPORTED_DYNAMIC_MAP
    assert torch.equal(ids, ids_before)
    assert torch.equal(weights, weights_before)
    assert layer.expert_map_manager is manager_before


def test_private_view_rejects_cpu_before_operator_or_static_retry():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    layer = mocked_routed_experts(adapter)
    layer.quant_method.supports_private_wna16_dispatch = True
    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]])
    view = private_view()
    manager_before = layer.expert_map_manager

    with pytest.raises(Phase4UnsupportedError) as error:
        layer.forward_modular(torch.ones(1, 2), weights, ids, generation_view=view)

    assert error.value.category is Phase4FailureCategory.VALIDATION
    layer.quant_method.apply.assert_not_called()
    assert torch.equal(ids, torch.tensor([[3, 1]], dtype=torch.int32))
    assert torch.equal(weights, torch.tensor([[0.25, 0.75]]))
    assert layer.expert_map_manager is manager_before


def test_wna16_private_dispatch_rejects_cpu_before_kernel():
    kernel = MagicMock()
    method = SimpleNamespace(
        is_monolithic=False,
        moe_kernel=kernel,
        wna16_backend=WNA16MoEBackend.TRITON,
    )
    method.num_bits = 4
    method.symmetric = True
    method.group_size = 32
    method.actorder = None
    view = private_view()
    layer = SimpleNamespace(
        use_ep=False,
        activation=MagicMock(),
        global_num_experts=4,
        apply_router_weight_on_input=False,
    )
    ids = torch.tensor([[2, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]])
    shared_experts = MagicMock()
    shared_experts_input = torch.ones(1, 2)

    with pytest.raises(Phase4UnsupportedError) as error:
        CompressedTensorsWNA16MoEMethod.apply(
            method,
            layer=layer,
            x=torch.ones(1, 2),
            topk_weights=weights,
            topk_ids=ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
            generation_view=view,
        )

    assert error.value.category is Phase4FailureCategory.VALIDATION
    kernel.apply_private_wna16.assert_not_called()


def test_stale_private_view_calls_neither_kernel_path():
    kernel = MagicMock()
    method = SimpleNamespace(
        is_monolithic=False,
        moe_kernel=kernel,
        wna16_backend=WNA16MoEBackend.TRITON,
        num_bits=4,
        symmetric=True,
        group_size=32,
        actorder=None,
    )
    view = private_view()
    object.__setattr__(
        view, "slot_map", torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    )
    layer = SimpleNamespace(
        use_ep=False,
        activation=MagicMock(),
        global_num_experts=4,
        apply_router_weight_on_input=False,
    )
    with pytest.raises(Phase4UnsupportedError) as error:
        CompressedTensorsWNA16MoEMethod.apply(
            method,
            layer=layer,
            x=torch.ones(1, 2),
            topk_weights=torch.ones(1, 2),
            topk_ids=torch.tensor([[0, 2]], dtype=torch.int32),
            shared_experts=None,
            shared_experts_input=None,
            generation_view=view,
        )
    assert error.value.category is Phase4FailureCategory.STALE_LEASE
    kernel.apply_private_wna16.assert_not_called()
    kernel.apply.assert_not_called()


def test_private_kernel_preserves_lora_and_shared_expert_contract():
    hidden_states = torch.ones(1, 2)
    lora_context = SimpleNamespace(original_hidden_states=None)
    observed_lora_inputs = []

    def apply(**kwargs):
        observed_lora_inputs.append(lora_context.original_hidden_states)
        kwargs["output"].zero_()

    fused_experts = SimpleNamespace(
        _lora_context=lora_context,
        a2_scale=None,
        moe_problem_size=lambda *_: (2, 1, 32, 2, 1),
        apply=apply,
    )
    impl = object.__new__(FusedMoEKernelModularImpl)
    impl.fused_experts = fused_experts
    impl._prepare = MagicMock(
        return_value=(
            hidden_states,
            None,
            None,
            torch.tensor([[0]], dtype=torch.int32),
            torch.ones(1, 1),
        )
    )
    impl._allocate_buffers = MagicMock(
        return_value=(None, None, torch.empty_like(hidden_states))
    )
    shared_experts = MagicMock()
    shared_experts_input = torch.full_like(hidden_states, 2)
    impl._finalize = MagicMock(return_value=torch.zeros_like(hidden_states))

    result = impl.apply_private_wna16(
        hidden_states,
        torch.zeros((2, 64, 16), dtype=torch.uint8),
        torch.zeros((2, 32, 16), dtype=torch.uint8),
        torch.ones((2, 64, 1)),
        torch.ones((2, 32, 1)),
        None,
        None,
        torch.ones(1, 1),
        torch.tensor([[0]], dtype=torch.int32),
        activation=MagicMock(),
        global_num_experts=2,
        slot_map=torch.tensor([0, 1], dtype=torch.int32),
        apply_router_weight_on_input=False,
        shared_experts=shared_experts,
        shared_experts_input=shared_experts_input,
    )

    assert torch.equal(result, torch.zeros_like(hidden_states))
    assert observed_lora_inputs == [hidden_states]
    assert lora_context.original_hidden_states is None
    assert impl._finalize.call_args.kwargs["shared_experts"] is shared_experts
    assert (
        impl._finalize.call_args.kwargs["shared_experts_input"] is shared_experts_input
    )


def test_wna16_private_dispatch_cpu_mock_forwards_raw_stable_operands():
    kernel = MagicMock()
    method = SimpleNamespace(
        is_monolithic=False,
        moe_kernel=kernel,
        wna16_backend=WNA16MoEBackend.TRITON,
        num_bits=4,
        symmetric=True,
        group_size=32,
        actorder=None,
    )
    view = private_view()
    operands = _private_wna16_dispatch_operands(view)
    layer = SimpleNamespace(
        use_ep=False,
        activation=MagicMock(),
        global_num_experts=4,
        apply_router_weight_on_input=False,
    )
    x = torch.ones(1, 2)
    topk_weights = torch.ones(1, 2)
    topk_ids = torch.tensor([[2, 0]], dtype=torch.int32)
    bundle_getattribute = WNA16ExpertBundle.__getattribute__
    view_getattribute = WNA16GenerationView.__getattribute__

    def guard_bundle(self, name):
        if name in WNA16ExpertBundle._TENSOR_FIELDS:
            raise AssertionError("production dispatch read a public tensor snapshot")
        return bundle_getattribute(self, name)

    def guard_view(self, name):
        if name == "slot_map":
            raise AssertionError("production dispatch read a public map snapshot")
        return view_getattribute(self, name)

    with (
        patch(
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors_moe.compressed_tensors_moe_wna16."
            "validate_private_wna16_dispatch_inputs"
        ) as validate,
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        patch.object(WNA16ExpertBundle, "__getattribute__", guard_bundle),
        patch.object(WNA16GenerationView, "__getattribute__", guard_view),
    ):
        CompressedTensorsWNA16MoEMethod.apply(
            method,
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
            generation_view=view,
        )

    validate.assert_called_once_with(
        view,
        hidden_states=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        global_num_experts=4,
        layer_id=3,
    )
    call = kernel.apply_private_wna16.call_args
    assert call.args[:9] == (
        x,
        operands.w13,
        operands.w2,
        operands.w13_scale,
        operands.w2_scale,
        operands.w13_zero,
        operands.w2_zero,
        topk_weights,
        topk_ids,
    )
    assert call.kwargs["slot_map"] is operands.slot_map
    kernel.apply.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_wna16_private_dispatch_cuda_forwarding_seam():
    kernel = MagicMock()
    method = SimpleNamespace(
        is_monolithic=False,
        moe_kernel=kernel,
        wna16_backend=WNA16MoEBackend.TRITON,
        num_bits=4,
        symmetric=True,
        group_size=32,
        actorder=None,
    )
    private_bundle = WNA16ExpertBundle(
        schema_version=1,
        backend="triton",
        layer_id=3,
        global_num_experts=4,
        slot_count=2,
        generation=9,
        quant_type="W4A16",
        num_bits=4,
        symmetric=True,
        group_size=32,
        act_order=False,
        w13=torch.zeros((2, 64, 16), device="cuda", dtype=torch.uint8),
        w2=torch.zeros((2, 32, 16), device="cuda", dtype=torch.uint8),
        w13_scale=torch.ones((2, 64, 1), device="cuda"),
        w2_scale=torch.ones((2, 32, 1), device="cuda"),
    )
    lease = WNA16UseLease(layer_id=3, generation=9, bundle=private_bundle, token=1)
    view = WNA16GenerationView(
        bundle=private_bundle,
        slot_map=torch.tensor([0, -1, 1, -1], device="cuda", dtype=torch.int32),
        map_generation=9,
        use_lease=lease,
    )
    view = _bind_controller_wna16_slot_storage(
        view, _new_controller_wna16_stable_slot_capability()
    )
    operands = _private_wna16_dispatch_operands(view)
    layer = SimpleNamespace(
        use_ep=False,
        activation=MagicMock(),
        global_num_experts=4,
        apply_router_weight_on_input=False,
    )
    x = torch.ones(1, 2, device="cuda")
    topk_weights = torch.ones(1, 2, device="cuda")
    topk_ids = torch.tensor([[2, 0]], device="cuda", dtype=torch.int32)
    ids_before, weights_before = topk_ids.clone(), topk_weights.clone()

    CompressedTensorsWNA16MoEMethod.apply(
        method,
        layer=layer,
        x=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
        generation_view=view,
    )

    kernel.apply_private_wna16.assert_called_once()
    call = kernel.apply_private_wna16.call_args
    assert call.args[0] is x
    for received, expected in zip(
        call.args[1:5],
        (operands.w13, operands.w2, operands.w13_scale, operands.w2_scale),
        strict=True,
    ):
        assert received is expected
        assert received.is_cuda
    assert call.args[5] is None
    assert call.args[6] is None
    assert call.args[7] is topk_weights
    assert call.args[8] is topk_ids
    assert call.kwargs["slot_map"] is operands.slot_map
    assert call.kwargs["slot_map"].is_cuda
    assert call.kwargs["global_num_experts"] == 4
    assert call.kwargs["shared_experts"] is None
    assert call.kwargs["shared_experts_input"] is None
    assert torch.equal(topk_ids, ids_before)
    assert torch.equal(topk_weights, weights_before)
    kernel.apply.assert_not_called()


def test_selected_private_layer_fails_closed_without_controller_factory():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    layer = mocked_routed_experts(adapter)

    with pytest.raises(
        Phase4UnsupportedError, match="registered controller provider factory"
    ):
        bind_private_wna16_generation_view_provider(layer_id=3, routed_experts=layer)

    assert layer._private_wna16_generation_view_provider is None


def test_public_view_cannot_issue_or_extract_stable_slot_storage():
    view = unbound_private_view()

    for operation in (validate_wna16_generation_view, _private_wna16_dispatch_operands):
        with pytest.raises(Phase4UnsupportedError) as error:
            operation(view)
        assert error.value.category is Phase4FailureCategory.STALE_LEASE

    with pytest.raises(Phase4UnsupportedError) as error:
        _bind_controller_wna16_slot_storage(view, object())  # type: ignore[arg-type]
    assert error.value.category is Phase4FailureCategory.STALE_LEASE


def test_registered_controller_factory_binds_dynamic_view_after_routing():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    routed_experts = mocked_routed_experts(adapter)
    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]])
    view = unbound_private_view()
    factory_requests = []
    provider_calls = []

    def factory(request):
        factory_requests.append(request)

        def provider(*, topk_ids, topk_weights):
            provider_calls.append((topk_ids, topk_weights))
            return view

        return provider

    registration = register_private_wna16_provider_factory(factory)
    try:
        bind_private_wna16_generation_view_provider(
            layer_id=3, routed_experts=routed_experts
        )
        runner = object.__new__(MoERunner)
        torch.nn.Module.__init__(runner)
        runner.routed_experts = routed_experts
        runner.router = MagicMock()
        runner.router.select_experts.return_value = (weights, ids)
        runner._shared_experts = None
        routed_experts.forward_modular = MagicMock(return_value=torch.tensor([1.0]))

        _, output = runner._apply_quant_method(
            hidden_states=torch.ones(1, 2),
            router_logits=torch.ones(1, 4),
            shared_experts_input=None,
        )
    finally:
        registration.close()

    assert factory_requests[0].layer_id == 3
    assert factory_requests[0].routed_experts is routed_experts
    assert provider_calls == [(ids, weights)]
    routed_experts.forward_modular.assert_called_once()
    assert routed_experts.forward_modular.call_args.kwargs["generation_view"] is view
    assert isinstance(output, torch.Tensor)
    assert torch.equal(output, torch.tensor([1.0]))


def test_private_provider_cannot_fall_back_when_it_returns_no_view():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    layer = mocked_routed_experts(adapter)
    layer.set_private_wna16_generation_view_provider(
        cast("Callable[..., WNA16GenerationView]", lambda **_: None), layer_id=3
    )

    with pytest.raises(Phase4UnsupportedError, match="returned no generation view"):
        layer.get_private_wna16_generation_view(
            topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.25, 0.75]]),
        )


def test_private_view_provider_receives_current_router_outputs():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    layer = mocked_routed_experts(adapter)
    layer._private_wna16_generation_view_provider = None
    view = private_view()
    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]])
    observed = []

    with pytest.raises(Phase4UnsupportedError, match="no bound controller"):
        layer.get_private_wna16_generation_view(topk_ids=ids, topk_weights=weights)

    def provider(*, topk_ids, topk_weights):
        observed.append((topk_ids, topk_weights))
        return view

    layer.set_private_wna16_generation_view_provider(provider, layer_id=3)

    assert (
        layer.get_private_wna16_generation_view(topk_ids=ids, topk_weights=weights)
        is view
    )
    assert observed == [(ids, weights)]


def test_private_dispatch_rejects_quantization_mismatch_before_kernel():
    kernel = MagicMock()
    method = SimpleNamespace(
        is_monolithic=False,
        moe_kernel=kernel,
        wna16_backend=WNA16MoEBackend.TRITON,
        num_bits=8,
        symmetric=True,
        group_size=32,
        actorder=None,
    )
    layer = SimpleNamespace(
        use_ep=False,
        activation=MagicMock(),
        global_num_experts=4,
        apply_router_weight_on_input=False,
    )

    with pytest.raises(Phase4UnsupportedError) as error:
        CompressedTensorsWNA16MoEMethod.apply(
            method,
            layer=layer,
            x=torch.ones(1, 2),
            topk_weights=torch.ones(1, 2),
            topk_ids=torch.tensor([[0, 2]], dtype=torch.int32),
            shared_experts=None,
            shared_experts_input=None,
            generation_view=private_view(),
        )

    assert error.value.category is Phase4FailureCategory.VALIDATION
    kernel.apply_private_wna16.assert_not_called()


def test_registration_close_revokes_an_already_bound_provider():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    layer = mocked_routed_experts(adapter)
    calls = []

    def provider(**_):
        calls.append(True)
        return unbound_private_view()

    registration = register_private_wna16_provider_factory(lambda _: provider)
    bind_private_wna16_generation_view_provider(layer_id=3, routed_experts=layer)
    issued = layer.get_private_wna16_generation_view(
        topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
        topk_weights=torch.tensor([[0.25, 0.75]]),
    )
    assert issued is not None
    registration.close()

    with pytest.raises(Phase4UnsupportedError) as error:
        _private_wna16_dispatch_operands(issued)
    assert error.value.category is Phase4FailureCategory.STALE_LEASE

    with pytest.raises(Phase4UnsupportedError) as error:
        layer.get_private_wna16_generation_view(
            topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.25, 0.75]]),
        )

    assert error.value.category is Phase4FailureCategory.STALE_LEASE
    assert calls == [True]


def test_private_request_use_event_defers_final_release_without_cuda():
    """The preparatory hook is testable with a CPU fake event, not CUDA work."""
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    routed_experts = mocked_routed_experts(adapter)
    registration = register_private_wna16_provider_factory(
        cast(Callable, lambda _: lambda **_: unbound_private_view())
    )
    try:
        bind_private_wna16_generation_view_provider(
            layer_id=3, routed_experts=routed_experts
        )
        view = routed_experts.get_private_wna16_generation_view(
            topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.25, 0.75]]),
        )
        assert view is not None
        calls: list[str] = []

        class FakeEvent:
            complete = False

            def __init__(self, *, enable_timing: bool) -> None:
                assert enable_timing is False
                calls.append("event-create")

            def record(self) -> None:
                calls.append("event-record")

            def query(self) -> bool:
                calls.append("event-query")
                return self.complete

        kernel = MagicMock()

        def dispatch(*_, **__) -> None:
            calls.append("kernel")
            registration.close()

        kernel.apply_private_wna16.side_effect = dispatch
        method = SimpleNamespace(
            is_monolithic=False,
            moe_kernel=kernel,
            wna16_backend=WNA16MoEBackend.TRITON,
            num_bits=4,
            symmetric=True,
            group_size=32,
            actorder=None,
        )
        layer = SimpleNamespace(
            use_ep=False,
            activation=MagicMock(),
            global_num_experts=4,
            apply_router_weight_on_input=False,
        )
        with (
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.compressed_tensors_moe_wna16."
                "validate_private_wna16_dispatch_inputs"
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.Event", FakeEvent),
            pytest.raises(Phase4UnsupportedError) as error,
        ):
            CompressedTensorsWNA16MoEMethod.apply(
                method,
                layer=layer,
                x=torch.ones(1, 2),
                topk_weights=torch.ones(1, 2),
                topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
                shared_experts=None,
                shared_experts_input=None,
                generation_view=view,
            )

        request_use_lease = object.__getattribute__(view, "_request_use_lease")
        assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
        assert calls == ["event-create", "kernel", "event-record", "event-query"]
        assert request_use_lease.state is _WNA16RequestUseState.FENCED_PENDING
        assert len(request_use_lease.manager._by_id) == 1
        with pytest.raises(Phase4UnsupportedError) as error:
            acquire_private_wna16_request_use(view)
        assert error.value.category is Phase4FailureCategory.STALE_LEASE
        request_use_lease.manager.finalize_completed()
        assert request_use_lease.state is _WNA16RequestUseState.FENCED_PENDING
        request_use_lease.completion_event.complete = True
        request_use_lease.manager.finalize_completed()
        assert request_use_lease.state is _WNA16RequestUseState.RELEASED
        assert len(request_use_lease.manager._by_id) == 0
        assert calls[-3:] == ["event-query", "event-query", "event-query"]
    finally:
        registration.close()


def test_close_retries_completed_event_attachment_after_non_drained_failure():
    """A close cannot succeed before an in-flight dispatch records its fence."""
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    routed_experts = mocked_routed_experts(adapter)
    registration = register_private_wna16_provider_factory(
        cast(Callable, lambda _: lambda **_: unbound_private_view())
    )
    try:
        bind_private_wna16_generation_view_provider(
            layer_id=3, routed_experts=routed_experts
        )
        view = routed_experts.get_private_wna16_generation_view(
            topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.25, 0.75]]),
        )
        assert view is not None
        lease = acquire_private_wna16_request_use(view)
        assert lease is not None
        mark_private_wna16_request_dispatched(lease)
        begin_private_wna16_request_enqueue(lease)

        class CompleteEvent:
            def query(self) -> bool:
                return True

        with pytest.raises(Phase4UnsupportedError) as error:
            registration.close()
        assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
        assert lease.state is _WNA16RequestUseState.ENQUEUING
        assert lease.manager._by_id[id(lease)] is lease
        release_private_wna16_request_use_pending(lease, CompleteEvent())

        assert lease.state is _WNA16RequestUseState.RELEASED
        assert len(lease.manager._by_id) == 0
        registration.close()
    finally:
        registration.close()


@pytest.mark.parametrize("result", [1, 0, "complete", None])
def test_completion_event_query_requires_exact_bool_and_retains_lease(result):
    registration, view, _, _ = _request_use_dispatch_target(MagicMock())
    try:
        lease = acquire_private_wna16_request_use(view)
        assert lease is not None
        mark_private_wna16_request_dispatched(lease)
        begin_private_wna16_request_enqueue(lease)

        class InvalidResultEvent:
            def query(self):
                return result

        event = InvalidResultEvent()
        release_private_wna16_request_use_pending(lease, event)
        with pytest.raises(Phase4UnsupportedError) as error:
            lease.manager.finalize_completed()
        assert error.value.category is Phase4FailureCategory.EVENT_QUERY
        assert lease.state is _WNA16RequestUseState.FENCED_PENDING
        assert lease.completion_event is event
        assert lease.manager._by_id[id(lease)] is lease
        with pytest.raises(Phase4UnsupportedError) as stale:
            acquire_private_wna16_request_use(view)
        assert stale.value.category is Phase4FailureCategory.STALE_LEASE
        lease.completion_event = SimpleNamespace(query=lambda: True)
    finally:
        registration.close()


def test_completion_event_missing_query_and_exception_retain_lease():
    registration, view, _, _ = _request_use_dispatch_target(MagicMock())
    try:
        lease = acquire_private_wna16_request_use(view)
        assert lease is not None
        mark_private_wna16_request_dispatched(lease)
        begin_private_wna16_request_enqueue(lease)
        event = object()
        release_private_wna16_request_use_pending(lease, event)
        with pytest.raises(Phase4UnsupportedError) as error:
            lease.manager.finalize_completed()
        assert error.value.category is Phase4FailureCategory.EVENT_QUERY
        assert lease.state is _WNA16RequestUseState.FENCED_PENDING
        assert lease.completion_event is event

        class RaisingEvent:
            def query(self) -> bool:
                raise RuntimeError("query failed")

        raising_event = RaisingEvent()
        lease.completion_event = raising_event
        with pytest.raises(Phase4UnsupportedError) as error:
            lease.manager.finalize_completed()
        assert error.value.category is Phase4FailureCategory.EVENT_QUERY
        assert isinstance(error.value.__cause__, RuntimeError)
        assert lease.state is _WNA16RequestUseState.FENCED_PENDING
        assert lease.completion_event is raising_event
        assert lease.manager._by_id[id(lease)] is lease
        lease.completion_event = SimpleNamespace(query=lambda: True)
    finally:
        registration.close()


def test_finalize_completed_queries_later_leases_after_event_query_error():
    registration, first_view, _, _ = _request_use_dispatch_target(MagicMock())
    try:
        second_view = registration._registration.request_use_manager.bind(
            unbound_private_view()
        )
        first_lease = acquire_private_wna16_request_use(first_view)
        second_lease = acquire_private_wna16_request_use(second_view)
        assert first_lease is not None
        assert second_lease is not None
        mark_private_wna16_request_dispatched(first_lease)
        begin_private_wna16_request_enqueue(first_lease)
        mark_private_wna16_request_dispatched(second_lease)
        begin_private_wna16_request_enqueue(second_lease)
        calls: list[str] = []

        class RaisingEvent:
            def query(self) -> bool:
                calls.append("first")
                raise RuntimeError("query failed")

        class CompleteEvent:
            def query(self) -> bool:
                calls.append("second")
                return True

        failing_event = RaisingEvent()
        release_private_wna16_request_use_pending(first_lease, failing_event)
        release_private_wna16_request_use_pending(second_lease, CompleteEvent())

        with pytest.raises(Phase4UnsupportedError) as error:
            first_lease.manager.finalize_completed()
        assert error.value.category is Phase4FailureCategory.EVENT_QUERY
        assert isinstance(error.value.__cause__, RuntimeError)
        assert calls == ["first", "second"]
        assert first_lease.state is _WNA16RequestUseState.FENCED_PENDING
        assert first_lease.completion_event is failing_event
        assert first_lease.manager._by_id[id(first_lease)] is first_lease
        assert not first_lease.query_in_progress
        assert second_lease.state is _WNA16RequestUseState.RELEASED
        assert id(second_lease) not in second_lease.manager._by_id
        first_lease.completion_event = SimpleNamespace(query=lambda: True)
    finally:
        registration.close()


def test_closed_registration_retries_pending_event_query_without_reopening_admission():
    calls = MagicMock()
    registration = register_private_wna16_provider_factory(
        cast(Callable, lambda _: calls)
    )
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    routed_experts = mocked_routed_experts(adapter)
    try:
        calls.return_value = unbound_private_view()
        bind_private_wna16_generation_view_provider(
            layer_id=3, routed_experts=routed_experts
        )
        view = routed_experts.get_private_wna16_generation_view(
            topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.25, 0.75]]),
        )
        assert view is not None
        lease = acquire_private_wna16_request_use(view)
        assert lease is not None
        mark_private_wna16_request_dispatched(lease)
        begin_private_wna16_request_enqueue(lease)

        class RetryableEvent:
            complete = False

            def query(self) -> bool:
                if not self.complete:
                    raise RuntimeError("query failed")
                return True

        event = RetryableEvent()
        release_private_wna16_request_use_pending(lease, event)
        with pytest.raises(Phase4UnsupportedError) as error:
            registration.close()
        assert error.value.category is Phase4FailureCategory.EVENT_QUERY
        assert lease.state is _WNA16RequestUseState.FENCED_PENDING
        assert lease.completion_event is event
        with pytest.raises(Phase4UnsupportedError) as stale:
            routed_experts.get_private_wna16_generation_view(
                topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
                topk_weights=torch.tensor([[0.25, 0.75]]),
            )
        assert stale.value.category is Phase4FailureCategory.STALE_LEASE
        assert calls.call_count == 1

        event.complete = True
        registration.close()
        assert lease.state is _WNA16RequestUseState.RELEASED
        assert id(lease) not in lease.manager._by_id
        assert calls.call_count == 1
    finally:
        registration.close()


def test_provider_admission_query_error_does_not_call_provider():
    calls = MagicMock()
    registration = register_private_wna16_provider_factory(
        cast(Callable, lambda _: calls)
    )
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    routed_experts = mocked_routed_experts(adapter)
    try:
        calls.return_value = unbound_private_view()
        bind_private_wna16_generation_view_provider(
            layer_id=3, routed_experts=routed_experts
        )
        view = routed_experts.get_private_wna16_generation_view(
            topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.25, 0.75]]),
        )
        assert view is not None
        lease = acquire_private_wna16_request_use(view)
        assert lease is not None
        mark_private_wna16_request_dispatched(lease)
        begin_private_wna16_request_enqueue(lease)

        class RaisingEvent:
            def query(self) -> bool:
                raise RuntimeError("query failed")

        event = RaisingEvent()
        release_private_wna16_request_use_pending(lease, event)
        with pytest.raises(Phase4UnsupportedError) as error:
            routed_experts.get_private_wna16_generation_view(
                topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
                topk_weights=torch.tensor([[0.25, 0.75]]),
            )
        assert error.value.category is Phase4FailureCategory.EVENT_QUERY
        assert calls.call_count == 1
        assert lease.state is _WNA16RequestUseState.FENCED_PENDING
        assert lease.completion_event is event
        lease.completion_event = SimpleNamespace(query=lambda: True)
    finally:
        registration.close()


def test_closed_manager_release_pending_query_error_is_not_record_error():
    registration, view, _, _ = _request_use_dispatch_target(MagicMock())
    try:
        lease = acquire_private_wna16_request_use(view)
        assert lease is not None
        mark_private_wna16_request_dispatched(lease)
        begin_private_wna16_request_enqueue(lease)

        class RaisingEvent:
            def query(self) -> bool:
                raise RuntimeError("query failed")

        event = RaisingEvent()
        with pytest.raises(Phase4UnsupportedError) as drain_error:
            registration.close()
        assert drain_error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
        with pytest.raises(Phase4UnsupportedError) as error:
            release_private_wna16_request_use_pending(lease, event)
        assert error.value.category is Phase4FailureCategory.EVENT_QUERY
        assert isinstance(error.value.__cause__, RuntimeError)
        assert lease.state is _WNA16RequestUseState.FENCED_PENDING
        assert lease.completion_event is event
        assert lease.manager._by_id[id(lease)] is lease
        lease.completion_event = SimpleNamespace(query=lambda: True)
    finally:
        registration.close()


def test_close_failure_remains_primary_over_recorded_event_query_error():
    calls: list[str] = []

    class RaisingQueryEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is False
            calls.append("event-create")

        def record(self) -> None:
            calls.append("event-record")

        def query(self) -> bool:
            calls.append("event-query")
            raise RuntimeError("query failed")

    kernel = MagicMock()
    registration, view, method, layer = _request_use_dispatch_target(kernel)

    def close_registration(*_args, **_kwargs):
        registration.close()

    kernel.apply_private_wna16.side_effect = close_registration
    try:
        with (
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.compressed_tensors_moe_wna16."
                "validate_private_wna16_dispatch_inputs"
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.Event", RaisingQueryEvent),
            pytest.raises(Phase4UnsupportedError) as error,
        ):
            _apply_private_request_use(method, layer, view)

        lease = object.__getattribute__(view, "_request_use_lease")
        assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
        assert calls == ["event-create", "event-record", "event-query"]
        assert lease.state is _WNA16RequestUseState.FENCED_PENDING
        assert lease.completion_event is not None
        assert lease.manager._by_id[id(lease)] is lease
        kernel.apply_private_wna16.assert_called_once()
        kernel.apply.assert_not_called()
        lease.completion_event = SimpleNamespace(query=lambda: True)
    finally:
        registration.close()


def test_kernel_error_remains_primary_when_closed_event_query_fails():
    class KernelFailure(Exception):
        pass

    calls: list[str] = []

    class RaisingQueryEvent:
        should_complete = False

        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is False
            calls.append("event-create")

        def record(self) -> None:
            calls.append("event-record")

        def query(self) -> bool:
            calls.append("event-query")
            if self.should_complete:
                return True
            raise RuntimeError("query failed")

    kernel = MagicMock()
    registration, view, method, layer = _request_use_dispatch_target(kernel)

    def close_then_fail(*_args, **_kwargs) -> None:
        with pytest.raises(Phase4UnsupportedError) as drain_error:
            registration.close()
        assert drain_error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
        raise KernelFailure("kernel failed")

    kernel.apply_private_wna16.side_effect = close_then_fail
    try:
        with (
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.compressed_tensors_moe_wna16."
                "validate_private_wna16_dispatch_inputs"
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.Event", RaisingQueryEvent),
            pytest.raises(KernelFailure, match="kernel failed"),
        ):
            _apply_private_request_use(method, layer, view)

        lease = object.__getattribute__(view, "_request_use_lease")
        assert calls == ["event-create", "event-record", "event-query"]
        assert lease.state is _WNA16RequestUseState.FENCED_PENDING
        assert lease.manager._by_id[id(lease)] is lease
        lease.completion_event.should_complete = True
        lease.manager.finalize_completed()
        assert lease.state is _WNA16RequestUseState.RELEASED
        assert id(lease) not in lease.manager._by_id
    finally:
        registration.close()


def test_finalize_completed_avoids_duplicate_query_in_progress():
    registration, view, _, _ = _request_use_dispatch_target(MagicMock())
    query_entered = Event()
    allow_query_return = Event()
    try:
        lease = acquire_private_wna16_request_use(view)
        assert lease is not None
        mark_private_wna16_request_dispatched(lease)
        begin_private_wna16_request_enqueue(lease)

        class BlockingEvent:
            calls = 0
            complete = False

            def query(self) -> bool:
                self.calls += 1
                query_entered.set()
                assert allow_query_return.wait(timeout=5)
                return self.complete

        event = BlockingEvent()
        release_private_wna16_request_use_pending(lease, event)
        query_thread = Thread(target=lease.manager.finalize_completed)
        query_thread.start()
        assert query_entered.wait(timeout=5)
        lease.manager.finalize_completed()
        assert event.calls == 1
        allow_query_return.set()
        query_thread.join(timeout=5)
        assert not query_thread.is_alive()
        assert lease.state is _WNA16RequestUseState.FENCED_PENDING
        assert not lease.query_in_progress
        event.complete = True
        lease.manager.finalize_completed()
        assert lease.state is _WNA16RequestUseState.RELEASED
        assert id(lease) not in lease.manager._by_id
    finally:
        allow_query_return.set()
        registration.close()


def test_completion_event_query_runs_outside_manager_lock():
    registration, view, _, _ = _request_use_dispatch_target(MagicMock())
    try:
        lease = acquire_private_wna16_request_use(view)
        assert lease is not None
        mark_private_wna16_request_dispatched(lease)
        begin_private_wna16_request_enqueue(lease)

        class LockCheckingEvent:
            def query(self) -> bool:
                acquired: list[bool] = []

                def acquire_manager_lock() -> None:
                    acquired_lock = lease.manager._lock.acquire(blocking=False)
                    acquired.append(acquired_lock)
                    if acquired_lock:
                        lease.manager._lock.release()

                lock_thread = Thread(target=acquire_manager_lock)
                lock_thread.start()
                lock_thread.join(timeout=5)
                assert not lock_thread.is_alive()
                assert acquired == [True]
                return True

        release_private_wna16_request_use_pending(lease, LockCheckingEvent())
        lease.manager.finalize_completed()
        assert lease.state is _WNA16RequestUseState.RELEASED
        assert id(lease) not in lease.manager._by_id
    finally:
        registration.close()


def test_shutdown_wins_pre_enqueue_releases_lease_and_never_calls_kernel():
    """A marked-but-not-admitted use is revocable without a completion proof."""
    kernel = MagicMock()
    registration, view, method, layer = _request_use_dispatch_target(kernel)
    marked = Event()
    resume = Event()
    outcome: list[BaseException] = []
    original_mark = mark_private_wna16_request_dispatched

    def mark_and_pause(lease) -> None:
        original_mark(lease)
        marked.set()
        assert resume.wait(timeout=5)

    def apply_in_thread() -> None:
        try:
            _apply_private_request_use(method, layer, view)
        except BaseException as error:
            outcome.append(error)

    try:
        with (
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.compressed_tensors_moe_wna16."
                "validate_private_wna16_dispatch_inputs"
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.Event", MagicMock()),
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.compressed_tensors_moe_wna16."
                "mark_private_wna16_request_dispatched",
                side_effect=mark_and_pause,
            ),
        ):
            thread = Thread(target=apply_in_thread)
            thread.start()
            assert marked.wait(timeout=5)
            lease = object.__getattribute__(view, "_request_use_lease")
            assert lease.state is _WNA16RequestUseState.PRE_ENQUEUE
            registration.close()
            assert lease.state is _WNA16RequestUseState.RELEASED
            assert id(lease) not in lease.manager._by_id
            kernel.apply_private_wna16.assert_not_called()
            resume.set()
            thread.join(timeout=5)
            assert not thread.is_alive()
        assert len(outcome) == 1
        assert isinstance(outcome[0], Phase4UnsupportedError)
        assert outcome[0].category is Phase4FailureCategory.STALE_LEASE
        kernel.apply_private_wna16.assert_not_called()
    finally:
        resume.set()


def test_enqueue_wins_pre_enqueue_race_retains_lease_until_fence_completes():
    """Once enqueue admission owns the lock, close must retain the live use."""
    kernel = MagicMock()
    registration, view, method, layer = _request_use_dispatch_target(kernel)
    admitted = Event()
    resume = Event()
    outcome: list[BaseException] = []
    original_begin = begin_private_wna16_request_enqueue

    class PendingEvent:
        complete = False

        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is False

        def record(self) -> None:
            pass

        def query(self) -> bool:
            return self.complete

    def begin_and_pause(lease) -> None:
        original_begin(lease)
        admitted.set()
        assert resume.wait(timeout=5)

    def apply_in_thread() -> None:
        try:
            _apply_private_request_use(method, layer, view)
        except BaseException as error:
            outcome.append(error)

    try:
        with (
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.compressed_tensors_moe_wna16."
                "validate_private_wna16_dispatch_inputs"
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.Event", PendingEvent),
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.compressed_tensors_moe_wna16."
                "begin_private_wna16_request_enqueue",
                side_effect=begin_and_pause,
            ),
        ):
            thread = Thread(target=apply_in_thread)
            thread.start()
            assert admitted.wait(timeout=5)
            lease = object.__getattribute__(view, "_request_use_lease")
            assert lease.state is _WNA16RequestUseState.ENQUEUING
            with pytest.raises(Phase4UnsupportedError) as error:
                registration.close()
            assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
            assert lease.manager._by_id[id(lease)] is lease
            resume.set()
            thread.join(timeout=5)
            assert not thread.is_alive()
            assert not outcome
            kernel.apply_private_wna16.assert_called_once()
            assert lease.state is _WNA16RequestUseState.FENCED_PENDING
            with pytest.raises(Phase4UnsupportedError) as error:
                registration.close()
            assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
            lease.completion_event.complete = True
            registration.close()
            assert lease.state is _WNA16RequestUseState.RELEASED
            assert id(lease) not in lease.manager._by_id
    finally:
        resume.set()
        registration.close()


def test_verified_shutdown_drain_proof_rejects_wrong_lease_and_replay():
    """A shutdown-only proof is exact, one-shot, and cannot release a fence."""
    factory = _VerifiedShutdownDrainFactory()
    first = register_private_wna16_provider_factory(cast(Callable, factory))
    first_view = first._registration.request_use_manager.bind(unbound_private_view())
    first_lease = None
    try:
        first_lease = acquire_private_wna16_request_use(first_view)
        assert first_lease is not None
        mark_private_wna16_request_dispatched(first_lease)
        begin_private_wna16_request_enqueue(first_lease)
        first_lease.manager.quarantine_unfenced(first_lease)
        second_view = first._registration.request_use_manager.bind(
            unbound_private_view()
        )
        second_lease = acquire_private_wna16_request_use(second_view)
        assert second_lease is not None
        mark_private_wna16_request_dispatched(second_lease)
        begin_private_wna16_request_enqueue(second_lease)
        second_lease.manager.quarantine_unfenced(second_lease)
        with pytest.raises(Phase4UnsupportedError) as error:
            first.close()
        assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE

        factory.expected_lease = first_lease
        first_proof = first.issue_verified_shutdown_drain_proof(first_lease)
        with pytest.raises(Phase4UnsupportedError) as error:
            first.release_verified_drain(second_lease, first_proof)
        assert error.value.category is Phase4FailureCategory.STALE_LEASE
        assert second_lease.state is _WNA16RequestUseState.UNFENCED_QUARANTINED

        first.release_verified_drain(first_lease, first_proof)
        with pytest.raises(Phase4UnsupportedError) as error:
            first.release_verified_drain(first_lease, first_proof)
        assert error.value.category is Phase4FailureCategory.STALE_LEASE
        factory.expected_lease = second_lease
        second_proof = first.issue_verified_shutdown_drain_proof(second_lease)
        first.release_verified_drain(second_lease, second_proof)
        first.close()
    finally:
        if first_lease is not None and first_lease.manager._by_id:
            remaining = next(iter(first_lease.manager._by_id.values()))
            factory.expected_lease = remaining
            proof = first.issue_verified_shutdown_drain_proof(remaining)
            first.release_verified_drain(remaining, proof)
        first.close()


def test_legacy_factory_cannot_supply_an_arbitrary_drain_callback():
    """A registration stores no verifier for legacy callable factories."""
    registration, view, _, _ = _request_use_dispatch_target(MagicMock())
    lease = _quarantine_request_use(registration, view)
    try:
        with pytest.raises(TypeError):
            cast(Callable, registration.issue_verified_shutdown_drain_proof)(
                lease, lambda _: True
            )
        with pytest.raises(Phase4UnsupportedError) as error:
            registration.issue_verified_shutdown_drain_proof(lease)
        assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
        assert lease.manager._by_id[id(lease)] is lease
    finally:
        # Test-only teardown: legacy factories cannot produce a valid proof, so
        # remove the intentionally retained lease before closing the registry.
        if lease.manager._by_id:
            lease.manager._release(lease)
        registration.close()


def test_shutdown_drain_blocks_re_registration_until_old_storage_releases():
    """A failed close keeps the revoked registration active until it drains."""
    first_factory = _VerifiedShutdownDrainFactory()
    first = register_private_wna16_provider_factory(cast(Callable, first_factory))
    first_view = first._registration.request_use_manager.bind(unbound_private_view())
    lease = _quarantine_request_use(first, first_view)
    try:
        second_factory = _VerifiedShutdownDrainFactory()
        with pytest.raises(Phase4UnsupportedError) as error:
            register_private_wna16_provider_factory(cast(Callable, second_factory))
        assert error.value.category is Phase4FailureCategory.UNSUPPORTED_DYNAMIC_MAP

        first_factory.expected_lease = lease
        proof = first.issue_verified_shutdown_drain_proof(lease)
        first.release_verified_drain(lease, proof)
        first.close()
        second = register_private_wna16_provider_factory(cast(Callable, second_factory))
        second.close()
    finally:
        first.close()


def test_placeholder_controller_cannot_issue_verified_shutdown_drain_proof():
    """The bootstrap placeholder has no verified drain backend."""
    controller = PrivateWNA16ResidencyController(layer_id=3)
    registration = register_private_wna16_provider_factory(controller)
    view = registration._registration.request_use_manager.bind(unbound_private_view())
    lease = _quarantine_request_use(registration, view)
    with pytest.raises(Phase4UnsupportedError) as error:
        registration.issue_verified_shutdown_drain_proof(lease)
    assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
    assert lease.manager._by_id[id(lease)] is lease
    # Test-only teardown: this placeholder deliberately cannot prove drain.
    lease.manager._release(lease)
    registration.close()


def test_completion_event_create_failure_releases_acquired_request_use():
    kernel = MagicMock()
    registration, view, method, layer = _request_use_dispatch_target(kernel)
    try:
        with (
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.compressed_tensors_moe_wna16."
                "validate_private_wna16_dispatch_inputs"
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.Event", side_effect=RuntimeError("create failed")),
            pytest.raises(Phase4UnsupportedError, match="create a completion"),
        ):
            _apply_private_request_use(method, layer, view)

        lease = object.__getattribute__(view, "_request_use_lease")
        assert lease.state is _WNA16RequestUseState.RELEASED
        assert len(lease.manager._by_id) == 0
        kernel.apply_private_wna16.assert_not_called()
    finally:
        registration.close()


def test_apply_monolithic_assertion_cancels_issued_request_use():
    kernel = MagicMock()
    registration, view, method, layer = _request_use_dispatch_target(kernel)
    try:
        method.is_monolithic = True
        with pytest.raises(AssertionError):
            _apply_private_request_use(method, layer, view)

        lease = object.__getattribute__(view, "_request_use_lease")
        assert lease.state is _WNA16RequestUseState.RELEASED
        assert len(lease.manager._by_id) == 0
        kernel.apply_private_wna16.assert_not_called()
        kernel.apply.assert_not_called()
    finally:
        registration.close()


def test_apply_missing_kernel_assertion_cancels_issued_request_use():
    kernel = MagicMock()
    registration, view, method, layer = _request_use_dispatch_target(kernel)
    try:
        method.moe_kernel = None
        with pytest.raises(AssertionError):
            _apply_private_request_use(method, layer, view)

        lease = object.__getattribute__(view, "_request_use_lease")
        assert lease.state is _WNA16RequestUseState.RELEASED
        assert len(lease.manager._by_id) == 0
        kernel.apply_private_wna16.assert_not_called()
        kernel.apply.assert_not_called()
    finally:
        registration.close()


def test_input_validation_failure_cancels_issued_request_use():
    kernel = MagicMock()
    registration, view, method, layer = _request_use_dispatch_target(kernel)
    try:
        with pytest.raises(Phase4UnsupportedError) as error:
            _apply_private_request_use(method, layer, view)

        lease = object.__getattribute__(view, "_request_use_lease")
        assert error.value.category is Phase4FailureCategory.VALIDATION
        assert lease.state is _WNA16RequestUseState.RELEASED
        assert len(lease.manager._by_id) == 0
        kernel.apply_private_wna16.assert_not_called()
        kernel.apply.assert_not_called()
    finally:
        registration.close()


def test_cuda_capture_failure_cancels_issued_request_use():
    kernel = MagicMock()
    registration, view, method, layer = _request_use_dispatch_target(kernel)
    try:
        with (
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.compressed_tensors_moe_wna16."
                "validate_private_wna16_dispatch_inputs"
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            pytest.raises(
                Phase4UnsupportedError, match="does not support CUDA graph capture"
            ) as error,
        ):
            _apply_private_request_use(method, layer, view)

        lease = object.__getattribute__(view, "_request_use_lease")
        assert error.value.category is Phase4FailureCategory.VALIDATION
        assert lease.state is _WNA16RequestUseState.RELEASED
        assert len(lease.manager._by_id) == 0
        kernel.apply_private_wna16.assert_not_called()
        kernel.apply.assert_not_called()
    finally:
        registration.close()


def test_completion_event_record_failure_retains_dispatched_request_use():
    class RecordFailureEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is False

        def record(self) -> None:
            raise RuntimeError("record failed")

    kernel = MagicMock()
    factory = _VerifiedShutdownDrainFactory()
    registration, view, method, layer = _request_use_dispatch_target(kernel, factory)
    lease = None
    try:
        with (
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.compressed_tensors_moe_wna16."
                "validate_private_wna16_dispatch_inputs"
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.Event", RecordFailureEvent),
            pytest.raises(Phase4UnsupportedError, match="record a completion"),
        ):
            _apply_private_request_use(method, layer, view)

        lease = object.__getattribute__(view, "_request_use_lease")
        assert lease.state is _WNA16RequestUseState.UNFENCED_QUARANTINED
        assert len(lease.manager._by_id) == 1
        with pytest.raises(Phase4UnsupportedError) as error:
            registration.close()
        assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
        assert lease.state is _WNA16RequestUseState.UNFENCED_QUARANTINED
        assert len(lease.manager._by_id) == 1
        with pytest.raises(Phase4UnsupportedError) as retry_error:
            registration.close()
        assert retry_error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
        with pytest.raises(Phase4UnsupportedError) as stale:
            acquire_private_wna16_request_use(view)
        assert stale.value.category is Phase4FailureCategory.STALE_LEASE
        assert lease is not None
        factory.expected_lease = lease
        proof = registration.issue_verified_shutdown_drain_proof(lease)
        registration.release_verified_drain(lease, proof)
        assert lease.state is _WNA16RequestUseState.RELEASED
        assert len(lease.manager._by_id) == 0
        registration.close()
    finally:
        if lease is not None and lease.manager._by_id:
            factory.expected_lease = lease
            proof = registration.issue_verified_shutdown_drain_proof(lease)
            registration.release_verified_drain(lease, proof)
        registration.close()


def test_kernel_error_remains_primary_when_event_record_also_fails():
    class KernelFailure(Exception):
        pass

    class RecordFailureEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is False

        def record(self) -> None:
            raise RuntimeError("record failed")

    kernel = MagicMock()
    kernel.apply_private_wna16.side_effect = KernelFailure("kernel failed")
    factory = _VerifiedShutdownDrainFactory()
    registration, view, method, layer = _request_use_dispatch_target(kernel, factory)
    lease = None
    try:
        with (
            patch(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe.compressed_tensors_moe_wna16."
                "validate_private_wna16_dispatch_inputs"
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.Event", RecordFailureEvent),
            pytest.raises(KernelFailure, match="kernel failed"),
        ):
            _apply_private_request_use(method, layer, view)

        lease = object.__getattribute__(view, "_request_use_lease")
        assert lease.state is _WNA16RequestUseState.UNFENCED_QUARANTINED
        assert len(lease.manager._by_id) == 1
    finally:
        if lease is not None and lease.manager._by_id:
            with pytest.raises(Phase4UnsupportedError) as error:
                registration.close()
            assert error.value.category is Phase4FailureCategory.DRAIN_INCOMPLETE
            factory.expected_lease = lease
            proof = registration.issue_verified_shutdown_drain_proof(lease)
            registration.release_verified_drain(lease, proof)
        registration.close()


def test_provider_close_during_dispatch_rejects_before_stable_binding():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    layer = mocked_routed_experts(adapter)
    registration = None
    returned_view = unbound_private_view()

    def provider(**_):
        assert registration is not None
        registration.close()
        return returned_view

    registration = register_private_wna16_provider_factory(lambda _: provider)
    bind_private_wna16_generation_view_provider(layer_id=3, routed_experts=layer)

    with pytest.raises(Phase4UnsupportedError) as error:
        layer.get_private_wna16_generation_view(
            topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.25, 0.75]]),
        )

    assert error.value.category is Phase4FailureCategory.STALE_LEASE
    assert object.__getattribute__(returned_view, "_stable_slot_storage") is None


def test_stale_close_cannot_revoke_same_factory_replacement():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )

    def factory(_):
        return lambda **_: unbound_private_view()

    first = register_private_wna16_provider_factory(factory)
    first.close()
    replacement = register_private_wna16_provider_factory(factory)
    first.close()
    try:
        layer = mocked_routed_experts(adapter)
        bind_private_wna16_generation_view_provider(layer_id=3, routed_experts=layer)
        assert (
            layer.get_private_wna16_generation_view(
                topk_ids=torch.tensor([[3, 1]], dtype=torch.int32),
                topk_weights=torch.tensor([[0.25, 0.75]]),
            )
            is not None
        )
    finally:
        replacement.close()


@pytest.mark.parametrize("factory", [None, object()])
def test_registration_rejects_non_callable_factories(factory):
    with pytest.raises(Phase4UnsupportedError, match="must be callable"):
        register_private_wna16_provider_factory(factory)


def test_binding_rejects_non_callable_provider():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    registration = register_private_wna16_provider_factory(lambda _: object())
    try:
        with pytest.raises(Phase4UnsupportedError, match="callable provider"):
            bind_private_wna16_generation_view_provider(
                layer_id=3, routed_experts=mocked_routed_experts(adapter)
            )
    finally:
        registration.close()


def test_binding_fails_if_factory_closes_its_registration():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )
    registration = None

    def factory(_):
        assert registration is not None
        registration.close()
        return lambda **_: private_view()

    registration = register_private_wna16_provider_factory(factory)
    with pytest.raises(Phase4UnsupportedError) as error:
        bind_private_wna16_generation_view_provider(
            layer_id=3, routed_experts=mocked_routed_experts(adapter)
        )
    assert error.value.category is Phase4FailureCategory.STALE_LEASE


def _staging_controller():
    controller = PrivateWNA16ResidencyController(3, slot_count=2)
    layer = canonical_cpu_bank_layer()
    controller(
        PrivateWNA16ProviderBindingRequest(
            layer_id=3, routed_experts=cast(RoutedExperts, layer)
        )
    )
    controller.capture_post_conversion(
        layer=cast(RoutedExperts, layer),
        backend=WNA16MoEBackend.TRITON,
        num_bits=4,
        symmetric=True,
        group_size=32,
        act_order=False,
    )
    return controller


def test_private_cuda_slot_staging_cpu_contract_is_fenced_but_unpublished():
    class CompleteEvent:
        def __init__(self):
            self.recorded = False

        def record(self):
            self.recorded = True

        def query(self):
            return True

    controller = _staging_controller()
    staged = staged_cpu_slot(controller)
    original_map = staged.slot_map.clone()
    event = CompleteEvent()

    controller._stage_controller_private_cuda_slot(
        expert_id=0, staged_slot=staged, fill_event_factory=lambda: event
    )

    assert event.recorded
    assert torch.equal(staged.slot_map, original_map)
    assert len(controller._retained_cpu_source_slots) == 1
    assert controller._slots[0].state.name == "RESERVED_EMPTY"
    with pytest.raises(Phase4UnsupportedError, match="not implemented"):
        controller._request_generation_view(
            topk_ids=torch.tensor([[0]], dtype=torch.int32),
            topk_weights=torch.tensor([[1.0]]),
        )
    controller._finalize_completed_cpu_source_slot_retentions()
    assert not controller._retained_cpu_source_slots
    assert controller._slots[0].state.name == "ABSENT"
    controller.close()


@pytest.mark.parametrize("failure", ["create", "record"])
def test_private_cuda_slot_staging_rolls_back_event_failures(failure):
    class RecordFailureEvent:
        def record(self):
            raise RuntimeError("record failed")

    controller = _staging_controller()
    staged = staged_cpu_slot(controller)
    factory = (
        (lambda: (_ for _ in ()).throw(RuntimeError("create failed")))
        if failure == "create"
        else RecordFailureEvent
    )

    with pytest.raises(RuntimeError, match=failure):
        controller._stage_controller_private_cuda_slot(
            expert_id=0, staged_slot=staged, fill_event_factory=factory
        )

    assert not controller._reservations
    assert not controller._retained_cpu_source_slots
    assert controller._slots[0].state.name == "ABSENT"
    assert controller._plan_slot_transaction(0).status is _WNA16SlotPlanStatus.RESERVED
    controller.close()


@pytest.mark.parametrize(
    "primary",
    [SystemExit("factory exit"), KeyboardInterrupt()],
    ids=["system-exit", "keyboard-interrupt"],
)
def test_private_cuda_slot_staging_rolls_back_base_exceptions(primary):
    controller = _staging_controller()
    staged = staged_cpu_slot(controller)
    event_factory = MagicMock(side_effect=primary)

    with pytest.raises(type(primary)) as caught:
        controller._stage_controller_private_cuda_slot(
            expert_id=0, staged_slot=staged, fill_event_factory=event_factory
        )

    assert caught.value is primary
    assert not controller._reservations
    assert not controller._retained_cpu_source_slots
    assert controller._slots[0].state.name == "ABSENT"
    reservation = controller._plan_slot_transaction(0).reservation
    assert reservation is not None
    controller._rollback_slot_transaction(reservation)
    controller.close()


def test_private_cuda_slot_staging_rejects_incomplete_map_before_event():
    controller = _staging_controller()
    staged = staged_cpu_slot(controller)
    staged.slot_map[0] = -1
    event_factory = MagicMock()

    with pytest.raises(Phase4UnsupportedError, match="slot map is incomplete"):
        controller._stage_controller_private_cuda_slot(
            expert_id=0, staged_slot=staged, fill_event_factory=event_factory
        )

    event_factory.assert_not_called()
    assert not controller._reservations
    assert not controller._retained_cpu_source_slots
    assert controller._slots[0].state.name == "ABSENT"
    controller.close()


def test_private_cuda_slot_staging_rejects_non_cpu_map_before_read():
    controller = _staging_controller()
    staged = staged_cpu_slot(controller, map_device="meta")
    event_factory = MagicMock()

    with pytest.raises(Phase4UnsupportedError, match="must be CPU-resident"):
        controller._stage_controller_private_cuda_slot(
            expert_id=0, staged_slot=staged, fill_event_factory=event_factory
        )

    event_factory.assert_not_called()
    assert not controller._reservations
    assert not controller._retained_cpu_source_slots
    assert controller._slots[0].state.name == "ABSENT"
    controller.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_private_wna16_cuda_abi_preflight_only():
    """ABI preflight only; no CUDA execution, H2D, residency, or parity proof."""
    assert callable(torch.cuda.Event)
    assert callable(torch.cuda.is_current_stream_capturing)
    assert callable(FusedMoEKernelModularImpl.apply_private_wna16)
