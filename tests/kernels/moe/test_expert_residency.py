# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import astuple, replace
from random import Random

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_residency import (
    ExpertBundle,
    ExpertKey,
    ExpertResidencyTable,
    EventKind,
    LoadStatus,
    Phase4FailureCategory,
    Phase4GpuResidencyAdapter,
    Phase4UnsupportedError,
    ResidencyAccounting,
    SlotState,
    WNA16ExpertBundle,
    WNA16GenerationView,
    WNA16UseLease,
    validate_private_wna16_dispatch_inputs,
    validate_wna16_generation_view,
)


def bundle(generation: int, payload: object | None = None) -> ExpertBundle:
    return ExpertBundle(generation=generation, payload=payload)


def private_view(slot_map: torch.Tensor | None = None) -> WNA16GenerationView:
    w13 = torch.zeros((2, 64, 16), dtype=torch.uint8)
    w2 = torch.zeros((2, 32, 16), dtype=torch.uint8)
    w13_scale = torch.ones((2, 64, 1), dtype=torch.float32)
    w2_scale = torch.ones((2, 32, 1), dtype=torch.float32)
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
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
    )
    if slot_map is None:
        slot_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    lease = WNA16UseLease(layer_id=3, generation=9, bundle=bundle, token=1)
    return WNA16GenerationView(bundle, slot_map, 9, lease)


def test_wna16_private_view_validates_complete_bundle_and_map():
    validate_wna16_generation_view(private_view())


def test_wna16_private_view_accepts_actual_triton_n_first_layout():
    view = private_view()
    assert view.bundle.w13.dtype is torch.uint8
    validate_wna16_generation_view(view)


def test_wna16_private_view_rejects_backend_layout_mismatch():
    view = private_view()
    bundle = replace(view.bundle, backend="humming")
    with pytest.raises(Phase4UnsupportedError, match="backend"):
        validate_wna16_generation_view(replace(view, bundle=bundle))


def test_wna16_private_view_rejects_slot_map_on_different_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    view = private_view(torch.tensor([0, -1, 1, -1], device="cuda", dtype=torch.int32))
    with pytest.raises(Phase4UnsupportedError, match="slot_map.*device"):
        validate_wna16_generation_view(view)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "w13",
            torch.zeros((2, 64, 16), dtype=torch.float16),
            "weights must be uint8",
        ),
        (
            "w13",
            torch.zeros((2, 64, 1), dtype=torch.uint8),
            "complete quantization groups",
        ),
        ("w13", torch.zeros((2, 64), dtype=torch.uint8), "rank 3"),
        ("w13_scale", torch.ones((2, 64, 2)), "scale shapes"),
        ("w2", torch.zeros((2, 16, 16), dtype=torch.uint8), "twice the w2 rows"),
    ),
)
def test_wna16_private_view_rejects_malformed_tensor_schema(
    field: str, value: torch.Tensor, message: str
):
    view = private_view()
    bundle = replace(view.bundle, **{field: value})
    view = replace(
        view,
        bundle=bundle,
        use_lease=WNA16UseLease(3, 9, bundle, 1),
    )

    with pytest.raises(Phase4UnsupportedError, match=message) as error:
        validate_wna16_generation_view(view)
    assert error.value.category is Phase4FailureCategory.VALIDATION


def test_wna16_private_view_rejects_duplicate_slot_before_dispatch():
    view = private_view(torch.tensor([0, 0, 1, -1], dtype=torch.int32))
    with pytest.raises(Phase4UnsupportedError, match="unique") as error:
        validate_wna16_generation_view(view)
    assert error.value.category is Phase4FailureCategory.VALIDATION


def test_wna16_private_view_is_default_off_even_when_structurally_valid():
    adapter = Phase4GpuResidencyAdapter()
    with pytest.raises(Phase4UnsupportedError, match="disabled by default") as error:
        adapter.validate_generation_view(private_view())
    assert error.value.category is Phase4FailureCategory.UNSUPPORTED_DYNAMIC_MAP


def test_private_dispatch_rejects_cpu_inputs_before_kernel_dispatch():
    with pytest.raises(Phase4UnsupportedError, match="one CUDA device") as error:
        validate_private_wna16_dispatch_inputs(
            private_view(),
            hidden_states=torch.ones(1, 2),
            topk_ids=torch.tensor([[0, 2]], dtype=torch.int32),
            topk_weights=torch.ones(1, 2),
            global_num_experts=4,
            layer_id=3,
        )
    assert error.value.category is Phase4FailureCategory.VALIDATION


def test_private_dispatch_requires_bound_layer_even_when_enabled():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
    )

    with pytest.raises(Phase4UnsupportedError, match="bound layer id"):
        adapter.validate_generation_view(private_view())


def test_private_dispatch_accepts_explicit_bound_layer():
    adapter = Phase4GpuResidencyAdapter(
        enabled=True,
        model_family="Qwen3-30B-A3B",
        quantization="WNA16",
        backend_supports_dynamic_map=True,
        private_dispatch_enabled=True,
        private_dispatch_layer_id=3,
    )

    adapter.validate_generation_view(private_view())


@pytest.mark.parametrize(
    ("field", "value"),
    (("quant_type", "invalid"), ("num_bits", 16), ("group_size", 0)),
)
def test_wna16_private_view_rejects_invalid_quantization_metadata(field, value):
    view = private_view()
    bundle = replace(view.bundle, **{field: value})
    view = replace(view, bundle=bundle)
    with pytest.raises(Phase4UnsupportedError) as error:
        validate_wna16_generation_view(view)
    assert error.value.category is Phase4FailureCategory.VALIDATION


def test_wna16_private_view_rejects_malformed_nested_bundle():
    view = replace(private_view(), bundle=object())
    with pytest.raises(Phase4UnsupportedError) as error:
        validate_wna16_generation_view(view)
    assert error.value.category is Phase4FailureCategory.VALIDATION


def test_wna16_private_view_snapshots_input_tensors():
    view = private_view()
    source = torch.zeros((2, 64, 16), dtype=torch.uint8)
    bundle = replace(view.bundle, w13=source)
    source.fill_(7)
    view.bundle.w13.fill_(7)
    view.slot_map.fill_(1)
    view = replace(
        view,
        bundle=bundle,
        use_lease=WNA16UseLease(3, 9, bundle, 1),
    )
    validate_wna16_generation_view(view)
    assert torch.count_nonzero(view.bundle.w13) == 0


def test_wna16_private_view_rejects_stale_or_unassociated_lease():
    view = private_view()
    stale = WNA16UseLease(3, 8, view.bundle, 1)
    with pytest.raises(Phase4UnsupportedError) as error:
        validate_wna16_generation_view(replace(view, use_lease=stale))
    assert error.value.category is Phase4FailureCategory.STALE_LEASE
    foreign = private_view().bundle
    unrelated = WNA16UseLease(3, 9, foreign, 1)
    with pytest.raises(Phase4UnsupportedError) as error:
        validate_wna16_generation_view(replace(view, use_lease=unrelated))
    assert error.value.category is Phase4FailureCategory.STALE_LEASE


def test_wna16_private_view_rejects_asymmetric_zero_point_mismatch():
    view = private_view()
    bundle = replace(
        view.bundle,
        symmetric=False,
        w13_zero=torch.zeros((2, 31, 1), dtype=torch.uint8),
        w2_zero=torch.zeros((2, 16, 1), dtype=torch.uint8),
    )
    view = replace(
        view,
        bundle=bundle,
        use_lease=WNA16UseLease(3, 9, bundle, 1),
    )
    with pytest.raises(Phase4UnsupportedError, match="shapes"):
        validate_wna16_generation_view(view)


@pytest.mark.parametrize(
    "zero",
    (object(), torch.zeros((2, 8), dtype=torch.uint8)[:, ::2]),
)
def test_wna16_private_view_rejects_malformed_zero_point_tensor(zero):
    view = private_view()
    bundle = replace(
        view.bundle,
        symmetric=False,
        w13_zero=zero,
        w2_zero=torch.zeros((2, 16, 1), dtype=torch.uint8),
    )
    view = replace(
        view,
        bundle=bundle,
        use_lease=WNA16UseLease(3, 9, bundle, 1),
    )
    with pytest.raises(Phase4UnsupportedError, match="zero-point"):
        validate_wna16_generation_view(view)


def test_layer_qualified_keys_are_immutable_and_reject_negative_ids():
    first_layer = ExpertKey(layer_id=0, logical_expert_id=3)
    second_layer = ExpertKey(layer_id=1, logical_expert_id=3)

    assert first_layer != second_layer
    assert len({first_layer, second_layer}) == 2
    with pytest.raises(ValueError, match="non-negative"):
        ExpertKey(layer_id=-1, logical_expert_id=0)
    with pytest.raises(ValueError, match="non-negative"):
        ExpertKey(layer_id=0, logical_expert_id=-1)
    with pytest.raises(AttributeError):
        first_layer.layer_id = 2


def test_warm_hit_returns_the_published_bundle_without_loading_again():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda key, value: True)
    key = ExpertKey(0, 0)
    loaded = table.resolve(key, lambda _: bundle(1, "weights"))

    hit = table.resolve(key, lambda _: pytest.fail("warm hit must not load"))

    assert loaded.status is LoadStatus.LOADED
    assert hit.status is LoadStatus.HIT
    assert hit.bundle == bundle(1, "weights")
    assert hit.lease is None
    assert table.accounting.hits == 1
    assert table.accounting.load_successes == 1
    table.assert_invariants()


def test_cold_miss_publishes_only_a_validated_complete_bundle():
    seen: list[ExpertKey] = []
    table = ExpertResidencyTable(
        capacity=1,
        validate_bundle=lambda key, value: seen.append(key) is None
        and value.payload == "complete",
    )
    key = ExpertKey(2, 7)

    result = table.resolve(key, lambda requested: bundle(4, "complete"))

    assert result.status is LoadStatus.LOADED
    assert result.key == key
    assert result.bundle == bundle(4, "complete")
    assert table.slot_for(key) == result.slot
    assert table.slot_state(result.slot) is SlotState.RESIDENT
    assert seen == [key]
    assert table.accounting.requests == 1
    assert table.accounting.misses == 1
    assert table.accounting.load_successes == 1
    table.assert_invariants()


def test_deterministic_unpinned_eviction_unpublishes_before_reload():
    table = ExpertResidencyTable(capacity=2, validate_bundle=lambda key, value: True)
    first = ExpertKey(0, 0)
    second = ExpertKey(0, 1)
    third = ExpertKey(0, 2)
    table.resolve(first, lambda _: bundle(1))
    table.resolve(second, lambda _: bundle(2))

    replacement = table.resolve(third, lambda _: bundle(3))

    assert replacement.status is LoadStatus.LOADED
    assert table.slot_for(first) is None
    assert table.slot_for(second) is not None
    assert table.slot_for(third) == replacement.slot

    reloaded = table.resolve(first, lambda _: bundle(4))

    assert reloaded.status is LoadStatus.LOADED
    assert reloaded.slot == 1
    assert table.accounting.evictions == 2
    table.assert_invariants()


def test_pin_and_duplicate_loading_protect_slots_from_reuse():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda key, value: True)
    pinned = ExpertKey(0, 0)
    requested = ExpertKey(0, 1)
    table.resolve(pinned, lambda _: bundle(1))
    table.pin(pinned)

    fallback = table.resolve(requested, lambda _: bundle(2))

    assert fallback.status is LoadStatus.FALLBACK
    assert fallback.lease is None
    assert table.slot_for(pinned) == 0
    assert table.slot_for(requested) is None
    assert table.pin_count(pinned) == 1
    no_capacity = table.begin_load(requested)
    assert no_capacity.status is LoadStatus.FALLBACK
    assert no_capacity.slot is None
    assert no_capacity.lease is None
    table.unpin(pinned)
    reserved = table.begin_load(requested)
    assert reserved.status is LoadStatus.RESERVED
    assert reserved.slot == 0
    with pytest.raises(RuntimeError, match="already loading"):
        table.begin_load(requested)
    assert table.slot_state(0) is SlotState.RESIDENT
    assert table.slot_for(pinned) == 0
    table.assert_invariants()


def test_failed_load_releases_capacity_for_a_different_key():
    table = ExpertResidencyTable(
        capacity=1,
        validate_bundle=lambda _, value: value.payload == "valid",
    )
    failed = ExpertKey(0, 0)
    replacement = ExpertKey(0, 1)

    failed_result = table.resolve(failed, lambda _: bundle(1, "invalid"))
    replacement_result = table.resolve(replacement, lambda _: bundle(2, "valid"))

    assert failed_result.status is LoadStatus.FALLBACK
    assert failed_result.bundle is None
    assert table.slot_for(failed) is None
    assert replacement_result.status is LoadStatus.LOADED
    assert replacement_result.slot == 0
    assert table.slot_for(replacement) == 0
    assert table.slot_state(0) is SlotState.RESIDENT
    assert table.accounting.load_failures == 1
    assert table.accounting.load_successes == 1
    table.assert_invariants()


def test_assert_invariants_rejects_resident_slot_without_exact_reverse_mapping():
    table = ExpertResidencyTable(capacity=2, validate_bundle=lambda _, value: True)
    key = ExpertKey(0, 0)
    table.resolve(key, lambda _: bundle(1))
    duplicate = table._slots[1]
    duplicate.state = SlotState.RESIDENT
    duplicate.key = key
    duplicate.bundle = bundle(2)
    duplicate.generation = 2

    with pytest.raises(AssertionError):
        table.assert_invariants()


def test_assert_invariants_rejects_published_mapping_outside_capacity():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    key = ExpertKey(0, 0)
    table.resolve(key, lambda _: bundle(1))
    table._published[key] = 1

    with pytest.raises(AssertionError):
        table.assert_invariants()


def test_assert_invariants_rejects_key_in_published_and_loading():
    table = ExpertResidencyTable(capacity=2, validate_bundle=lambda _, value: True)
    published = ExpertKey(0, 0)
    pending = ExpertKey(0, 1)
    table.resolve(published, lambda _: bundle(1))
    table.begin_load(pending)
    reservation = table._loading.pop(pending)
    table._loading[published] = reservation

    with pytest.raises(AssertionError):
        table.assert_invariants()


@pytest.mark.parametrize(
    "field",
    (
        "requests",
        "hits",
        "misses",
        "load_successes",
        "load_failures",
        "evictions",
    ),
)
def test_assert_invariants_rejects_negative_accounting(field: str):
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    setattr(table.accounting, field, -1)

    with pytest.raises(AssertionError):
        table.assert_invariants()


def test_assert_invariants_rejects_quiescent_evicting_slot():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    table._slots[0].state = SlotState.EVICTING

    with pytest.raises(AssertionError):
        table.assert_invariants()


def test_failed_load_never_publishes_and_leaves_existing_mapping_usable():
    table = ExpertResidencyTable(
        capacity=2,
        validate_bundle=lambda _, value: value.payload == "valid",
    )
    existing = ExpertKey(0, 0)
    failed = ExpertKey(0, 1)
    table.resolve(existing, lambda _: bundle(1, "valid"))

    result = table.resolve(failed, lambda _: bundle(2, "invalid"))
    hit = table.resolve(existing, lambda _: pytest.fail("existing mapping was lost"))

    assert result.status is LoadStatus.FALLBACK
    assert result.key == failed
    assert result.bundle is None
    assert table.slot_for(failed) is None
    assert table.slot_state(result.slot) is SlotState.FAILED
    assert hit.status is LoadStatus.HIT
    assert table.accounting.load_failures == 1
    table.assert_invariants()


def test_begin_load_distinguishes_resident_hit_and_new_reservation():
    table = ExpertResidencyTable(capacity=2, validate_bundle=lambda _, value: True)
    resident = ExpertKey(0, 0)
    requested = ExpertKey(0, 1)
    table.resolve(resident, lambda _: bundle(1, "resident"))

    hit = table.begin_load(resident)
    reserved = table.begin_load(requested)

    assert hit.status is LoadStatus.HIT
    assert hit.slot == table.slot_for(resident)
    assert hit.bundle == bundle(1, "resident")
    assert hit.lease is None
    assert reserved.status is LoadStatus.RESERVED
    assert reserved.slot is not None
    assert reserved.bundle is None
    assert table.slot_state(reserved.slot) is SlotState.LOADING
    assert reserved.lease is not None
    table.fail_load(reserved.lease)
    table.assert_invariants()


def test_complete_and_fail_load_reject_hit_results_without_mutating_resident_mapping():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    resident = ExpertKey(0, 0)
    resident_bundle = bundle(1, "resident")
    table.resolve(resident, lambda _: resident_bundle)

    hit = table.begin_load(resident)

    assert hit.status is LoadStatus.HIT
    with pytest.raises(RuntimeError):
        table.complete_load(hit.lease, bundle(2, "replacement"))
    with pytest.raises(RuntimeError):
        table.fail_load(hit.lease)
    assert table.slot_for(resident) == hit.slot
    resolved = table.resolve(
        resident,
        lambda _: pytest.fail("resident mapping was lost"),
    )
    assert resolved.bundle == resident_bundle
    table.assert_invariants()


def test_complete_and_fail_load_reject_fallback_results_without_publishing():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    resident = ExpertKey(0, 0)
    fallback = ExpertKey(0, 1)
    table.resolve(resident, lambda _: bundle(1, "resident"))
    table.pin(resident)

    result = table.begin_load(fallback)

    assert result.status is LoadStatus.FALLBACK
    assert result.lease is None
    with pytest.raises(RuntimeError):
        table.complete_load(result.lease, bundle(2, "candidate"))
    with pytest.raises(RuntimeError):
        table.fail_load(result.lease)
    assert table.slot_for(fallback) is None
    resolved = table.resolve(
        resident,
        lambda _: pytest.fail("resident mapping was lost"),
    )
    assert resolved.status is LoadStatus.HIT
    table.assert_invariants()


def test_complete_and_fail_load_accept_reserved_results():
    table = ExpertResidencyTable(capacity=2, validate_bundle=lambda _, value: True)
    completed = ExpertKey(0, 0)
    failed = ExpertKey(0, 1)

    complete_reservation = table.begin_load(completed)
    assert complete_reservation.status is LoadStatus.RESERVED
    assert complete_reservation.lease is not None
    table.complete_load(complete_reservation.lease, bundle(1, "complete"))
    assert table.slot_for(completed) == complete_reservation.slot
    assert table.slot_state(complete_reservation.slot) is SlotState.RESIDENT

    fail_reservation = table.begin_load(failed)
    assert fail_reservation.status is LoadStatus.RESERVED
    assert fail_reservation.lease is not None
    table.fail_load(fail_reservation.lease)
    assert table.slot_for(failed) is None
    assert table.slot_state(fail_reservation.slot) is SlotState.FAILED
    table.assert_invariants()


def test_stale_lease_cannot_finalize_a_re_reserved_key():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    key = ExpertKey(0, 0)
    first = table.begin_load(key)

    table.fail_load(first.lease)
    second = table.begin_load(key)
    accounting = ResidencyAccounting(*astuple(table.accounting))
    events = table.events()

    assert first.lease != second.lease
    assert first.lease is not None
    assert second.lease is not None
    assert first.lease.token < second.lease.token
    forged = replace(second.lease)
    mismatched = replace(second.lease, token=second.lease.token + 1)
    with pytest.raises(RuntimeError, match="current reservation"):
        table.complete_load(forged, bundle(1, "forged"))
    with pytest.raises(RuntimeError, match="current reservation"):
        table.complete_load(mismatched, bundle(1, "mismatched"))
    with pytest.raises(RuntimeError, match="current reservation"):
        table.complete_load(first.lease, bundle(1, "stale"))
    with pytest.raises(RuntimeError, match="current reservation"):
        table.fail_load(first.lease)
    with pytest.raises(RuntimeError, match="current reservation"):
        table.fail_load(forged)
    with pytest.raises(RuntimeError, match="current reservation"):
        table.fail_load(mismatched)
    assert table.accounting == accounting
    assert table.events() == events
    assert table.slot_for(key) is None
    assert table.slot_state(second.slot) is SlotState.LOADING

    loaded = table.complete_load(second.lease, bundle(2, "current"))

    assert loaded.status is LoadStatus.LOADED
    assert table.slot_for(key) == second.slot
    table.assert_invariants()


def test_reservation_leases_are_immutable_and_tokens_increase_globally():
    table = ExpertResidencyTable(capacity=2, validate_bundle=lambda _, value: True)
    first = table.begin_load(ExpertKey(0, 0))
    second = table.begin_load(ExpertKey(0, 1))

    assert first.lease is not None
    assert second.lease is not None
    with pytest.raises(AttributeError):
        first.lease.token = 0
    assert first.lease.token < second.lease.token
    assert len({first.lease.token, second.lease.token}) == 2

    table.fail_load(first.lease)
    table.fail_load(second.lease)
    table.assert_invariants()


def test_double_finalization_rejects_without_mutating_state():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    reservation = table.begin_load(ExpertKey(0, 0))
    assert reservation.lease is not None
    table.complete_load(reservation.lease, bundle(1, "complete"))
    accounting = ResidencyAccounting(*astuple(table.accounting))
    events = table.events()
    mapping = table.published_slots()

    with pytest.raises(RuntimeError, match="current reservation"):
        table.fail_load(reservation.lease)
    with pytest.raises(RuntimeError, match="current reservation"):
        table.complete_load(reservation.lease, bundle(2, "duplicate"))

    assert table.accounting == accounting
    assert table.events() == events
    assert table.published_slots() == mapping
    table.assert_invariants()


def test_loading_reservation_is_not_a_replacement_victim():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    loading = ExpertKey(0, 0)
    incoming = ExpertKey(0, 1)
    reservation = table.begin_load(loading)

    fallback = table.begin_load(incoming)

    assert fallback.status is LoadStatus.FALLBACK
    assert table.slot_state(reservation.slot) is SlotState.LOADING
    assert table.slot_for(loading) is None
    assert reservation.lease is not None
    table.fail_load(reservation.lease)
    table.assert_invariants()


def test_load_callback_exception_falls_back_without_losing_existing_mapping():
    table = ExpertResidencyTable(capacity=2, validate_bundle=lambda _, value: True)
    existing = ExpertKey(0, 0)
    failed = ExpertKey(0, 1)
    table.resolve(existing, lambda _: bundle(1, "resident"))

    def raise_from_load(_: ExpertKey) -> ExpertBundle:
        raise RuntimeError("load failed")

    result = table.resolve(failed, raise_from_load)
    hit = table.resolve(existing, lambda _: pytest.fail("existing mapping was lost"))

    assert result.status is LoadStatus.FALLBACK
    assert result.bundle is None
    assert table.slot_for(failed) is None
    assert table.slot_state(result.slot) is SlotState.FAILED
    assert hit.status is LoadStatus.HIT
    assert table.accounting.load_failures == 1
    table.assert_invariants()


def test_validation_callback_exception_falls_back_without_losing_existing_mapping():
    existing = ExpertKey(0, 0)
    failed = ExpertKey(0, 1)

    def validate_or_raise(key: ExpertKey, _: ExpertBundle) -> bool:
        if key == failed:
            raise RuntimeError("validation failed")
        return True

    table = ExpertResidencyTable(capacity=2, validate_bundle=validate_or_raise)
    table.resolve(existing, lambda _: bundle(1, "resident"))

    result = table.resolve(failed, lambda _: bundle(2, "candidate"))
    hit = table.resolve(existing, lambda _: pytest.fail("existing mapping was lost"))

    assert result.status is LoadStatus.FALLBACK
    assert result.bundle is None
    assert table.slot_for(failed) is None
    assert table.slot_state(result.slot) is SlotState.FAILED
    assert hit.status is LoadStatus.HIT
    assert table.accounting.load_failures == 1
    table.assert_invariants()


def test_seeded_sequence_preserves_capacity_uniqueness_and_accounting():
    table = ExpertResidencyTable(capacity=3, validate_bundle=lambda key, value: True)
    keys = [
        ExpertKey(layer_id=index % 2, logical_expert_id=index // 2)
        for index in range(8)
    ]
    random = Random(7)
    successful_loads = 0

    for generation in range(100):
        key = random.choice(keys)
        result = table.resolve(key, lambda _, g=generation: bundle(g))
        if result.status is LoadStatus.LOADED:
            successful_loads += 1
        table.assert_invariants()
        assert len(table.published_slots()) <= 3
        assert len(table.published_slots()) == len(
            set(table.published_slots().values())
        )

    assert table.accounting.requests == 100
    assert table.accounting.hits + table.accounting.misses == 100
    assert table.accounting.load_successes == successful_loads
    assert table.accounting.load_failures == 0


def test_full_replacement_loader_exception_preserves_victim_bundle_and_mapping():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    victim = ExpertKey(0, 0)
    requested = ExpertKey(0, 1)
    original = bundle(1, "original")
    table.resolve(victim, lambda _: original)

    def raise_from_load(_: ExpertKey) -> ExpertBundle:
        raise RuntimeError("load failed")

    result = table.resolve(requested, raise_from_load)
    hit = table.resolve(victim, lambda _: pytest.fail("victim mapping was lost"))

    assert result.status is LoadStatus.FALLBACK
    assert table.slot_for(requested) is None
    assert hit.status is LoadStatus.HIT
    assert hit.bundle == original
    assert table.slot_state(hit.slot) is SlotState.RESIDENT
    assert table.accounting.evictions == 0
    assert table.accounting.load_failures == 1
    table.assert_invariants()


def test_full_replacement_validator_exception_preserves_victim_bundle_and_mapping():
    victim = ExpertKey(0, 0)
    requested = ExpertKey(0, 1)
    original = bundle(1, "original")

    def validate_or_raise(key: ExpertKey, _: ExpertBundle) -> bool:
        if key == requested:
            raise RuntimeError("validation failed")
        return True

    table = ExpertResidencyTable(capacity=1, validate_bundle=validate_or_raise)
    table.resolve(victim, lambda _: original)

    result = table.resolve(requested, lambda _: bundle(2, "candidate"))
    hit = table.resolve(victim, lambda _: pytest.fail("victim mapping was lost"))

    assert result.status is LoadStatus.FALLBACK
    assert table.slot_for(requested) is None
    assert hit.status is LoadStatus.HIT
    assert hit.bundle == original
    assert table.slot_state(hit.slot) is SlotState.RESIDENT
    assert table.accounting.evictions == 0
    assert table.accounting.load_failures == 1
    table.assert_invariants()


def test_staged_replacement_falls_back_when_victim_is_pinned_before_completion():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    victim = ExpertKey(0, 0)
    incoming = ExpertKey(0, 1)
    original = bundle(1, "original")
    candidate = bundle(2, "candidate")
    table.resolve(victim, lambda _: original)

    reservation = table.begin_load(incoming)
    assert reservation.status is LoadStatus.RESERVED
    assert reservation.slot == table.slot_for(victim)
    table.pin(victim)

    assert reservation.lease is not None
    result = table.complete_load(reservation.lease, candidate)
    hit = table.begin_load(victim)

    assert result.status is LoadStatus.FALLBACK
    assert result.slot == reservation.slot
    assert result.bundle is None
    assert hit.status is LoadStatus.HIT
    assert hit.slot == reservation.slot
    assert hit.bundle == original
    assert table.slot_for(incoming) is None
    assert table.slot_state(reservation.slot) is SlotState.RESIDENT
    assert table.pin_count(victim) == 1
    assert table.accounting.evictions == 0
    assert table.accounting.loads == 2
    assert table.accounting.load_failures == 1
    assert EventKind.STAGED_ROLLBACK in [event.kind for event in table.events()]
    assert table.accounting.loads == (
        table.accounting.load_successes + table.accounting.load_failures
    )

    table.unpin(victim)
    retry = table.begin_load(incoming)
    assert retry.lease is not None
    loaded = table.complete_load(retry.lease, candidate)

    assert retry.status is LoadStatus.RESERVED
    assert retry.slot == reservation.slot
    assert loaded.status is LoadStatus.LOADED
    assert table.slot_for(victim) is None
    assert table.slot_for(incoming) == reservation.slot
    assert table.accounting.evictions == 1
    assert table.accounting.loads == 3
    assert table.accounting.load_successes == 2
    assert table.accounting.load_failures == 1
    table.assert_invariants()


def test_full_replacement_publishes_candidate_and_counts_one_eviction():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    victim = ExpertKey(0, 0)
    requested = ExpertKey(0, 1)
    table.resolve(victim, lambda _: bundle(1, "original"))

    result = table.resolve(requested, lambda _: bundle(2, "candidate"))

    assert result.status is LoadStatus.LOADED
    assert table.slot_for(victim) is None
    assert table.slot_for(requested) == result.slot
    assert result.bundle == bundle(2, "candidate")
    assert table.accounting.evictions == 1
    table.assert_invariants()


def test_accounting_counts_only_finalized_load_attempts():
    table = ExpertResidencyTable(capacity=3, validate_bundle=lambda _, value: True)
    successful = ExpertKey(0, 0)
    pending = ExpertKey(0, 1)
    failed = ExpertKey(0, 2)

    table.resolve(successful, lambda _: bundle(1))
    reservation = table.begin_load(pending)
    assert reservation.status is LoadStatus.RESERVED
    failed_reservation = table.begin_load(failed)
    assert failed_reservation.lease is not None
    table.fail_load(failed_reservation.lease)

    assert table.accounting.loads == 2
    assert table.accounting.loads == (
        table.accounting.load_successes + table.accounting.load_failures
    )
    assert table.accounting.load_successes == 1
    assert table.accounting.load_failures == 1
    table.assert_invariants()


def test_assert_invariants_rejects_negative_finalized_loads():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    table.accounting = ResidencyAccounting(loads=-1)

    with pytest.raises(AssertionError):
        table.assert_invariants()


def test_events_are_immutable_snapshots_of_request_load_and_hit_decisions():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    key = ExpertKey(3, 2)

    table.resolve(key, lambda _: bundle(7, "complete"))
    snapshot = table.events()
    table.resolve(key, lambda _: pytest.fail("warm hit must not load"))
    events = table.events()

    assert snapshot == events[:4]
    assert isinstance(events, tuple)
    assert [(event.kind, event.key, event.slot) for event in events] == [
        (EventKind.REQUEST, key, None),
        (EventKind.MISS, key, None),
        (EventKind.RESERVATION, key, 0),
        (EventKind.LOAD_SUCCESS, key, 0),
        (EventKind.REQUEST, key, 0),
        (EventKind.HIT, key, 0),
    ]
    with pytest.raises(AttributeError):
        events[0].slot = 9


def test_ordinary_loading_reservation_is_not_published_until_completion():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    key = ExpertKey(0, 4)

    reservation = table.begin_load(key)

    assert reservation.status is LoadStatus.RESERVED
    assert reservation.slot == 0
    assert table.slot_for(key) is None
    assert table.slot_state(reservation.slot) is SlotState.LOADING
    assert reservation.lease is not None
    table.complete_load(reservation.lease, bundle(1, "complete"))


def test_lru_replacement_evicts_untouched_resident_after_a_warm_touch():
    table = ExpertResidencyTable(capacity=2, validate_bundle=lambda _, value: True)
    oldest = ExpertKey(0, 0)
    touched = ExpertKey(0, 1)
    incoming = ExpertKey(0, 2)
    table.resolve(oldest, lambda _: bundle(1, "oldest"))
    table.resolve(touched, lambda _: bundle(2, "touched"))
    table.resolve(touched, lambda _: pytest.fail("warm touch must not load"))

    result = table.resolve(incoming, lambda _: bundle(3, "incoming"))

    assert result.status is LoadStatus.LOADED
    assert table.slot_for(oldest) is None
    assert table.slot_for(touched) is not None
    assert table.slot_for(incoming) == 0
    assert [event.kind for event in table.events()][-4:] == [
        EventKind.MISS,
        EventKind.RESERVATION,
        EventKind.EVICTION,
        EventKind.LOAD_SUCCESS,
    ]


def test_non_bundle_completion_fails_closed_without_publishing():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    key = ExpertKey(1, 1)
    reservation = table.begin_load(key)

    assert reservation.lease is not None
    result = table.complete_load(reservation.lease, object())

    assert result.status is LoadStatus.FALLBACK
    assert result.slot == reservation.slot
    assert table.slot_for(key) is None
    assert table.slot_state(reservation.slot) is SlotState.FAILED
    assert table.accounting.load_failures == 1
    assert [event.kind for event in table.events()][-1] is EventKind.LOAD_FAILURE


def test_duplicate_and_pin_blocked_requests_have_explicit_event_decisions():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, value: True)
    resident = ExpertKey(0, 0)
    pending = ExpertKey(0, 1)
    table.resolve(resident, lambda _: bundle(1, "resident"))
    table.pin(resident)
    fallback = table.begin_load(pending)
    assert fallback.status is LoadStatus.FALLBACK
    table.unpin(resident)
    table.begin_load(pending)

    with pytest.raises(RuntimeError, match="already loading"):
        table.begin_load(pending)

    kinds = [event.kind for event in table.events()]
    assert EventKind.CAPACITY_PIN_BLOCKED_FALLBACK in kinds
    assert EventKind.DUPLICATE_LOAD_REJECTION in kinds


def test_validator_reentrancy_keeps_inner_completion_as_only_success():
    key = ExpertKey(0, 0)
    inner_bundle = bundle(1, "inner")
    outer_bundle = bundle(2, "outer")
    reentered = False

    def validate(_: ExpertKey, __: ExpertBundle) -> bool:
        nonlocal reentered
        if not reentered:
            reentered = True
            table.complete_load(lease, inner_bundle)
        return True

    table = ExpertResidencyTable(capacity=1, validate_bundle=validate)
    reservation = table.begin_load(key)
    assert reservation.lease is not None
    lease = reservation.lease

    with pytest.raises(RuntimeError, match="current reservation"):
        table.complete_load(lease, outer_bundle)

    assert table.slot_for(key) == lease.slot
    assert table.slot_state(lease.slot) is SlotState.RESIDENT
    inner_hit = table.resolve(
        key, lambda _: pytest.fail("inner bundle was replaced")
    )
    assert inner_hit.bundle == inner_bundle
    assert table.accounting == ResidencyAccounting(
        requests=2, hits=1, misses=1, loads=1, load_successes=1
    )
    assert [(event.kind, event.key, event.slot, event.staged_replacement)
            for event in table.events()] == [
        (EventKind.REQUEST, key, None, False),
        (EventKind.MISS, key, None, False),
        (EventKind.RESERVATION, key, 0, False),
        (EventKind.LOAD_SUCCESS, key, 0, False),
        (EventKind.REQUEST, key, 0, False),
        (EventKind.HIT, key, 0, False),
    ]
    table.assert_invariants()


def test_resolve_load_callback_reentrancy_rejects_outer_stale_completion():
    key = ExpertKey(0, 0)
    inner_bundle = bundle(1, "inner")
    outer_bundle = bundle(2, "outer")

    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, __: True)

    def load(_: ExpertKey) -> ExpertBundle:
        lease = table._loading[key]
        table.complete_load(lease, inner_bundle)
        return outer_bundle

    with pytest.raises(RuntimeError, match="current reservation"):
        table.resolve(key, load)

    assert table.slot_for(key) == 0
    assert table.slot_state(0) is SlotState.RESIDENT
    inner_hit = table.resolve(
        key, lambda _: pytest.fail("inner bundle was replaced")
    )
    assert inner_hit.bundle == inner_bundle
    assert table.accounting == ResidencyAccounting(
        requests=2, hits=1, misses=1, loads=1, load_successes=1
    )
    assert [(event.kind, event.key, event.slot, event.staged_replacement)
            for event in table.events()] == [
        (EventKind.REQUEST, key, None, False),
        (EventKind.MISS, key, None, False),
        (EventKind.RESERVATION, key, 0, False),
        (EventKind.LOAD_SUCCESS, key, 0, False),
        (EventKind.REQUEST, key, 0, False),
        (EventKind.HIT, key, 0, False),
    ]
    table.assert_invariants()


def test_validator_callback_reentrancy_then_raise_preserves_inner_completion():
    key = ExpertKey(0, 0)
    inner_bundle = bundle(1, "inner")
    outer_bundle = bundle(2, "outer")
    reentered = False

    def validate(_: ExpertKey, __: ExpertBundle) -> bool:
        nonlocal reentered
        if not reentered:
            reentered = True
            table.complete_load(lease, inner_bundle)
            raise RuntimeError("validator failed after inner completion")
        return True

    table = ExpertResidencyTable(capacity=1, validate_bundle=validate)
    reservation = table.begin_load(key)
    assert reservation.lease is not None
    lease = reservation.lease

    with pytest.raises(RuntimeError, match="current reservation"):
        table.complete_load(lease, outer_bundle)

    assert table.slot_for(key) == lease.slot
    assert table.slot_state(lease.slot) is SlotState.RESIDENT
    inner_hit = table.resolve(
        key, lambda _: pytest.fail("inner bundle was replaced")
    )
    assert inner_hit.bundle == inner_bundle
    assert table.accounting == ResidencyAccounting(
        requests=2, hits=1, misses=1, loads=1, load_successes=1
    )
    assert [(event.kind, event.key, event.slot, event.staged_replacement)
            for event in table.events()] == [
        (EventKind.REQUEST, key, None, False),
        (EventKind.MISS, key, None, False),
        (EventKind.RESERVATION, key, 0, False),
        (EventKind.LOAD_SUCCESS, key, 0, False),
        (EventKind.REQUEST, key, 0, False),
        (EventKind.HIT, key, 0, False),
    ]
    table.assert_invariants()


def test_loader_callback_reentrancy_then_raise_preserves_inner_completion():
    key = ExpertKey(0, 0)
    inner_bundle = bundle(1, "inner")

    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, __: True)

    def load(_: ExpertKey) -> ExpertBundle:
        lease = table._loading[key]
        table.complete_load(lease, inner_bundle)
        raise RuntimeError("loader failed after inner completion")

    with pytest.raises(RuntimeError, match="current reservation"):
        table.resolve(key, load)

    assert table.slot_for(key) == 0
    assert table.slot_state(0) is SlotState.RESIDENT
    inner_hit = table.resolve(
        key, lambda _: pytest.fail("inner bundle was replaced")
    )
    assert inner_hit.bundle == inner_bundle
    assert table.accounting == ResidencyAccounting(
        requests=2, hits=1, misses=1, loads=1, load_successes=1
    )
    assert [(event.kind, event.key, event.slot, event.staged_replacement)
            for event in table.events()] == [
        (EventKind.REQUEST, key, None, False),
        (EventKind.MISS, key, None, False),
        (EventKind.RESERVATION, key, 0, False),
        (EventKind.LOAD_SUCCESS, key, 0, False),
        (EventKind.REQUEST, key, 0, False),
        (EventKind.HIT, key, 0, False),
    ]
    table.assert_invariants()


def test_invalid_unpin_operations_leave_residency_state_unchanged():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, __: True)
    resident = ExpertKey(0, 0)
    unknown = ExpertKey(0, 1)
    table.resolve(resident, lambda _: bundle(1, "resident"))
    before = (table.published_slots(), table.events(), replace(table.accounting))

    with pytest.raises(KeyError, match="not resident"):
        table.unpin(unknown)
    assert (
        table.published_slots(),
        table.events(),
        replace(table.accounting),
    ) == before
    with pytest.raises(ValueError, match="cannot unpin"):
        table.unpin(resident)
    assert (
        table.published_slots(),
        table.events(),
        replace(table.accounting),
    ) == before
    table.pin(resident)
    table.unpin(resident)
    after_valid_unpin = (
        table.published_slots(),
        table.events(),
        replace(table.accounting),
    )
    with pytest.raises(ValueError, match="cannot unpin"):
        table.unpin(resident)

    assert before == after_valid_unpin
    assert (
        table.published_slots(),
        table.events(),
        replace(table.accounting),
    ) == before
    table.assert_invariants()


def test_boundary_events_include_exact_payloads_for_rollbacks_and_rejections():
    table = ExpertResidencyTable(capacity=1, validate_bundle=lambda _, __: True)
    victim = ExpertKey(0, 0)
    incoming = ExpertKey(0, 1)
    table.resolve(victim, lambda _: bundle(1, "victim"))
    rollback = table.begin_load(incoming)
    assert rollback.lease is not None
    table.pin(victim)
    table.complete_load(rollback.lease, bundle(2, "candidate"))
    assert [(event.kind, event.key, event.slot, event.staged_replacement)
            for event in table.events()][-3:] == [
        (EventKind.RESERVATION, incoming, 0, True),
        (EventKind.LOAD_FAILURE, incoming, 0, True),
        (EventKind.STAGED_ROLLBACK, incoming, 0, True),
    ]
    table.unpin(victim)
    pending = table.begin_load(incoming)
    with pytest.raises(RuntimeError, match="already loading"):
        table.begin_load(incoming)
    assert [(event.kind, event.key, event.slot, event.staged_replacement)
            for event in table.events()][-4:] == [
        (EventKind.RESERVATION, incoming, 0, True),
        (EventKind.REQUEST, incoming, None, False),
        (EventKind.MISS, incoming, None, False),
        (EventKind.DUPLICATE_LOAD_REJECTION, incoming, 0, True),
    ]
    table.fail_load(pending.lease)
    table.pin(victim)
    fallback = table.begin_load(incoming)
    assert fallback.status is LoadStatus.FALLBACK
    assert [(event.kind, event.key, event.slot, event.staged_replacement)
            for event in table.events()][-3:] == [
        (EventKind.REQUEST, incoming, None, False),
        (EventKind.MISS, incoming, None, False),
        (EventKind.CAPACITY_PIN_BLOCKED_FALLBACK, incoming, None, False),
    ]
    table.assert_invariants()


EXPERT_PAYLOADS = {
    ExpertKey(layer, expert): (layer + 1) * 10 + expert
    for layer in range(2)
    for expert in range(4)
}
TOP_K_ROUTES = (
    ((0, 2), (1, 3)),
    ((3, 1), (0, 2)),
    ((2, 3), (1, 1)),
    ((1, 2), (3, 2)),
)


def _all_resident_output(layer: int, value: int, route_index: int) -> int:
    return sum(
        weight * (value * EXPERT_PAYLOADS[ExpertKey(layer, expert)] + expert)
        for expert, weight in TOP_K_ROUTES[route_index]
    )


def _resident_output(
    tables: dict[int, ExpertResidencyTable],
    layer: int,
    value: int,
    route_index: int,
) -> int:
    def load(key: ExpertKey) -> ExpertBundle:
        return ExpertBundle(
            generation=key.logical_expert_id,
            payload=EXPERT_PAYLOADS[key],
        )

    return sum(
        weight
        * (
            value * tables[layer].resolve(ExpertKey(layer, expert), load).bundle.payload
            + expert
        )
        for expert, weight in TOP_K_ROUTES[route_index]
    )


def test_two_layer_synthetic_moe_matches_all_resident_across_residency_paths():
    tables = {
        layer: ExpertResidencyTable(capacity=2, validate_bundle=lambda _, value: True)
        for layer in range(2)
    }
    sequence = [(0, 4, 0), (0, 4, 0), (0, 5, 1), (0, 6, 2), (1, 4, 0)]
    sequence.extend((index % 2, index + 2, index % 4) for index in range(20))

    outputs = [
        _resident_output(tables, layer, value, route)
        for layer, value, route in sequence
    ]
    expected = [
        _all_resident_output(layer, value, route)
        for layer, value, route in sequence
    ]

    assert outputs == expected
    assert EXPERT_PAYLOADS[ExpertKey(0, 0)] != EXPERT_PAYLOADS[ExpertKey(1, 0)]
    assert _all_resident_output(0, 4, 0) != _all_resident_output(1, 4, 0)
    first_layer_kinds = [event.kind for event in tables[0].events()]
    assert EventKind.MISS in first_layer_kinds
    assert EventKind.LOAD_SUCCESS in first_layer_kinds
    assert EventKind.HIT in first_layer_kinds
    assert EventKind.EVICTION in first_layer_kinds
    assert tables[0].events() != tables[1].events()
