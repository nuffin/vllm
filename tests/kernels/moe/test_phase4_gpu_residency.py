# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
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
    bind_private_wna16_generation_view_provider,
    register_private_wna16_provider_factory,
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


def unbound_private_view() -> WNA16GenerationView:
    """A public ABI view has snapshots but cannot issue private storage."""
    view = private_view()
    object.__setattr__(view, "_stable_slot_storage", None)
    return view


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
    lease = WNA16UseLease(
        layer_id=3, generation=9, bundle=private_bundle, token=1
    )
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
