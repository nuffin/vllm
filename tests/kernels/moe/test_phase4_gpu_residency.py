# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_residency import (
    Phase4FailureCategory,
    Phase4GpuResidencyAdapter,
    Phase4UnsupportedError,
    PostConversionExpertBundle,
    TorchCpuTransferReference,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


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
    layer.quant_method = MagicMock(is_monolithic=False)
    layer.quant_method.apply.return_value = torch.tensor([1.0])
    layer.expert_map_manager = MagicMock()
    return layer


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

    assert (
        adapter.capability().category
        is Phase4FailureCategory.UNSUPPORTED_DYNAMIC_MAP
    )
    with pytest.raises(Phase4UnsupportedError, match="not implemented"):
        adapter.validate_request(torch.tensor([[2, 0]]), torch.ones(1, 2))


def test_cpu_reference_transfers_complete_post_conversion_bundle():
    source = complete_bundle()

    transferred = TorchCpuTransferReference.transfer(source)

    assert all(t.device.type == "cpu" for t in transferred.tensors.values())
    assert all(
        source.tensors[name].equal(transferred.tensors[name])
        for name in source.tensors
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


def test_forward_modular_adapter_rejects_before_apply_without_mutation():
    layer = mocked_routed_experts(Phase4GpuResidencyAdapter())
    x = torch.ones(1, 2)
    weights = torch.tensor([[0.25, 0.75]])
    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    ids_before, weights_before = ids.clone(), weights.clone()
    manager_before = layer.expert_map_manager

    with pytest.raises(Phase4UnsupportedError):
        layer.forward_modular(x, weights, ids)

    assert torch.equal(ids, ids_before)
    assert torch.equal(weights, weights_before)
    assert layer.expert_map_manager is manager_before
    layer.quant_method.apply.assert_not_called()
