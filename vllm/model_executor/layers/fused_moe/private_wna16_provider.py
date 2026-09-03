# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Controller-owned binding contract for private WNA16 generation views.

The registry is deliberately process-local and stores a factory, never a
construction-time view. A worker/controller registers its factory during
bootstrap; model construction consumes it only for an explicitly selected
private layer. The provider is called after routing with that request's router
outputs, so it cannot truthfully be implemented as a static model tensor view.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Protocol

import torch

from vllm.model_executor.layers.fused_moe.expert_residency import (
    Phase4FailureCategory,
    Phase4UnsupportedError,
    WNA16GenerationView,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


class PrivateWNA16GenerationViewProvider(Protocol):
    """Acquire the controller's immutable view for one routed dispatch."""

    def __call__(
        self,
        *,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> WNA16GenerationView:
        """Return the current request-local, controller-owned generation view."""
        ...


@dataclass(frozen=True, slots=True)
class PrivateWNA16ProviderBindingRequest:
    """Immutable model-construction facts validated by a controller factory."""

    layer_id: int
    routed_experts: RoutedExperts


class PrivateWNA16ProviderFactory(Protocol):
    """Worker/controller factory for one selected private WNA16 layer."""

    def __call__(
        self,
        request: PrivateWNA16ProviderBindingRequest,
    ) -> PrivateWNA16GenerationViewProvider:
        """Validate ownership/layout and bind a dynamic per-dispatch provider."""
        ...


@dataclass(slots=True)
class _ActiveRegistration:
    factory: PrivateWNA16ProviderFactory
    closed: bool = False


_registry_lock = RLock()
_active_registration: _ActiveRegistration | None = None


def _lifecycle_error(message: str, *, stale: bool = False) -> Phase4UnsupportedError:
    return Phase4UnsupportedError(
        (
            Phase4FailureCategory.STALE_LEASE
            if stale
            else Phase4FailureCategory.UNSUPPORTED_DYNAMIC_MAP
        ),
        message,
    )


@dataclass(slots=True)
class PrivateWNA16ProviderFactoryRegistration:
    """Controller-owned registration lifetime; close during controller shutdown."""

    _registration: _ActiveRegistration

    def close(self) -> None:
        """Revoke this registration without disturbing a newer registration."""
        global _active_registration
        with _registry_lock:
            if self._registration.closed:
                return
            self._registration.closed = True
            if _active_registration is self._registration:
                _active_registration = None


@dataclass(frozen=True, slots=True)
class _RevocablePrivateWNA16Provider:
    registration: PrivateWNA16ProviderFactoryRegistration
    provider: PrivateWNA16GenerationViewProvider

    def __call__(
        self,
        *,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> WNA16GenerationView:
        with _registry_lock:
            if self.registration._registration.closed:
                raise _lifecycle_error(
                    "private WNA16 provider binding has been revoked", stale=True
                )
        try:
            return self.provider(topk_ids=topk_ids, topk_weights=topk_weights)
        except Phase4UnsupportedError:
            raise
        except Exception as error:
            raise _lifecycle_error(
                "private WNA16 controller provider failed during dispatch"
            ) from error


class PrivateWNA16ResidencyController:
    """Per-worker placeholder for the future CPU-to-GPU transfer controller."""

    def __init__(self, layer_id: int) -> None:
        self._layer_id = layer_id

    def make_provider(
        self, request: PrivateWNA16ProviderBindingRequest
    ) -> PrivateWNA16GenerationViewProvider:
        """Bind only the configured layer and fail closed until transfer exists."""
        if request.layer_id != self._layer_id:
            raise _lifecycle_error("private WNA16 controller received another layer")
        return self._request_generation_view

    def _request_generation_view(
        self,
        *,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> WNA16GenerationView:
        del topk_ids, topk_weights
        raise _lifecycle_error("CPU-to-GPU private WNA16 handoff is not implemented")


def register_private_wna16_provider_factory(
    factory: PrivateWNA16ProviderFactory,
) -> PrivateWNA16ProviderFactoryRegistration:
    """Register exactly one worker-local controller factory."""
    global _active_registration
    if not callable(factory):
        raise _lifecycle_error("private WNA16 provider factory must be callable")
    with _registry_lock:
        if _active_registration is not None:
            raise _lifecycle_error(
                "a private WNA16 provider factory is already registered"
            )
        registration = _ActiveRegistration(factory)
        _active_registration = registration
        return PrivateWNA16ProviderFactoryRegistration(registration)


def bind_private_wna16_generation_view_provider(
    *, layer_id: int, routed_experts: RoutedExperts
) -> None:
    """Bind the selected layer or fail closed before it can serve requests."""
    with _registry_lock:
        registration = _active_registration
    if registration is None:
        raise _lifecycle_error(
            "private WNA16 dispatch requires a registered controller provider factory"
        )
    try:
        provider = registration.factory(
            PrivateWNA16ProviderBindingRequest(
                layer_id=layer_id, routed_experts=routed_experts
            )
        )
    except Phase4UnsupportedError:
        raise
    except Exception as error:
        raise _lifecycle_error(
            "private WNA16 controller factory failed during binding"
        ) from error
    if not callable(provider):
        raise _lifecycle_error(
            "private WNA16 controller factory must return a callable provider"
        )
    with _registry_lock:
        if _active_registration is not registration:
            raise _lifecycle_error(
                "private WNA16 provider registration was revoked during binding",
                stale=True,
            )
        binding = PrivateWNA16ProviderFactoryRegistration(registration)
    routed_experts.set_private_wna16_generation_view_provider(
        _RevocablePrivateWNA16Provider(binding, provider), layer_id=layer_id
    )
