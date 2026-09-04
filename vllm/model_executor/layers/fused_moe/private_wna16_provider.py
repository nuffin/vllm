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

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from threading import RLock
from typing import TYPE_CHECKING, Protocol, cast

import torch

from vllm.model_executor.layers.fused_moe.expert_residency import (
    Phase4FailureCategory,
    Phase4UnsupportedError,
    WNA16GenerationView,
    _bind_controller_wna16_slot_storage,
    _new_controller_wna16_stable_slot_capability,
    _WNA16StableSlotCapability,
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


@dataclass(frozen=True, slots=True)
class _CanonicalWNA16CpuBank:
    """Controller-private detached CPU source for a future transfer controller."""

    layer_id: int
    global_num_experts: int
    num_bits: int
    symmetric: bool
    group_size: int
    w13_weight_packed: torch.Tensor
    w2_weight_packed: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor
    w13_weight_zero_point: torch.Tensor | None
    w2_weight_zero_point: torch.Tensor | None


def _copy_canonical_wna16_cpu_operand(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").clone().contiguous()


class _WNA16RequestUseState(Enum):
    ISSUED = auto()
    ACQUIRED = auto()
    PRE_ENQUEUE = auto()
    ENQUEUING = auto()
    FENCED_PENDING = auto()
    UNFENCED_QUARANTINED = auto()
    RELEASED = auto()


@dataclass(slots=True)
class _WNA16RequestUseLease:
    """Unforgeable-by-public-ABI request use authority for one bound view."""

    manager: _WNA16RequestUseLeaseManager
    view: WNA16GenerationView
    state: _WNA16RequestUseState = _WNA16RequestUseState.ISSUED
    completion_event: object | None = None
    query_in_progress: bool = False


class _WNA16RequestUseLeaseManager:
    """Own pending request uses without introducing a reaper or synchronization."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._closed = False
        self._by_id: dict[int, _WNA16RequestUseLease] = {}
        self._drain_authority: _WNA16RequestDrainAuthority | None = None

    def set_drain_authority(self, authority: _WNA16RequestDrainAuthority) -> None:
        """Attach the one controller-owned authority for this registration."""
        with self._lock:
            if self._drain_authority is not None:
                raise _lifecycle_error("private WNA16 drain authority already exists")
            self._drain_authority = authority

    def bind(self, view: WNA16GenerationView) -> WNA16GenerationView:
        with self._lock:
            if (
                self._closed
                or object.__getattribute__(view, "_request_use_lease") is not None
            ):
                raise _lifecycle_error(
                    "private WNA16 request use authority is stale", stale=True
                )
            lease = _WNA16RequestUseLease(self, view)
            self._by_id[id(lease)] = lease
            object.__setattr__(view, "_request_use_lease", lease)
            return view

    def acquire(self, view: WNA16GenerationView) -> _WNA16RequestUseLease:
        with self._lock:
            lease = object.__getattribute__(view, "_request_use_lease")
            if (
                self._closed
                or not isinstance(lease, _WNA16RequestUseLease)
                or lease.manager is not self
                or lease.view is not view
                or self._by_id.get(id(lease)) is not lease
                or lease.state is not _WNA16RequestUseState.ISSUED
            ):
                raise _lifecycle_error(
                    "private WNA16 request use lease is stale", stale=True
                )
            lease.state = _WNA16RequestUseState.ACQUIRED
            return lease

    def cancel_issued(self, lease: _WNA16RequestUseLease) -> None:
        """Cancel a bound view that failed before request-use acquisition."""
        with self._lock:
            self._require(lease, _WNA16RequestUseState.ISSUED)
            self._release(lease)

    def mark_pre_enqueue(self, lease: _WNA16RequestUseLease) -> None:
        with self._lock:
            self._require(lease, _WNA16RequestUseState.ACQUIRED)
            # The next admission transition shares this lock with close(). If
            # shutdown closes first, close releases this lease because no kernel
            # has entered; a later begin_enqueue() then fails stale.
            lease.state = _WNA16RequestUseState.PRE_ENQUEUE

    def begin_enqueue(self, lease: _WNA16RequestUseLease) -> None:
        """Admit the private enqueue unless shutdown won the pre-enqueue race."""
        with self._lock:
            self._require(lease, _WNA16RequestUseState.PRE_ENQUEUE)
            if self._closed:
                raise _lifecycle_error(
                    "private WNA16 enqueue admission was revoked", stale=True
                )
            lease.state = _WNA16RequestUseState.ENQUEUING

    def release_without_dispatch(self, lease: _WNA16RequestUseLease) -> None:
        with self._lock:
            self._require(lease, _WNA16RequestUseState.ACQUIRED)
            self._release(lease)

    def release_pending(self, lease: _WNA16RequestUseLease, event: object) -> None:
        with self._lock:
            self._require(lease, _WNA16RequestUseState.ENQUEUING)
            lease.completion_event = event
            lease.state = _WNA16RequestUseState.FENCED_PENDING
            finalize_now = self._closed
        if finalize_now:
            self._finalize_completed(lease)

    def quarantine_unfenced(self, lease: _WNA16RequestUseLease) -> None:
        """Retain a possibly enqueued use when attaching its fence failed."""
        with self._lock:
            self._require(lease, _WNA16RequestUseState.ENQUEUING)
            lease.state = _WNA16RequestUseState.UNFENCED_QUARANTINED

    def finalize_completed(self) -> None:
        with self._lock:
            leases = tuple(self._by_id.values())
        first_error: Phase4UnsupportedError | None = None
        for lease in leases:
            try:
                self._finalize_completed(lease)
            except Phase4UnsupportedError as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        """Close admission only after every tracked use has drained."""
        with self._lock:
            self._closed = True
            for lease in tuple(self._by_id.values()):
                if lease.state in (
                    _WNA16RequestUseState.ISSUED,
                    _WNA16RequestUseState.ACQUIRED,
                    _WNA16RequestUseState.PRE_ENQUEUE,
                ):
                    self._release(lease)
        self.finalize_completed()
        with self._lock:
            if self._by_id:
                raise _drain_incomplete_error()

    def release_with_verified_drain(
        self,
        lease: _WNA16RequestUseLease,
        authority: _WNA16RequestDrainAuthority,
        proof: _WNA16VerifiedDrainProof,
    ) -> None:
        """Release an unfenced use only after controller shutdown verification."""
        with self._lock:
            if self._drain_authority is not authority:
                raise _lifecycle_error(
                    "private WNA16 drain authority is foreign", stale=True
                )
            if not self._closed:
                raise _drain_incomplete_error()
            self._require(lease, _WNA16RequestUseState.UNFENCED_QUARANTINED)
            authority._require_exact_proof(lease, proof)
            self._release(lease)
            authority._consume_exact_proof(lease, proof)

    def _require(
        self, lease: _WNA16RequestUseLease, state: _WNA16RequestUseState
    ) -> None:
        if self._by_id.get(id(lease)) is not lease or lease.state is not state:
            raise _lifecycle_error(
                "private WNA16 request use lease is stale", stale=True
            )

    def _release(self, lease: _WNA16RequestUseLease) -> None:
        lease.state = _WNA16RequestUseState.RELEASED
        self._by_id.pop(id(lease), None)
        lease.completion_event = None

    def _finalize_completed(self, lease: _WNA16RequestUseLease) -> None:
        with self._lock:
            if (
                lease.state is not _WNA16RequestUseState.FENCED_PENDING
                or lease.query_in_progress
                or self._by_id.get(id(lease)) is not lease
            ):
                return
            event = lease.completion_event
            lease.query_in_progress = True
        completed = False
        try:
            try:
                query = getattr(event, "query", None)
                if not callable(query):
                    raise TypeError("completion event has no callable query")
                result = query()
            except Exception as error:
                raise _event_query_error("completion event query failed") from error
            if type(result) is not bool:
                raise _event_query_error("completion event query did not return bool")
            completed = result
        finally:
            with self._lock:
                lease.query_in_progress = False
                if (
                    completed
                    and self._by_id.get(id(lease)) is lease
                    and lease.state is _WNA16RequestUseState.FENCED_PENDING
                    and lease.completion_event is event
                ):
                    self._release(lease)


@dataclass(frozen=True, slots=True)
class _WNA16VerifiedDrainProof:
    """Opaque proof minted by one controller after its independent drain check."""

    manager: _WNA16RequestUseLeaseManager
    lease: _WNA16RequestUseLease
    registration_epoch: object
    nonce: object


class _WNA16RequestDrainAuthority:
    """Registration-owned authority for a controller-verified unfenced drain.

    This is not a normal completion proof: normal completion requires a recorded
    event and its ``query() is True``. This authority is only for a record
    failure after shutdown has closed admission. Registration captures the
    controller-owned check, which proves that this exact lease cannot enqueue or
    use storage; this manager neither synchronizes nor infers that fact.
    """

    def __init__(
        self,
        manager: _WNA16RequestUseLeaseManager,
        epoch: object,
        verifier: Callable[[_WNA16RequestUseLease], bool] | None,
    ) -> None:
        self._manager = manager
        self._epoch = epoch
        self._verifier = verifier
        self._proofs: dict[int, _WNA16VerifiedDrainProof] = {}

    def issue_verified_shutdown_drain_proof(
        self,
        lease: _WNA16RequestUseLease,
    ) -> _WNA16VerifiedDrainProof:
        """Mint a one-shot proof through the verifier captured at registration."""
        with self._manager._lock:
            if not self._manager._closed:
                raise _drain_incomplete_error()
            if lease.state is not _WNA16RequestUseState.UNFENCED_QUARANTINED:
                raise _drain_incomplete_error()
            self._manager._require(lease, _WNA16RequestUseState.UNFENCED_QUARANTINED)
            verifier = self._verifier
            if verifier is None or id(lease) in self._proofs:
                raise _drain_incomplete_error()
        try:
            proven = verifier(lease)
        except Exception as error:
            raise _drain_incomplete_error() from error
        if proven is not True:
            raise _drain_incomplete_error()
        with self._manager._lock:
            if not self._manager._closed:
                raise _drain_incomplete_error()
            self._manager._require(lease, _WNA16RequestUseState.UNFENCED_QUARANTINED)
            if id(lease) in self._proofs:
                raise _drain_incomplete_error()
            proof = _WNA16VerifiedDrainProof(
                self._manager, lease, self._epoch, object()
            )
            self._proofs[id(lease)] = proof
            return proof

    def release_verified_drain(
        self, lease: _WNA16RequestUseLease, proof: _WNA16VerifiedDrainProof
    ) -> None:
        self._manager.release_with_verified_drain(lease, self, proof)

    def _require_exact_proof(
        self, lease: _WNA16RequestUseLease, proof: _WNA16VerifiedDrainProof
    ) -> None:
        if (
            not isinstance(proof, _WNA16VerifiedDrainProof)
            or proof.manager is not self._manager
            or proof.lease is not lease
            or proof.registration_epoch is not self._epoch
            or self._proofs.get(id(lease)) is not proof
        ):
            raise _lifecycle_error(
                "private WNA16 verified drain proof is stale or foreign", stale=True
            )

    def _consume_exact_proof(
        self, lease: _WNA16RequestUseLease, proof: _WNA16VerifiedDrainProof
    ) -> None:
        self._require_exact_proof(lease, proof)
        self._proofs.pop(id(lease), None)


def acquire_private_wna16_request_use(
    view: WNA16GenerationView,
) -> _WNA16RequestUseLease | None:
    """Acquire controller-owned request authority when a provider issued one."""
    lease = object.__getattribute__(view, "_request_use_lease")
    if lease is None:
        return None
    if not isinstance(lease, _WNA16RequestUseLease):
        raise _lifecycle_error("private WNA16 request use lease is foreign", stale=True)
    return lease.manager.acquire(view)


def cancel_private_wna16_issued_request_use(view: WNA16GenerationView) -> None:
    """Release a controller-issued view after pre-dispatch validation failed."""
    lease = object.__getattribute__(view, "_request_use_lease")
    if lease is None:
        return
    if not isinstance(lease, _WNA16RequestUseLease):
        raise _lifecycle_error("private WNA16 request use lease is foreign", stale=True)
    lease.manager.cancel_issued(lease)


def mark_private_wna16_request_dispatched(lease: _WNA16RequestUseLease) -> None:
    lease.manager.mark_pre_enqueue(lease)


def begin_private_wna16_request_enqueue(lease: _WNA16RequestUseLease) -> None:
    lease.manager.begin_enqueue(lease)


def release_private_wna16_request_use_without_dispatch(
    lease: _WNA16RequestUseLease,
) -> None:
    lease.manager.release_without_dispatch(lease)


def release_private_wna16_request_use_pending(
    lease: _WNA16RequestUseLease, event: object
) -> None:
    lease.manager.release_pending(lease, event)


def quarantine_private_wna16_request_use_without_fence(
    lease: _WNA16RequestUseLease,
) -> None:
    lease.manager.quarantine_unfenced(lease)


@dataclass(slots=True)
class _ActiveRegistration:
    factory: PrivateWNA16ProviderFactory
    capability: _WNA16StableSlotCapability
    request_use_manager: _WNA16RequestUseLeaseManager
    request_drain_authority: _WNA16RequestDrainAuthority
    controller_close: Callable[[], None] | None
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


def _event_query_error(message: str) -> Phase4UnsupportedError:
    return Phase4UnsupportedError(Phase4FailureCategory.EVENT_QUERY, message)


def _drain_incomplete_error() -> Phase4UnsupportedError:
    return Phase4UnsupportedError(
        Phase4FailureCategory.DRAIN_INCOMPLETE,
        "private WNA16 request use drain is incomplete; storage remains owned",
    )


@dataclass(slots=True)
class PrivateWNA16ProviderFactoryRegistration:
    """Controller-owned registration lifetime; close during controller shutdown."""

    _registration: _ActiveRegistration

    def issue_verified_shutdown_drain_proof(
        self,
        lease: _WNA16RequestUseLease,
    ) -> _WNA16VerifiedDrainProof:
        """Request a shutdown-only proof from the registered controller verifier."""
        authority = self._registration.request_drain_authority
        return authority.issue_verified_shutdown_drain_proof(lease)

    def release_verified_drain(
        self, lease: _WNA16RequestUseLease, proof: _WNA16VerifiedDrainProof
    ) -> None:
        """Consume an exact proof through this registration-owned authority."""
        self._registration.request_drain_authority.release_verified_drain(lease, proof)

    def close(self) -> None:
        """Revoke admission and require a verified request-use drain.

        Closing is one-way: every call keeps provider admission revoked, while
        later calls retry event queries or recheck independently verified drain.
        A non-drained lease raises ``DRAIN_INCOMPLETE`` and retains its storage.
        """
        global _active_registration
        with _registry_lock:
            if not self._registration.closed:
                self._registration.closed = True
                self._registration.capability.revoke()
        self._registration.request_use_manager.close()
        if self._registration.controller_close is not None:
            self._registration.controller_close()
        with _registry_lock:
            if _active_registration is self._registration:
                _active_registration = None


@dataclass(frozen=True, slots=True)
class _RevocablePrivateWNA16Provider:
    registration: PrivateWNA16ProviderFactoryRegistration
    provider: PrivateWNA16GenerationViewProvider
    capability: _WNA16StableSlotCapability

    def __call__(
        self,
        *,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> WNA16GenerationView:
        request_use_manager = self.registration._registration.request_use_manager
        with _registry_lock:
            if self.registration._registration.closed:
                raise _lifecycle_error(
                    "private WNA16 provider binding has been revoked", stale=True
                )
        request_use_manager.finalize_completed()
        with _registry_lock:
            if self.registration._registration.closed:
                raise _lifecycle_error(
                    "private WNA16 provider binding has been revoked", stale=True
                )
        try:
            view = self.provider(topk_ids=topk_ids, topk_weights=topk_weights)
        except Phase4UnsupportedError:
            raise
        except Exception as error:
            raise _lifecycle_error(
                "private WNA16 controller provider failed during dispatch"
            ) from error
        with _registry_lock:
            if self.registration._registration.closed:
                raise _lifecycle_error(
                    "private WNA16 provider binding was revoked during dispatch",
                    stale=True,
                )
            view = _bind_controller_wna16_slot_storage(view, self.capability)
            return self.registration._registration.request_use_manager.bind(view)


class PrivateWNA16ResidencyController:
    """Per-worker placeholder for the future CPU-to-GPU transfer controller."""

    def __init__(self, layer_id: int) -> None:
        self._lifecycle_lock = RLock()
        self._layer_id = layer_id
        self._bound_routed_experts: RoutedExperts | None = None
        self._canonical_cpu_bank: _CanonicalWNA16CpuBank | None = None
        self._closed = False

    def __call__(
        self, request: PrivateWNA16ProviderBindingRequest
    ) -> PrivateWNA16GenerationViewProvider:
        """Remain callable so controller registration retains the factory ABI."""
        return self.make_provider(request)

    def verify_shutdown_drain(self, lease: _WNA16RequestUseLease) -> bool:
        """Fail closed until a transfer controller owns verified drain evidence.

        A future transfer controller may implement this registration-captured
        hook with exact lease, lifecycle, and one-shot verification. This
        placeholder has no such backend and never manufactures proof success.
        """
        del lease
        return False

    def make_provider(
        self, request: PrivateWNA16ProviderBindingRequest
    ) -> PrivateWNA16GenerationViewProvider:
        """Bind only the configured layer and fail closed until transfer exists."""
        if request.layer_id != self._layer_id:
            raise _lifecycle_error("private WNA16 controller received another layer")
        with self._lifecycle_lock:
            if (
                self._bound_routed_experts is not None
                and self._bound_routed_experts is not request.routed_experts
            ):
                raise _lifecycle_error(
                    "private WNA16 controller received another layer"
                )
            self._bound_routed_experts = request.routed_experts
        return self._request_generation_view

    def capture_post_conversion(
        self,
        *,
        layer: RoutedExperts,
        backend: object,
        num_bits: int,
        symmetric: bool,
        group_size: int,
        act_order: bool,
    ) -> None:
        """Publish one detached CPU snapshot after Triton conversion completes."""
        with self._lifecycle_lock:
            if self._closed:
                raise _lifecycle_error(
                    "private WNA16 canonical CPU bank is stale", stale=True
                )
            if layer is not self._bound_routed_experts:
                return
            if (
                str(getattr(backend, "value", backend)).lower() != "triton"
                or num_bits not in (4, 8)
                or group_size != 32
                or act_order
                or getattr(layer, "use_ep", False)
                or self._canonical_cpu_bank is not None
            ):
                raise _lifecycle_error(
                    "private WNA16 canonical CPU bank is unsupported"
                )
            w13_weight = getattr(layer, "w13_weight", None)
            w2_weight = getattr(layer, "w2_weight", None)
            w13_packed = getattr(layer, "w13_weight_packed", None)
            w2_packed = getattr(layer, "w2_weight_packed", None)
            w13_scale = getattr(layer, "w13_weight_scale", None)
            w2_scale = getattr(layer, "w2_weight_scale", None)
            w13_zero = getattr(layer, "w13_weight_zero_point", None)
            w2_zero = getattr(layer, "w2_weight_zero_point", None)
            required = (w13_packed, w2_packed, w13_scale, w2_scale)
            if (
                w13_weight is not w13_packed
                or w2_weight is not w2_packed
                or not all(isinstance(tensor, torch.Tensor) for tensor in required)
                or (symmetric and (w13_zero is not None or w2_zero is not None))
                or (
                    not symmetric
                    and not isinstance(w13_zero, torch.Tensor)
                    or not symmetric
                    and not isinstance(w2_zero, torch.Tensor)
                )
            ):
                raise _lifecycle_error(
                    "private WNA16 canonical CPU operands are invalid"
                )
            try:
                bank = _CanonicalWNA16CpuBank(
                    layer_id=self._layer_id,
                    global_num_experts=layer.global_num_experts,
                    num_bits=num_bits,
                    symmetric=symmetric,
                    group_size=group_size,
                    w13_weight_packed=_copy_canonical_wna16_cpu_operand(
                        cast(torch.Tensor, w13_packed)
                    ),
                    w2_weight_packed=_copy_canonical_wna16_cpu_operand(
                        cast(torch.Tensor, w2_packed)
                    ),
                    w13_weight_scale=_copy_canonical_wna16_cpu_operand(
                        cast(torch.Tensor, w13_scale)
                    ),
                    w2_weight_scale=_copy_canonical_wna16_cpu_operand(
                        cast(torch.Tensor, w2_scale)
                    ),
                    w13_weight_zero_point=_copy_canonical_wna16_cpu_operand(w13_zero)
                    if w13_zero is not None
                    else None,
                    w2_weight_zero_point=_copy_canonical_wna16_cpu_operand(w2_zero)
                    if w2_zero is not None
                    else None,
                )
            except Exception as error:
                raise _lifecycle_error(
                    "private WNA16 canonical CPU copy failed"
                ) from error
            self._canonical_cpu_bank = bank

    def close(self) -> None:
        with self._lifecycle_lock:
            self._closed = True
            self._canonical_cpu_bank = None
            self._bound_routed_experts = None

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
    shutdown_drain_verifier = getattr(factory, "verify_shutdown_drain", None)
    if shutdown_drain_verifier is not None and not callable(shutdown_drain_verifier):
        raise _lifecycle_error(
            "private WNA16 controller shutdown drain verifier must be callable"
        )
    verifier = cast(
        Callable[[_WNA16RequestUseLease], bool] | None, shutdown_drain_verifier
    )
    with _registry_lock:
        if _active_registration is not None:
            raise _lifecycle_error(
                "a private WNA16 provider factory is already registered"
            )
        manager = _WNA16RequestUseLeaseManager()
        epoch = object()
        authority = _WNA16RequestDrainAuthority(manager, epoch, verifier)
        manager.set_drain_authority(authority)
        registration = _ActiveRegistration(
            factory,
            _new_controller_wna16_stable_slot_capability(),
            manager,
            authority,
            (
                factory.close
                if isinstance(factory, PrivateWNA16ResidencyController)
                else None
            ),
        )
        _active_registration = registration
        return PrivateWNA16ProviderFactoryRegistration(registration)


def capture_private_wna16_post_conversion(
    *,
    layer: RoutedExperts,
    backend: object,
    num_bits: int,
    symmetric: bool,
    group_size: int,
    act_order: bool,
) -> None:
    """Offer finalized WNA16 operands to an active private controller only."""
    with _registry_lock:
        registration = _active_registration
        if registration is None:
            return
        if registration.closed:
            raise _lifecycle_error(
                "private WNA16 canonical CPU bank is stale", stale=True
            )
        capture = getattr(registration.factory, "capture_post_conversion", None)
    if capture is not None:
        capture(
            layer=layer,
            backend=backend,
            num_bits=num_bits,
            symmetric=symmetric,
            group_size=group_size,
            act_order=act_order,
        )


def bind_private_wna16_generation_view_provider(
    *, layer_id: int, routed_experts: RoutedExperts
) -> None:
    """Bind the selected layer or fail closed before it can serve requests."""
    with _registry_lock:
        registration = _active_registration
        if registration is None or registration.closed:
            raise _lifecycle_error(
                "private WNA16 dispatch requires a registered controller "
                "provider factory"
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
        if _active_registration is not registration or registration.closed:
            raise _lifecycle_error(
                "private WNA16 provider registration was revoked during binding",
                stale=True,
            )
        binding = PrivateWNA16ProviderFactoryRegistration(registration)
    routed_experts.set_private_wna16_generation_view_provider(
        _RevocablePrivateWNA16Provider(binding, provider, registration.capability),
        layer_id=layer_id,
    )
