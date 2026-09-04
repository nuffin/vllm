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
from contextlib import suppress
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


class _WNA16SlotState(Enum):
    ABSENT = auto()
    RESERVED_EMPTY = auto()
    RESIDENT = auto()
    STAGED_REPLACEMENT = auto()


class _WNA16SlotPlanStatus(Enum):
    HIT = auto()
    RESERVED = auto()
    FALLBACK = auto()


@dataclass(slots=True)
class _WNA16LogicalSlot:
    state: _WNA16SlotState = _WNA16SlotState.ABSENT
    expert_id: int | None = None
    pins: int = 0
    last_used: int = 0


@dataclass(frozen=True, slots=True)
class _WNA16LogicalSlotSnapshot:
    state: _WNA16SlotState
    expert_id: int | None
    pins: int
    last_used: int


@dataclass(frozen=True, slots=True)
class _WNA16SlotReservation:
    controller: object
    slot: int
    expert_id: int
    token: object
    previous_slot: _WNA16LogicalSlotSnapshot


@dataclass(frozen=True, slots=True)
class _WNA16CpuSourceSlotRetention:
    """CPU-only future-transfer ownership held until a predicate proves release."""

    controller: object
    source: _CanonicalWNA16CpuBank
    reservation: _WNA16SlotReservation
    completion: object
    token: object


@dataclass(frozen=True, slots=True)
class _WNA16StagedCudaSlot:
    """Controller-private destination ABI for future CUDA slot publication.

    This is deliberately only a staging boundary. It validates a complete
    controller-owned destination bundle and map before retaining the CPU source,
    but neither allocates CUDA storage nor exposes a generation view.
    """

    w13_weight_packed: torch.Tensor
    w2_weight_packed: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor
    w13_weight_zero_point: torch.Tensor | None
    w2_weight_zero_point: torch.Tensor | None
    slot_map: torch.Tensor


@dataclass(frozen=True, slots=True)
class _WNA16PreparedCudaH2DTransaction:
    """Unpublished CUDA H2D ABI boundary with a CPU-validated control map.

    This records no CUDA work. A future controller-owned enqueue implementation
    must consume this exact reservation and never read device data on the host.
    """

    controller: object
    reservation: _WNA16SlotReservation
    candidate_cpu_slot_map: torch.Tensor
    staged_slot: _WNA16StagedCudaSlot
    token: object


@dataclass(frozen=True, slots=True)
class _WNA16CpuGenerationOperands:
    """Complete CPU operand schema for one unpublished generation."""

    w13_weight_packed: torch.Tensor
    w2_weight_packed: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor
    w13_weight_zero_point: torch.Tensor | None
    w2_weight_zero_point: torch.Tensor | None

    def __post_init__(self) -> None:
        """Snapshot every canonical operand at the transaction boundary."""
        for name in (
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_zero_point",
            "w2_weight_zero_point",
        ):
            value = object.__getattribute__(self, name)
            if value is not None:
                object.__setattr__(self, name, _copy_canonical_wna16_cpu_operand(value))

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name in {
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_zero_point",
            "w2_weight_zero_point",
        } and isinstance(value, torch.Tensor):
            return value.detach().clone()
        return value


@dataclass(frozen=True, slots=True)
class _WNA16CpuGenerationTransaction:
    """Controller-owned, pre-enqueue atomic CPU generation contract.

    This owns immutable candidate-map data, complete operands, exact logical
    reservations, and a monotonic identity. It is not publishable: no CUDA
    allocation, H2D, event, map authority, or generation view is issued here.
    """

    controller: object
    generation: int
    operands: _WNA16CpuGenerationOperands
    candidate_cpu_slot_map: torch.Tensor
    reservations: tuple[_WNA16SlotReservation, ...]
    source: _CanonicalWNA16CpuBank
    token: object

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_cpu_slot_map",
            self.candidate_cpu_slot_map.detach().clone().contiguous(),
        )

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name == "candidate_cpu_slot_map" and isinstance(value, torch.Tensor):
            return value.detach().clone()
        return value


@dataclass(frozen=True, slots=True)
class _WNA16CpuGenerationConstructionRollback:
    """Controller-private retry ownership after generation construction failed."""

    controller: object
    reservations: tuple[_WNA16SlotReservation, ...]
    source: _CanonicalWNA16CpuBank
    token: object


@dataclass(frozen=True, slots=True)
class _WNA16SlotPlan:
    status: _WNA16SlotPlanStatus
    slot: int | None
    reservation: _WNA16SlotReservation | None = None


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

    def __init__(self, layer_id: int, *, slot_count: int | None = None) -> None:
        if slot_count is not None and slot_count <= 0:
            raise ValueError("private WNA16 slot count must be positive")
        self._lifecycle_lock = RLock()
        self._layer_id = layer_id
        self._bound_routed_experts: RoutedExperts | None = None
        self._canonical_cpu_bank: _CanonicalWNA16CpuBank | None = None
        self._slot_count = slot_count
        self._slots: list[_WNA16LogicalSlot] = []
        self._slot_clock = 0
        self._reservations: dict[object, _WNA16SlotReservation] = {}
        self._prepared_cuda_h2d_transactions: dict[
            object, _WNA16PreparedCudaH2DTransaction
        ] = {}
        self._prepared_cpu_generation_transactions: dict[
            object, _WNA16CpuGenerationTransaction
        ] = {}
        self._cpu_generation_construction_rollbacks: dict[
            object, _WNA16CpuGenerationConstructionRollback
        ] = {}
        self._next_cpu_generation = 0
        self._retained_cpu_source_slots: dict[object, _WNA16CpuSourceSlotRetention] = {}
        self._cpu_source_slot_retention_queries: set[object] = set()
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
            self._slots = [
                _WNA16LogicalSlot()
                for _ in range(min(self._slot_count or 0, bank.global_num_experts))
            ]

    def _plan_slot_transaction(self, expert_id: int) -> _WNA16SlotPlan:
        """Reserve a logical future slot without allocating or copying to CUDA."""
        with self._lifecycle_lock:
            bank = self._canonical_cpu_bank
            if self._closed or bank is None or not self._slots:
                raise _lifecycle_error("private WNA16 slot planning is unavailable")
            if expert_id < 0 or expert_id >= bank.global_num_experts:
                raise _lifecycle_error("private WNA16 slot expert is invalid")
            for index, slot in enumerate(self._slots):
                if (
                    slot.state is _WNA16SlotState.RESIDENT
                    and slot.expert_id == expert_id
                ):
                    self._touch_slot(slot)
                    return _WNA16SlotPlan(_WNA16SlotPlanStatus.HIT, index)
            empty = next(
                (
                    index
                    for index, slot in enumerate(self._slots)
                    if slot.state is _WNA16SlotState.ABSENT
                ),
                None,
            )
            if empty is None:
                candidates = [
                    (slot.last_used, index)
                    for index, slot in enumerate(self._slots)
                    if slot.state is _WNA16SlotState.RESIDENT and slot.pins == 0
                ]
                if not candidates:
                    return _WNA16SlotPlan(_WNA16SlotPlanStatus.FALLBACK, None)
                _, empty = min(candidates)
                victim = self._slots[empty]
                previous_slot = self._snapshot_slot(victim)
                victim.state = _WNA16SlotState.STAGED_REPLACEMENT
            else:
                victim = self._slots[empty]
                previous_slot = self._snapshot_slot(victim)
                victim.state = _WNA16SlotState.RESERVED_EMPTY
            reservation = _WNA16SlotReservation(
                self, empty, expert_id, object(), previous_slot
            )
            self._reservations[reservation.token] = reservation
            return _WNA16SlotPlan(_WNA16SlotPlanStatus.RESERVED, empty, reservation)

    def _rollback_slot_transaction(self, reservation: _WNA16SlotReservation) -> None:
        """Rollback the exact reservation; CPU planning never commits a fill."""
        with self._lifecycle_lock:
            if (
                reservation.controller is not self
                or self._closed
                or self._reservations.get(reservation.token) is not reservation
                or any(
                    retention.reservation is reservation
                    for retention in self._retained_cpu_source_slots.values()
                )
            ):
                raise _lifecycle_error(
                    "private WNA16 slot reservation is stale", stale=True
                )
            if reservation.slot >= len(self._slots):
                raise _lifecycle_error(
                    "private WNA16 slot reservation is stale", stale=True
                )
            slot = self._slots[reservation.slot]
            expected_state = (
                _WNA16SlotState.RESERVED_EMPTY
                if reservation.previous_slot.state is _WNA16SlotState.ABSENT
                else _WNA16SlotState.STAGED_REPLACEMENT
            )
            if slot.state is not expected_state:
                raise _lifecycle_error(
                    "private WNA16 slot reservation is stale", stale=True
                )
            self._reservations.pop(reservation.token)
            late_pins = slot.pins - reservation.previous_slot.pins
            self._slots[reservation.slot] = _WNA16LogicalSlot(
                state=reservation.previous_slot.state,
                expert_id=reservation.previous_slot.expert_id,
                pins=reservation.previous_slot.pins + max(late_pins, 0),
                last_used=reservation.previous_slot.last_used,
            )

    def _retain_cpu_source_slot_until_completion(
        self, reservation: _WNA16SlotReservation, completion: object
    ) -> _WNA16CpuSourceSlotRetention:
        """Retain one exact CPU source and logical slot for a future handoff.

        This models only controller-private ownership. It is not a CUDA event,
        H2D operation, scheduler request completion, or device reuse fence.
        """
        with self._lifecycle_lock:
            source = self._canonical_cpu_bank
            if (
                self._closed
                or source is None
                or reservation.controller is not self
                or self._reservations.get(reservation.token) is not reservation
                or any(
                    retained.reservation is reservation
                    for retained in self._retained_cpu_source_slots.values()
                )
            ):
                raise _lifecycle_error(
                    "private WNA16 CPU source-slot retention is stale", stale=True
                )
            retention = _WNA16CpuSourceSlotRetention(
                self, source, reservation, completion, object()
            )
            self._retained_cpu_source_slots[retention.token] = retention
            return retention

    def _stage_controller_private_cuda_slot(
        self,
        *,
        expert_id: int,
        staged_slot: _WNA16StagedCudaSlot,
        fill_event_factory: Callable[[], object],
        payload_copy: Callable[[torch.Tensor, torch.Tensor, int], None] | None = None,
    ) -> None:
        """Validate and fence a complete private slot without publishing it.

        This controller-private seam has no production caller. In particular it
        does not allocate, copy, synchronize, change a map, mark a slot
        resident, or return a ``WNA16GenerationView``. A real H2D implementation
        must add those operations atomically with controller-owned CUDA storage.
        """
        reservation: _WNA16SlotReservation | None = None
        try:
            plan = self._plan_slot_transaction(expert_id)
            if plan.status is not _WNA16SlotPlanStatus.RESERVED:
                raise _lifecycle_error(
                    "private WNA16 CUDA slot staging requires an empty slot"
                )
            reservation = plan.reservation
            if reservation is None:
                raise _lifecycle_error("private WNA16 slot reservation is missing")
            self._validate_staged_cuda_slot(reservation, staged_slot)
            if payload_copy is not None:
                if not callable(payload_copy):
                    raise _lifecycle_error(
                        "private WNA16 CPU staging payload copy must be callable"
                    )
                bank = self._canonical_cpu_bank
                if bank is None:
                    raise _lifecycle_error(
                        "private WNA16 CPU staging is stale", stale=True
                    )
                for source, destination in (
                    (bank.w13_weight_packed, staged_slot.w13_weight_packed),
                    (bank.w2_weight_packed, staged_slot.w2_weight_packed),
                    (bank.w13_weight_scale, staged_slot.w13_weight_scale),
                    (bank.w2_weight_scale, staged_slot.w2_weight_scale),
                ):
                    payload_copy(
                        source[expert_id],
                        destination[reservation.slot],
                        reservation.slot,
                    )
            if not callable(fill_event_factory):
                raise _lifecycle_error(
                    "private WNA16 CUDA fill event factory must be callable"
                )
            event = fill_event_factory()
            record = getattr(event, "record", None)
            if not callable(record):
                raise _lifecycle_error(
                    "private WNA16 CUDA fill event has no callable record"
                )
            record()
            self._retain_cpu_source_slot_until_completion(reservation, event)
        except BaseException:
            if reservation is not None:
                with suppress(BaseException):
                    self._rollback_slot_transaction(reservation)
            raise

    def _prepare_controller_private_cuda_h2d_transaction(
        self, *, expert_id: int, staged_slot: _WNA16StagedCudaSlot
    ) -> _WNA16PreparedCudaH2DTransaction:
        """Validate an unpublished CUDA destination before any H2D enqueue.

        This deliberately stops before allocation, streams, copies, events,
        publication, and generation-view issuance. The downstream private ABI
        cannot yet safely publish a CUDA map without its controller authority.
        """
        reservation: _WNA16SlotReservation | None = None
        try:
            plan = self._plan_slot_transaction(expert_id)
            if plan.status is not _WNA16SlotPlanStatus.RESERVED:
                raise _lifecycle_error(
                    "private WNA16 CUDA H2D preparation requires an empty slot"
                )
            reservation = plan.reservation
            if reservation is None:
                raise _lifecycle_error("private WNA16 slot reservation is missing")
            candidate = self._build_cpu_candidate_slot_map(reservation)
            self._validate_cuda_h2d_destination(reservation, staged_slot)
            transaction = _WNA16PreparedCudaH2DTransaction(
                self, reservation, candidate, staged_slot, object()
            )
            with self._lifecycle_lock:
                if (
                    self._closed
                    or self._reservations.get(reservation.token) is not reservation
                    or transaction.token in self._prepared_cuda_h2d_transactions
                ):
                    raise _lifecycle_error(
                        "private WNA16 CUDA H2D transaction is stale", stale=True
                    )
                self._prepared_cuda_h2d_transactions[transaction.token] = transaction
            return transaction
        except BaseException:
            if reservation is not None:
                with suppress(BaseException):
                    self._rollback_slot_transaction(reservation)
            raise

    def _prepare_controller_private_cpu_generation_transaction(
        self, *, expert_ids: tuple[int, ...]
    ) -> _WNA16CpuGenerationTransaction:
        """Atomically reserve a complete CPU-only candidate generation.

        All experts are planned from one logical snapshot before a slot is
        changed. This records the full WNA16 schema and CPU candidate map, but
        deliberately does not allocate CUDA storage, enqueue H2D, record an
        event, publish a map, or issue a generation view.
        """
        with self._lifecycle_lock:
            self._retry_controller_private_cpu_generation_construction_rollbacks()
            bank = self._canonical_cpu_bank
            if self._closed or bank is None or not self._slots:
                raise _lifecycle_error("private WNA16 CPU generation is unavailable")
            if not expert_ids:
                raise _lifecycle_error("private WNA16 CPU generation has no experts")
            if any(type(expert_id) is not int for expert_id in expert_ids):
                raise _lifecycle_error("private WNA16 CPU generation expert is invalid")
            unique_expert_ids = tuple(dict.fromkeys(expert_ids))
            if any(
                expert_id < 0 or expert_id >= bank.global_num_experts
                for expert_id in unique_expert_ids
            ):
                raise _lifecycle_error("private WNA16 CPU generation expert is invalid")

            snapshots = [self._snapshot_slot(slot) for slot in self._slots]
            planned: list[tuple[int, int, _WNA16LogicalSlotSnapshot]] = []
            planned_slots: set[int] = set()
            for expert_id in unique_expert_ids:
                if any(
                    snapshot.state is _WNA16SlotState.RESIDENT
                    and snapshot.expert_id == expert_id
                    for snapshot in snapshots
                ):
                    continue
                empty = next(
                    (
                        index
                        for index, snapshot in enumerate(snapshots)
                        if snapshot.state is _WNA16SlotState.ABSENT
                        and index not in planned_slots
                    ),
                    None,
                )
                if empty is None:
                    candidates = [
                        (snapshot.last_used, index)
                        for index, snapshot in enumerate(snapshots)
                        if snapshot.state is _WNA16SlotState.RESIDENT
                        and snapshot.pins == 0
                        and index not in planned_slots
                    ]
                    if not candidates:
                        raise _lifecycle_error(
                            "private WNA16 CPU generation has insufficient capacity"
                        )
                    _, empty = min(candidates)
                previous = snapshots[empty]
                planned.append((expert_id, empty, previous))
                planned_slots.add(empty)
                snapshots[empty] = _WNA16LogicalSlotSnapshot(
                    _WNA16SlotState.RESERVED_EMPTY
                    if previous.state is _WNA16SlotState.ABSENT
                    else _WNA16SlotState.STAGED_REPLACEMENT,
                    previous.expert_id,
                    previous.pins,
                    previous.last_used,
                )

            candidate = torch.full(
                (bank.global_num_experts,), -1, dtype=torch.int32, device="cpu"
            )
            for slot_index, snapshot in enumerate(snapshots):
                if snapshot.state is _WNA16SlotState.RESIDENT:
                    if snapshot.expert_id is None:
                        raise _lifecycle_error(
                            "private WNA16 CPU generation is invalid"
                        )
                    candidate[snapshot.expert_id] = slot_index
            for expert_id, slot_index, _ in planned:
                candidate[expert_id] = slot_index
            values = candidate.tolist()
            if (
                not candidate.is_contiguous()
                or len({value for value in values if value >= 0})
                != sum(value >= 0 for value in values)
                or any(value < -1 or value >= len(self._slots) for value in values)
            ):
                raise _lifecycle_error("private WNA16 CPU generation map is invalid")

            reservations: list[_WNA16SlotReservation] = []
            try:
                for expert_id, slot_index, previous in planned:
                    slot = self._slots[slot_index]
                    if self._snapshot_slot(slot) != previous:
                        raise _lifecycle_error(
                            "private WNA16 CPU generation is stale", stale=True
                        )
                    slot.state = (
                        _WNA16SlotState.RESERVED_EMPTY
                        if previous.state is _WNA16SlotState.ABSENT
                        else _WNA16SlotState.STAGED_REPLACEMENT
                    )
                    reservation = _WNA16SlotReservation(
                        self, slot_index, expert_id, object(), previous
                    )
                    self._reservations[reservation.token] = reservation
                    reservations.append(reservation)
                generation = self._next_cpu_generation + 1
                transaction = _WNA16CpuGenerationTransaction(
                    controller=self,
                    generation=generation,
                    operands=_WNA16CpuGenerationOperands(
                        bank.w13_weight_packed,
                        bank.w2_weight_packed,
                        bank.w13_weight_scale,
                        bank.w2_weight_scale,
                        bank.w13_weight_zero_point,
                        bank.w2_weight_zero_point,
                    ),
                    candidate_cpu_slot_map=candidate,
                    reservations=tuple(reservations),
                    source=bank,
                    token=object(),
                )
                self._prepared_cpu_generation_transactions[transaction.token] = (
                    transaction
                )
                self._next_cpu_generation = generation
                return transaction
            except BaseException:
                for reservation in reversed(reservations):
                    with suppress(BaseException):
                        self._rollback_slot_transaction(reservation)
                if any(
                    self._reservations.get(reservation.token) is reservation
                    for reservation in reservations
                ):
                    rollback = _WNA16CpuGenerationConstructionRollback(
                        self, tuple(reservations), bank, object()
                    )
                    self._cpu_generation_construction_rollbacks[rollback.token] = (
                        rollback
                    )
                raise

    def _retry_controller_private_cpu_generation_construction_rollbacks(self) -> None:
        """Retry controller-owned cleanup left by failed transaction construction."""
        with self._lifecycle_lock:
            rollbacks = tuple(self._cpu_generation_construction_rollbacks.values())
        first_error: BaseException | None = None
        for rollback in rollbacks:
            with self._lifecycle_lock:
                if (
                    rollback.controller is not self
                    or rollback.source is not self._canonical_cpu_bank
                    or self._cpu_generation_construction_rollbacks.get(rollback.token)
                    is not rollback
                ):
                    error = _lifecycle_error(
                        "private WNA16 CPU generation construction rollback is stale",
                        stale=True,
                    )
                    if first_error is None:
                        first_error = error
                    continue
            for reservation in reversed(rollback.reservations):
                with self._lifecycle_lock:
                    owned = self._reservations.get(reservation.token)
                    if owned is None:
                        continue
                    if owned is not reservation:
                        error = _lifecycle_error(
                            "private WNA16 CPU generation is stale", stale=True
                        )
                        if first_error is None:
                            first_error = error
                        continue
                try:
                    self._rollback_slot_transaction(reservation)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            with self._lifecycle_lock:
                if not any(
                    reservation.token in self._reservations
                    for reservation in rollback.reservations
                ):
                    del self._cpu_generation_construction_rollbacks[rollback.token]
        if first_error is not None:
            raise first_error

    def _rollback_controller_private_cpu_generation_transaction(
        self, transaction: _WNA16CpuGenerationTransaction
    ) -> None:
        """Release one exact unpublished CPU generation transaction."""
        with self._lifecycle_lock:
            if (
                not isinstance(transaction, _WNA16CpuGenerationTransaction)
                or transaction.controller is not self
                or self._closed
                or self._prepared_cpu_generation_transactions.get(transaction.token)
                is not transaction
                or transaction.source is not self._canonical_cpu_bank
            ):
                raise _lifecycle_error(
                    "private WNA16 CPU generation is stale", stale=True
                )

        first_error: BaseException | None = None
        for reservation in reversed(transaction.reservations):
            with self._lifecycle_lock:
                owned = self._reservations.get(reservation.token)
                if owned is None:
                    continue
                if owned is not reservation:
                    error = _lifecycle_error(
                        "private WNA16 CPU generation is stale", stale=True
                    )
                    if first_error is None:
                        first_error = error
                    continue
            try:
                self._rollback_slot_transaction(reservation)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

        with self._lifecycle_lock:
            if self._prepared_cpu_generation_transactions.get(
                transaction.token
            ) is not transaction or any(
                reservation.token in self._reservations
                for reservation in transaction.reservations
            ):
                raise _lifecycle_error(
                    "private WNA16 CPU generation is stale", stale=True
                )
            del self._prepared_cpu_generation_transactions[transaction.token]

    def _rollback_prepared_controller_private_cuda_h2d_transaction(
        self, transaction: _WNA16PreparedCudaH2DTransaction
    ) -> None:
        """Release a pre-enqueue ABI reservation with no CUDA ownership."""
        with self._lifecycle_lock:
            if (
                not isinstance(transaction, _WNA16PreparedCudaH2DTransaction)
                or transaction.controller is not self
                or self._closed
                or self._prepared_cuda_h2d_transactions.get(transaction.token)
                is not transaction
                or self._reservations.get(transaction.reservation.token)
                is not transaction.reservation
            ):
                raise _lifecycle_error(
                    "private WNA16 CUDA H2D transaction is stale", stale=True
                )
            del self._prepared_cuda_h2d_transactions[transaction.token]
        self._rollback_slot_transaction(transaction.reservation)

    def _build_cpu_candidate_slot_map(
        self, reservation: _WNA16SlotReservation
    ) -> torch.Tensor:
        """Build and validate the only host-readable control-map candidate."""
        with self._lifecycle_lock:
            bank = self._canonical_cpu_bank
            if (
                self._closed
                or bank is None
                or self._reservations.get(reservation.token) is not reservation
            ):
                raise _lifecycle_error(
                    "private WNA16 CPU control map is stale", stale=True
                )
            candidate = torch.full(
                (bank.global_num_experts,), -1, dtype=torch.int32, device="cpu"
            )
            for slot_index, slot in enumerate(self._slots):
                if slot.state is _WNA16SlotState.RESIDENT:
                    if slot.expert_id is None:
                        raise _lifecycle_error(
                            "private WNA16 CPU control map is invalid"
                        )
                    candidate[slot.expert_id] = slot_index
            candidate[reservation.expert_id] = reservation.slot
            values = candidate.tolist()
            if (
                not candidate.is_contiguous()
                or candidate.dtype is not torch.int32
                or any(value < -1 or value >= len(self._slots) for value in values)
                or len({value for value in values if value >= 0})
                != sum(value >= 0 for value in values)
                or values[reservation.expert_id] != reservation.slot
            ):
                raise _lifecycle_error("private WNA16 CPU control map is invalid")
            return candidate

    def _validate_cuda_h2d_destination(
        self,
        reservation: _WNA16SlotReservation,
        staged_slot: _WNA16StagedCudaSlot,
    ) -> None:
        """Check CUDA metadata only; never read payload or map contents on host."""
        with self._lifecycle_lock:
            bank = self._canonical_cpu_bank
            if (
                self._closed
                or bank is None
                or self._reservations.get(reservation.token) is not reservation
            ):
                raise _lifecycle_error(
                    "private WNA16 CUDA H2D destination is stale", stale=True
                )
            destination_device: torch.device | None = None
            required = (
                (bank.w13_weight_packed, staged_slot.w13_weight_packed),
                (bank.w2_weight_packed, staged_slot.w2_weight_packed),
                (bank.w13_weight_scale, staged_slot.w13_weight_scale),
                (bank.w2_weight_scale, staged_slot.w2_weight_scale),
            )
            for source, destination in required:
                if (
                    not destination.is_cuda
                    or not destination.is_contiguous()
                    or destination.dtype is not source.dtype
                    or destination.ndim != source.ndim
                    or destination.shape[0] != len(self._slots)
                    or tuple(destination.shape[1:]) != tuple(source.shape[1:])
                ):
                    raise _lifecycle_error(
                        "private WNA16 CUDA H2D destination operands are incomplete"
                    )
                if destination_device is None:
                    destination_device = destination.device
                elif destination.device != destination_device:
                    raise _lifecycle_error(
                        "private WNA16 CUDA H2D destination operands disagree on device"
                    )
            for source, destination in (
                (bank.w13_weight_zero_point, staged_slot.w13_weight_zero_point),
                (bank.w2_weight_zero_point, staged_slot.w2_weight_zero_point),
            ):
                if (source is None) != (destination is None):
                    raise _lifecycle_error(
                        "private WNA16 CUDA destination zero points are incomplete"
                    )
                if (
                    source is not None
                    and destination is not None
                    and (
                        not destination.is_cuda
                        or not destination.is_contiguous()
                        or destination.dtype is not source.dtype
                        or destination.shape != (len(self._slots), *source.shape[1:])
                        or destination.device != destination_device
                    )
                ):
                    raise _lifecycle_error(
                        "private WNA16 CUDA destination zero points are invalid"
                    )
            slot_map = staged_slot.slot_map
            if (
                not slot_map.is_cuda
                or not slot_map.is_contiguous()
                or slot_map.dtype is not torch.int32
                or slot_map.ndim != 1
                or slot_map.shape[0] != bank.global_num_experts
                or slot_map.device != destination_device
            ):
                raise _lifecycle_error(
                    "private WNA16 CUDA H2D destination map is invalid"
                )

    def _validate_staged_cuda_slot(
        self,
        reservation: _WNA16SlotReservation,
        staged_slot: _WNA16StagedCudaSlot,
    ) -> None:
        """Validate all destination operands/map before a fill event is recorded."""
        with self._lifecycle_lock:
            bank = self._canonical_cpu_bank
            if (
                self._closed
                or bank is None
                or self._reservations.get(reservation.token) is not reservation
                or reservation.slot >= len(self._slots)
            ):
                raise _lifecycle_error(
                    "private WNA16 CUDA slot staging is stale", stale=True
                )
            expected = (
                (bank.w13_weight_packed, staged_slot.w13_weight_packed),
                (bank.w2_weight_packed, staged_slot.w2_weight_packed),
                (bank.w13_weight_scale, staged_slot.w13_weight_scale),
                (bank.w2_weight_scale, staged_slot.w2_weight_scale),
            )
            optional_expected = (
                (bank.w13_weight_zero_point, staged_slot.w13_weight_zero_point),
                (bank.w2_weight_zero_point, staged_slot.w2_weight_zero_point),
            )
            destination_device: torch.device | None = None
            for source, destination in expected:
                if (
                    not isinstance(destination, torch.Tensor)
                    or destination.dtype is not source.dtype
                    or destination.ndim != source.ndim
                    or destination.shape[0] != len(self._slots)
                    or tuple(destination.shape[1:]) != tuple(source.shape[1:])
                ):
                    raise _lifecycle_error(
                        "private WNA16 CUDA destination operands are incomplete"
                    )
                if destination_device is None:
                    destination_device = destination.device
                elif destination.device != destination_device:
                    raise _lifecycle_error(
                        "private WNA16 CUDA destination operands disagree on device"
                    )
            for source, destination in optional_expected:
                if (source is None) != (destination is None):
                    raise _lifecycle_error(
                        "private WNA16 CUDA destination zero points are incomplete"
                    )
                if (
                    source is not None
                    and destination is not None
                    and (
                        destination.dtype is not source.dtype
                        or destination.ndim != source.ndim
                        or destination.shape[0] != len(self._slots)
                        or tuple(destination.shape[1:]) != tuple(source.shape[1:])
                        or destination.device != destination_device
                    )
                ):
                    raise _lifecycle_error(
                        "private WNA16 CUDA destination zero points are invalid"
                    )
            slot_map = staged_slot.slot_map
            if (
                not isinstance(slot_map, torch.Tensor)
                or slot_map.dtype is not torch.int32
                or slot_map.ndim != 1
                or slot_map.shape[0] != bank.global_num_experts
            ):
                raise _lifecycle_error("private WNA16 CUDA slot map is invalid")
            if slot_map.device.type != "cpu":
                raise _lifecycle_error(
                    "private WNA16 CUDA slot map must be CPU-resident"
                )
            if slot_map.device != destination_device:
                raise _lifecycle_error("private WNA16 CUDA slot map is invalid")
            values = slot_map.tolist()
            if (
                values[reservation.expert_id] != reservation.slot
                or any(value < -1 or value >= len(self._slots) for value in values)
                or len({value for value in values if value >= 0})
                != sum(value >= 0 for value in values)
            ):
                raise _lifecycle_error("private WNA16 CUDA slot map is incomplete")

    def _finalize_completed_cpu_source_slot_retentions(self) -> None:
        """Poll CPU-only completion predicates without holding lifecycle ownership."""
        with self._lifecycle_lock:
            retentions = tuple(self._retained_cpu_source_slots.values())
        first_error: Phase4UnsupportedError | None = None
        for retention in retentions:
            try:
                self._finalize_completed_cpu_source_slot_retention(retention)
            except Phase4UnsupportedError as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _finalize_completed_cpu_source_slot_retention(
        self, retention: _WNA16CpuSourceSlotRetention
    ) -> None:
        with self._lifecycle_lock:
            if (
                retention.token in self._cpu_source_slot_retention_queries
                or not self._is_exact_cpu_source_slot_retention(retention)
            ):
                return
            completion = retention.completion
            self._cpu_source_slot_retention_queries.add(retention.token)
        completed = False
        try:
            try:
                query = getattr(completion, "query", None)
                if not callable(query):
                    raise TypeError("completion predicate has no callable query")
                result = query()
            except Exception as error:
                raise _event_query_error("completion predicate query failed") from error
            if type(result) is not bool:
                raise _event_query_error(
                    "completion predicate query did not return bool"
                )
            completed = result
        finally:
            with self._lifecycle_lock:
                self._cpu_source_slot_retention_queries.discard(retention.token)
                if completed and self._is_exact_cpu_source_slot_retention(retention):
                    reservation = retention.reservation
                    slot = self._slots[reservation.slot]
                    self._reservations.pop(reservation.token)
                    self._retained_cpu_source_slots.pop(retention.token)
                    late_pins = slot.pins - reservation.previous_slot.pins
                    self._slots[reservation.slot] = _WNA16LogicalSlot(
                        state=reservation.previous_slot.state,
                        expert_id=reservation.previous_slot.expert_id,
                        pins=reservation.previous_slot.pins + max(late_pins, 0),
                        last_used=reservation.previous_slot.last_used,
                    )

    def _is_exact_cpu_source_slot_retention(
        self, retention: _WNA16CpuSourceSlotRetention
    ) -> bool:
        reservation = retention.reservation
        if (
            retention.controller is not self
            or self._retained_cpu_source_slots.get(retention.token) is not retention
            or retention.source is not self._canonical_cpu_bank
            or reservation.controller is not self
            or self._reservations.get(reservation.token) is not reservation
            or reservation.slot >= len(self._slots)
        ):
            return False
        expected_state = (
            _WNA16SlotState.RESERVED_EMPTY
            if reservation.previous_slot.state is _WNA16SlotState.ABSENT
            else _WNA16SlotState.STAGED_REPLACEMENT
        )
        return self._slots[reservation.slot].state is expected_state

    def _seed_logical_resident_for_test(
        self, slot_index: int, expert_id: int, *, last_used: int = 0
    ) -> None:
        """Install CPU-only logical test state without a transfer or mapping change."""
        with self._lifecycle_lock:
            bank = self._canonical_cpu_bank
            if self._closed or bank is None or not self._slots:
                raise _lifecycle_error("private WNA16 slot planning is unavailable")
            if not 0 <= slot_index < len(self._slots):
                raise _lifecycle_error("private WNA16 slot index is invalid")
            if not 0 <= expert_id < bank.global_num_experts:
                raise _lifecycle_error("private WNA16 slot expert is invalid")
            if self._slots[slot_index].state is not _WNA16SlotState.ABSENT:
                raise _lifecycle_error("private WNA16 logical slot is occupied")
            self._slots[slot_index] = _WNA16LogicalSlot(
                state=_WNA16SlotState.RESIDENT,
                expert_id=expert_id,
                last_used=last_used,
            )
            self._slot_clock = max(self._slot_clock, last_used)

    def _acquire_logical_resident_pin_for_test(self, expert_id: int) -> None:
        """Acquire a CPU-only logical pin, including a staged victim."""
        with self._lifecycle_lock:
            if self._closed:
                raise _lifecycle_error("private WNA16 slot planning is unavailable")
            for slot in self._slots:
                if slot.expert_id == expert_id and slot.state in (
                    _WNA16SlotState.RESIDENT,
                    _WNA16SlotState.STAGED_REPLACEMENT,
                ):
                    slot.pins += 1
                    return
            raise _lifecycle_error("private WNA16 logical resident is unavailable")

    @staticmethod
    def _snapshot_slot(slot: _WNA16LogicalSlot) -> _WNA16LogicalSlotSnapshot:
        return _WNA16LogicalSlotSnapshot(
            state=slot.state,
            expert_id=slot.expert_id,
            pins=slot.pins,
            last_used=slot.last_used,
        )

    def _touch_slot(self, slot: _WNA16LogicalSlot) -> None:
        self._slot_clock += 1
        slot.last_used = self._slot_clock

    def close(self) -> None:
        with self._lifecycle_lock:
            self._closed = True
        self._finalize_completed_cpu_source_slot_retentions()
        with self._lifecycle_lock:
            if self._retained_cpu_source_slots:
                raise _drain_incomplete_error()
            self._canonical_cpu_bank = None
            self._bound_routed_experts = None
            self._slots.clear()
            self._reservations.clear()
            self._prepared_cuda_h2d_transactions.clear()
            self._prepared_cpu_generation_transactions.clear()
            self._cpu_generation_construction_rollbacks.clear()
            self._cpu_source_slot_retention_queries.clear()

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
