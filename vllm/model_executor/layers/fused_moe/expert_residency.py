# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only expert-residency state machine with synchronous lifecycle APIs.

Reservations are authorized by object identity, rather than by equal lease
values. Stale and double finalization attempts are rejected. A staged victim
remains published until successful replacement; a pin observed at completion
causes rollback without eviction. ``EVICTING`` is an internal, synchronous
transition during that replacement and is never externally stable. Failed
slots are reusable. Callbacks may re-enter the table, so their return boundary
revalidates the exact current lease before outer lifecycle mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Mapping

import torch


@dataclass(frozen=True, slots=True)
class ExpertKey:
    """Layer-qualified identity for one logical expert."""

    layer_id: int
    logical_expert_id: int

    def __post_init__(self) -> None:
        if self.layer_id < 0 or self.logical_expert_id < 0:
            raise ValueError("expert identifiers must be non-negative")


@dataclass(frozen=True, slots=True)
class ExpertBundle:
    """Opaque validated data for a resident expert generation."""

    generation: int
    payload: object


class SlotState(Enum):
    ABSENT = auto()
    LOADING = auto()
    RESIDENT = auto()
    EVICTING = auto()
    FAILED = auto()


class LoadStatus(Enum):
    HIT = auto()
    RESERVED = auto()
    LOADED = auto()
    FALLBACK = auto()


class EventKind(Enum):
    REQUEST = auto()
    HIT = auto()
    MISS = auto()
    RESERVATION = auto()
    LOAD_SUCCESS = auto()
    LOAD_FAILURE = auto()
    EVICTION = auto()
    STAGED_ROLLBACK = auto()
    DUPLICATE_LOAD_REJECTION = auto()
    CAPACITY_PIN_BLOCKED_FALLBACK = auto()


@dataclass(frozen=True, slots=True)
class ResidencyEvent:
    """Immutable observable record of one residency decision."""

    kind: EventKind
    key: ExpertKey
    slot: int | None = None
    staged_replacement: bool = False


@dataclass(frozen=True, slots=True)
class ReservationLease:
    """Opaque immutable authorization to finalize one reserved load.

    A lease is valid only when this exact object is the current reservation;
    equal copies, stale leases, and double finalization are rejected.
    """

    key: ExpertKey
    slot: int
    token: int
    staged_replacement: bool = False


@dataclass(frozen=True, slots=True)
class Resolution:
    """Outcome of resolving or reserving one logical expert."""

    key: ExpertKey
    status: LoadStatus
    slot: int | None
    bundle: ExpertBundle | None
    lease: ReservationLease | None = None


@dataclass(slots=True)
class ResidencyAccounting:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    loads: int = 0
    load_successes: int = 0
    load_failures: int = 0
    evictions: int = 0


@dataclass(slots=True)
class _Slot:
    state: SlotState = SlotState.ABSENT
    key: ExpertKey | None = None
    bundle: ExpertBundle | None = None
    generation: int | None = None
    pins: int = 0
    last_used: int = 0


class ExpertResidencyTable:
    """Synchronous, non-thread-safe CPU residency reference.

    This is not a live asynchronous primitive without external synchronization.
    Load and validation callbacks may re-enter the table. On return, the outer
    operation verifies that its exact lease remains current before it mutates
    mappings, slots, accounting, or events.
    """

    def __init__(
        self,
        capacity: int,
        validate_bundle: Callable[[ExpertKey, ExpertBundle], bool],
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._slots = [_Slot() for _ in range(capacity)]
        self._published: dict[ExpertKey, int] = {}
        self._loading: dict[ExpertKey, ReservationLease] = {}
        self._events: list[ResidencyEvent] = []
        self._validate_bundle = validate_bundle
        self._clock = 0
        self._next_reservation_token = 0
        self.accounting = ResidencyAccounting()
        self.assert_invariants()

    def resolve(
        self,
        key: ExpertKey,
        load: Callable[[ExpertKey], ExpertBundle],
    ) -> Resolution:
        """Resolve a resident key or synchronously load it via ``load``."""
        slot = self._published.get(key)
        self.accounting.requests += 1
        self._record(EventKind.REQUEST, key, slot)
        if slot is not None:
            self.accounting.hits += 1
            self._record(EventKind.HIT, key, slot)
            self._touch(slot)
            value = self._slots[slot].bundle
            assert value is not None
            self.assert_invariants()
            return Resolution(key, LoadStatus.HIT, slot, value)

        self.accounting.misses += 1
        self._record(EventKind.MISS, key)
        try:
            lease = self._begin_load(key)
        except RuntimeError:
            self.assert_invariants()
            raise
        if lease is None:
            self._record(EventKind.CAPACITY_PIN_BLOCKED_FALLBACK, key)
            self.assert_invariants()
            return Resolution(key, LoadStatus.FALLBACK, None, None)
        try:
            value = load(key)
        except Exception:
            self._require_current_lease(lease)
            self.fail_load(lease)
            return Resolution(key, LoadStatus.FALLBACK, lease.slot, None)
        self._require_current_lease(lease)
        return self.complete_load(lease, value)

    def begin_load(self, key: ExpertKey) -> Resolution:
        """Report a hit, reserve a load slot, or report capacity fallback."""
        self.accounting.requests += 1
        slot = self._published.get(key)
        self._record(EventKind.REQUEST, key, slot)
        if slot is not None:
            self.accounting.hits += 1
            self._record(EventKind.HIT, key, slot)
            self._touch(slot)
            bundle = self._slots[slot].bundle
            assert bundle is not None
            self.assert_invariants()
            return Resolution(key, LoadStatus.HIT, slot, bundle)

        self.accounting.misses += 1
        self._record(EventKind.MISS, key)
        lease = self._begin_load(key)
        self.assert_invariants()
        if lease is None:
            self._record(EventKind.CAPACITY_PIN_BLOCKED_FALLBACK, key)
            return Resolution(key, LoadStatus.FALLBACK, None, None)
        return Resolution(key, LoadStatus.RESERVED, lease.slot, None, lease)

    def complete_load(
        self, lease: ReservationLease, bundle: ExpertBundle
    ) -> Resolution:
        """Validate and atomically publish the exact current reservation."""
        reservation = self._require_current_lease(lease)
        key = lease.key
        slot = lease.slot
        valid = isinstance(bundle, ExpertBundle)
        if valid:
            try:
                valid = self._validate_bundle(key, bundle)
            except Exception:
                valid = False
        reservation = self._require_current_lease(lease)
        if not valid:
            self.fail_load(lease)
            return Resolution(key, LoadStatus.FALLBACK, slot, None)

        entry = self._slots[slot]
        if reservation.staged_replacement:
            assert entry.state is SlotState.RESIDENT
            assert entry.key is not None
            if entry.pins != 0:
                self.fail_load(lease)
                return Resolution(key, LoadStatus.FALLBACK, slot, None)
            entry.state = SlotState.EVICTING
            self._published.pop(entry.key)
            self.accounting.evictions += 1
            self._record(EventKind.EVICTION, entry.key, slot, True)
        entry.key = key
        entry.bundle = bundle
        entry.generation = bundle.generation
        entry.pins = 0
        entry.state = SlotState.RESIDENT
        self._loading.pop(key)
        self._published[key] = slot
        self.accounting.loads += 1
        self.accounting.load_successes += 1
        self._record(EventKind.LOAD_SUCCESS, key, slot, reservation.staged_replacement)
        self._touch(slot)
        self.assert_invariants()
        return Resolution(key, LoadStatus.LOADED, slot, bundle)

    def fail_load(self, lease: ReservationLease) -> None:
        """Fail the exact current reservation without publishing a mapping."""
        reservation = self._require_current_lease(lease)
        key = lease.key
        slot = lease.slot
        self._loading.pop(key)
        self._record(EventKind.LOAD_FAILURE, key, slot, reservation.staged_replacement)
        if reservation.staged_replacement:
            self._record(EventKind.STAGED_ROLLBACK, key, slot, True)
            self.accounting.loads += 1
            self.accounting.load_failures += 1
            self.assert_invariants()
            return
        entry = self._slots[slot]
        entry.state = SlotState.FAILED
        entry.bundle = None
        entry.generation = None
        self.accounting.loads += 1
        self.accounting.load_failures += 1
        self.assert_invariants()

    def pin(self, key: ExpertKey) -> None:
        slot = self._require_published_slot(key)
        self._slots[slot].pins += 1
        self.assert_invariants()

    def unpin(self, key: ExpertKey) -> None:
        slot = self._require_published_slot(key)
        entry = self._slots[slot]
        if entry.pins == 0:
            raise ValueError("cannot unpin an unpinned expert")
        entry.pins -= 1
        self.assert_invariants()

    def pin_count(self, key: ExpertKey) -> int:
        return self._slots[self._require_published_slot(key)].pins

    def slot_for(self, key: ExpertKey) -> int | None:
        return self._published.get(key)

    def slot_state(self, slot: int | None) -> SlotState:
        if slot is None or not 0 <= slot < len(self._slots):
            raise IndexError("slot is outside table capacity")
        return self._slots[slot].state

    def published_slots(self) -> dict[ExpertKey, int]:
        return dict(self._published)

    def events(self) -> tuple[ResidencyEvent, ...]:
        """Return an immutable snapshot of append-only reference decisions.

        This CPU reference ledger is intentionally unbounded and each call
        allocates a tuple snapshot; it is not a production telemetry interface.
        """
        return tuple(self._events)

    def _begin_load(self, key: ExpertKey) -> ReservationLease | None:
        if key in self._loading:
            reservation = self._loading[key]
            self._record(
                EventKind.DUPLICATE_LOAD_REJECTION,
                key,
                reservation.slot,
                reservation.staged_replacement,
            )
            raise RuntimeError(f"expert {key} is already loading")
        failed_slot = next(
            (
                index
                for index, entry in enumerate(self._slots)
                if entry.state is SlotState.FAILED and entry.key == key
            ),
            None,
        )
        slot = failed_slot if failed_slot is not None else self._available_slot()
        staged_replacement = False
        if slot is None:
            slot = self._replacement_victim()
            if slot is None:
                return None
            staged_replacement = True
        entry = self._slots[slot]
        if not staged_replacement:
            entry.state = SlotState.LOADING
            entry.key = key
            entry.bundle = None
            entry.generation = None
            entry.pins = 0
        lease = ReservationLease(
            key=key,
            slot=slot,
            token=self._next_reservation_token,
            staged_replacement=staged_replacement,
        )
        self._next_reservation_token += 1
        self._loading[key] = lease
        self._record(EventKind.RESERVATION, key, slot, staged_replacement)
        return lease

    def _available_slot(self) -> int | None:
        for index, entry in enumerate(self._slots):
            if entry.state in (SlotState.ABSENT, SlotState.FAILED):
                return index
        return None

    def _replacement_victim(self) -> int | None:
        reserved_victims = {
            reservation.slot
            for reservation in self._loading.values()
            if reservation.staged_replacement
        }
        evictable = [
            (entry.last_used, index)
            for index, entry in enumerate(self._slots)
            if (
                entry.state is SlotState.RESIDENT
                and entry.pins == 0
                and index not in reserved_victims
            )
        ]
        if not evictable:
            return None
        _, slot = min(evictable)
        return slot

    def _record(
        self,
        kind: EventKind,
        key: ExpertKey,
        slot: int | None = None,
        staged_replacement: bool = False,
    ) -> None:
        self._events.append(ResidencyEvent(kind, key, slot, staged_replacement))

    def _touch(self, slot: int) -> None:
        self._clock += 1
        self._slots[slot].last_used = self._clock

    def _require_current_lease(self, lease: ReservationLease) -> ReservationLease:
        if not isinstance(lease, ReservationLease):
            raise RuntimeError("load finalization requires a current reservation")
        current = self._loading.get(lease.key)
        if current is not lease:
            raise RuntimeError("lease is not the current reservation")
        return current

    def _require_published_slot(self, key: ExpertKey) -> int:
        try:
            return self._published[key]
        except KeyError as error:
            raise KeyError(f"expert {key} is not resident") from error

    def assert_invariants(self) -> None:
        """Assert mapping, state, capacity, and accounting consistency."""
        accounting = self.accounting
        assert all(
            value >= 0
            for value in (
                accounting.requests,
                accounting.hits,
                accounting.misses,
                accounting.loads,
                accounting.load_successes,
                accounting.load_failures,
                accounting.evictions,
            )
        )
        assert accounting.requests == accounting.hits + accounting.misses
        assert accounting.loads == (
            accounting.load_successes + accounting.load_failures
        )
        assert accounting.loads <= accounting.misses
        assert self._clock >= 0
        assert len(self._published) <= len(self._slots)
        assert len(set(self._published.values())) == len(self._published)
        assert not self._published.keys() & self._loading.keys()
        for key, slot in self._published.items():
            assert 0 <= slot < len(self._slots)
            entry = self._slots[slot]
            assert entry.state is SlotState.RESIDENT
            assert entry.key == key
            assert entry.bundle is not None
            assert entry.generation == entry.bundle.generation
        for key, reservation in self._loading.items():
            slot = reservation.slot
            assert 0 <= slot < len(self._slots)
            entry = self._slots[slot]
            assert key not in self._published
            if reservation.staged_replacement:
                assert entry.state is SlotState.RESIDENT
                assert entry.key is not None
                assert entry.key != key
                assert self._published.get(entry.key) == slot
                continue
            assert entry.state is SlotState.LOADING
            assert entry.key == key
        for index, entry in enumerate(self._slots):
            assert entry.last_used >= 0
            assert entry.pins >= 0
            if entry.state is SlotState.ABSENT:
                assert entry.key is None
                assert entry.bundle is None
                assert entry.generation is None
                assert entry.pins == 0
            elif entry.state is SlotState.RESIDENT:
                assert entry.key is not None
                assert self._published.get(entry.key) == index
                assert entry.bundle is not None
                assert entry.generation == entry.bundle.generation
            elif entry.state is SlotState.LOADING:
                assert entry.key is not None
                reservation = self._loading.get(entry.key)
                assert reservation is not None
                assert reservation.slot == index
                assert not reservation.staged_replacement
                assert entry.key not in self._published
                assert entry.bundle is None
                assert entry.generation is None
                assert entry.pins == 0
            elif entry.state is SlotState.FAILED:
                assert entry.key is not None
                assert entry.key not in self._published
                assert entry.key not in self._loading
                assert entry.bundle is None
                assert entry.generation is None
                assert entry.pins == 0
            else:
                assert False, f"unexpected quiescent slot state: {entry.state}"


class Phase4FailureCategory(Enum):
    """Categories reserved by the fail-closed residency protocol."""

    CAPACITY = "capacity"
    TRANSFER = "transfer"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    STALE_LEASE = "stale lease"
    UNSUPPORTED_DYNAMIC_MAP = "unsupported dynamic map"


class Phase4UnsupportedError(RuntimeError):
    """Raised when the bounded GPU residency seam cannot be used safely."""

    def __init__(self, category: Phase4FailureCategory, diagnostic: str) -> None:
        self.category = category
        self.diagnostic = diagnostic
        super().__init__(f"Phase 4 {category.value}: {diagnostic}")


@dataclass(frozen=True, slots=True)
class Phase4Capability:
    """Explicit result of checking the narrow Phase-4 support matrix."""

    supported: bool
    category: Phase4FailureCategory
    reason: str


@dataclass(frozen=True, slots=True)
class PostConversionExpertBundle:
    """Complete post-conversion expert data suitable for a transfer reference.

    ``tensors`` contains the three named WNA16 projections consumed by the
    quantized apply path. This type deliberately has no checkpoint or
    tensor-slice loading API.
    """

    schema_version: int
    tensors: Mapping[str, torch.Tensor]
    metadata: Mapping[str, object]


class TorchCpuTransferReference:
    """CPU-only transfer oracle for complete post-conversion bundles."""

    REQUIRED_TENSORS = frozenset({"gate_proj", "up_proj", "down_proj"})
    REQUIRED_METADATA = frozenset({"weight_shapes", "quantization"})
    ALLOWED_METADATA = REQUIRED_METADATA

    @classmethod
    def transfer(
        cls, bundle: PostConversionExpertBundle
    ) -> PostConversionExpertBundle:
        """Clone a validated complete bundle onto CPU, or reject it."""
        if not isinstance(bundle, PostConversionExpertBundle):
            raise ValueError("expected a post-conversion expert bundle")
        if bundle.schema_version != 1:
            raise ValueError("unsupported post-conversion bundle schema")
        if not isinstance(bundle.tensors, Mapping):
            raise ValueError("bundle tensors must be a named tensor mapping")
        if not isinstance(bundle.metadata, Mapping):
            raise ValueError("bundle metadata must be a mapping")
        tensor_names = frozenset(bundle.tensors)
        if tensor_names != cls.REQUIRED_TENSORS:
            raise ValueError("bundle must contain exactly all WNA16 tensors")
        if any(not isinstance(t, torch.Tensor) for t in bundle.tensors.values()):
            raise ValueError("bundle tensor values must be tensors")
        if any(t.device.type != "cpu" for t in bundle.tensors.values()):
            raise ValueError("CPU transfer oracle accepts CPU tensors only")
        if any(not t.is_contiguous() for t in bundle.tensors.values()):
            raise ValueError("bundle tensors must be contiguous")
        if frozenset(bundle.metadata) != cls.ALLOWED_METADATA:
            raise ValueError("bundle metadata is incomplete")
        shapes = bundle.metadata["weight_shapes"]
        if not isinstance(shapes, Mapping) or frozenset(shapes) != tensor_names:
            raise ValueError("bundle weight shapes do not match tensor identities")
        if any(
            not isinstance(shapes[name], (tuple, list, torch.Size))
            or any(not isinstance(dim, int) or dim < 0 for dim in shapes[name])
            for name in tensor_names
        ):
            raise ValueError("bundle weight shapes are invalid")
        if bundle.metadata["quantization"] != "WNA16":
            raise ValueError("bundle quantization must be WNA16")
        if any(
            tuple(t.shape) != tuple(shapes[name])
            for name, t in bundle.tensors.items()
        ):
            raise ValueError("bundle weight shapes are inconsistent")
        copied = {name: t.detach().clone() for name, t in bundle.tensors.items()}
        return PostConversionExpertBundle(
            schema_version=bundle.schema_version,
            tensors=copied,
            metadata=dict(bundle.metadata),
        )


class Phase4GpuResidencyAdapter:
    """Default-off, fail-closed adapter for the unproven GPU backend seam.

    The adapter intentionally does not copy tensors or mutate routing. Until a
    backend demonstrates per-call dynamic map support, ``validate_request``
    rejects every enabled request with an actionable diagnostic.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        model_family: str = "",
        quantization: str = "",
        execution_mode: str = "eager",
        modular: bool = True,
        backend_supports_dynamic_map: bool = False,
    ) -> None:
        self.enabled = enabled
        self.model_family = model_family
        self.quantization = quantization
        self.execution_mode = execution_mode
        self.modular = modular
        self.backend_supports_dynamic_map = backend_supports_dynamic_map
        self.unsupported_requests = 0

    def capability(self) -> Phase4Capability:
        if not self.enabled:
            return Phase4Capability(
                False,
                Phase4FailureCategory.UNSUPPORTED_DYNAMIC_MAP,
                "disabled by default",
            )
        if self.model_family != "Qwen3-30B-A3B":
            return Phase4Capability(
                False,
                Phase4FailureCategory.VALIDATION,
                "only Qwen3-30B-A3B is supported",
            )
        if self.quantization not in ("AWQ", "WNA16"):
            return Phase4Capability(
                False,
                Phase4FailureCategory.VALIDATION,
                "only AWQ/WNA16 is supported",
            )
        if self.execution_mode != "eager":
            return Phase4Capability(
                False,
                Phase4FailureCategory.VALIDATION,
                "only eager execution is supported",
            )
        if not self.modular:
            return Phase4Capability(
                False,
                Phase4FailureCategory.VALIDATION,
                "only modular execution is supported",
            )
        if not self.backend_supports_dynamic_map:
            return Phase4Capability(
                False,
                Phase4FailureCategory.UNSUPPORTED_DYNAMIC_MAP,
                "backend lacks per-call dynamic map support",
            )
        return Phase4Capability(
            False,
            Phase4FailureCategory.UNSUPPORTED_DYNAMIC_MAP,
            "GPU transfer controller is not implemented",
        )

    def validate_request(
        self, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> None:
        """Reject before apply; canonical router outputs are never rewritten."""
        del topk_ids, topk_weights
        self.unsupported_requests += 1
        capability = self.capability()
        raise Phase4UnsupportedError(capability.category, capability.reason)
