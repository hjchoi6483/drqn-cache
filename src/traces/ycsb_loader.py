from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class YCSBTraceEvent:
    index: int
    timestep: int
    operation: str
    key: int
    is_cache_lookup: bool
    count_read_hit_rate: bool
    workload: str | None = None


def _parse_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field} must be boolean-compatible, got {value!r}")


def event_from_mapping(raw: dict) -> YCSBTraceEvent:
    index = int(raw.get("index", raw.get("timestep")))
    return YCSBTraceEvent(
        index=index,
        timestep=int(raw.get("timestep", index)),
        operation=str(raw["operation"]).upper(),
        key=int(raw["key"]),
        is_cache_lookup=_parse_bool(raw["is_cache_lookup"], "is_cache_lookup"),
        count_read_hit_rate=_parse_bool(raw["count_read_hit_rate"], "count_read_hit_rate"),
        workload=str(raw["workload"]).upper() if raw.get("workload") is not None else None,
    )


def load_ycsb_events(path: str | Path) -> List[YCSBTraceEvent]:
    events: List[YCSBTraceEvent] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"YCSB JSONL line {line_no} must be an object")
            events.append(event_from_mapping(raw))
    return events


def lookup_stream_from_events(events: Iterable[YCSBTraceEvent], *, read_hit_rate_only: bool = True) -> List[int]:
    """Convert YCSB events to the integer request stream consumed by the simulator.

    By default this uses only operations counted in read-hit-rate metrics: READ
    and READ_MODIFY_WRITE. UPDATE and INSERT-only events are excluded, which also
    ensures Belady is computed over the cache-lookup/read subsequence.
    """
    if read_hit_rate_only:
        return [event.key for event in events if event.count_read_hit_rate]
    return [event.key for event in events if event.is_cache_lookup]


def load_ycsb_lookup_stream(path: str | Path, *, read_hit_rate_only: bool = True) -> List[int]:
    return lookup_stream_from_events(load_ycsb_events(path), read_hit_rate_only=read_hit_rate_only)
