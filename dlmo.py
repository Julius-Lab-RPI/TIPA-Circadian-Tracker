"""Recoverable CBTmin/DLMO candidate detection and tracking from MATLAB helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_DEFAULTS: dict[str, float] = {
    "warmup_hours": 48,
    "smooth_hours": 1,
    "confirm_hours": 2,
    "shape_hours": 6,
    "local_scale_hours": 24,
    "refine_hours": 0.5,
    "min_prominence_fraction": 0.10,
    "min_shoulder_fraction": 0.003,
    "min_width_hours": 2,
    "max_gap_hours": 0.25,
    "offset_hours": 7,
    "expected_period_hours": 24,
    "preferred_min_hours": 18,
    "preferred_max_hours": 30,
    "min_separation_hours": 12,
    "timing_scale_hours": 6,
    "restart_hours": 36,
    "event_reward": 3,
    "restart_penalty": 2,
    "ambiguity_margin": 0.75,
    "ambiguity_separation_hours": 2,
    "relax_final_trough": 1,
    "endpoint_confirm_hours": 0.5,
}


def compare_dlmo_sequence(
    hours: NDArray[np.float64],
    quality: NDArray[np.float64],
    segment: NDArray[np.int64],
    options: dict[str, float],
) -> tuple[NDArray[np.int64], NDArray[np.float64], float]:
    """Choose the highest-scoring retrospective path through trough candidates."""
    h = np.asarray(hours, dtype=np.float64).reshape(-1)
    q = np.asarray(quality, dtype=np.float64).reshape(-1)
    seg = np.asarray(segment, dtype=np.int64).reshape(-1)
    if not (h.size == q.size == seg.size) or np.any(np.diff(h) <= 0):
        raise ValueError("hours, quality, and segment must match and hours must increase")

    def solve(excluded: set[int]) -> tuple[NDArray[np.int64], float]:
        count = len(h)
        value = np.full(count, -np.inf)
        parent = np.full(count, -1, dtype=np.int64)
        for b in range(count):
            if b in excluded:
                continue
            reward = options["event_reward"] + q[b]
            value[b] = reward
            for a in range(b):
                gap = h[b] - h[a]
                if not np.isfinite(value[a]) or gap < options["min_separation_hours"]:
                    continue
                if seg[a] != seg[b] or gap >= options["restart_hours"]:
                    cost = options["restart_penalty"]
                else:
                    cost = ((gap - options["expected_period_hours"]) / options["timing_scale_hours"]) ** 2
                proposal = value[a] + reward - cost
                if proposal > value[b]:
                    value[b] = proposal
                    parent[b] = a
        if not count:
            return np.empty(0, dtype=np.int64), 0.0
        last = int(np.argmax(value))
        if not np.isfinite(value[last]) or value[last] <= 0:
            return np.empty(0, dtype=np.int64), 0.0
        path: list[int] = []
        while last >= 0:
            path.append(last)
            last = int(parent[last])
        return np.asarray(path[::-1], dtype=np.int64), float(np.max(value))

    chosen, best = solve(set())
    margin = np.empty(chosen.size)
    for index, event in enumerate(chosen):
        nearby = set(np.flatnonzero(np.abs(h - h[event]) <= options["ambiguity_separation_hours"]))
        _, alternative = solve(nearby)
        margin[index] = max(0.0, best - alternative)
    return chosen, margin, best


def _smooth_segment(t: NDArray[np.float64], x: NDArray[np.float64], width: float) -> NDArray[np.float64]:
    if width <= 0 or t.size <= 1:
        return x.copy()
    half = width / 2.0
    result = np.empty_like(x)
    left = 0
    right = 0
    running_sum = 0.0
    for index, now in enumerate(t):
        while right < len(t) and t[right] <= now + half:
            running_sum += x[right]
            right += 1
        while left < len(t) and t[left] < now - half:
            running_sum -= x[left]
            left += 1
        result[index] = running_sum / (right - left)
    return result


def _timestamp(midnight: datetime | pd.Timestamp, hour: float) -> pd.Timestamp:
    return pd.Timestamp(midnight) + pd.to_timedelta(hour, unit="h")


def compare_dlmo_extract(
    time_hours: NDArray[np.float64],
    x: NDArray[np.float64],
    start_midnight: datetime | pd.Timestamp,
    options: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Detect, sequence, and label CBT troughs and DLMO events."""
    t = np.asarray(time_hours, dtype=np.float64).reshape(-1)
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(t) != len(values) or len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError("time and x must match, contain >=3 samples, and increase")
    midnight = pd.Timestamp(start_midnight)
    if pd.isna(midnight) or midnight != midnight.normalize():
        raise ValueError("start_midnight must be a valid midnight timestamp")
    opts = {**_DEFAULTS, **(options or {})}
    unknown = set(options or {}) - _DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unknown detector option(s): {', '.join(sorted(unknown))}")
    if any(not np.isfinite(value) or value < 0 for value in opts.values()):
        raise ValueError("detector options must be finite and nonnegative")
    if not (
        opts["confirm_hours"] > 0
        and opts["shape_hours"] >= opts["confirm_hours"]
        and opts["local_scale_hours"] >= 2 * opts["shape_hours"]
        and opts["max_gap_hours"] > 0
        and opts["min_separation_hours"] > 0
        and opts["timing_scale_hours"] > 0
        and opts["preferred_min_hours"] < opts["expected_period_hours"] < opts["preferred_max_hours"]
        and opts["restart_hours"] > opts["preferred_max_hours"]
        and opts["min_separation_hours"] < opts["preferred_min_hours"]
        and opts["event_reward"] > opts["restart_penalty"]
        and opts["relax_final_trough"] in (0, 1)
        and 0 < opts["endpoint_confirm_hours"] <= opts["confirm_hours"]
    ):
        raise ValueError("inconsistent detector options")

    smoothed = np.full(values.shape, np.nan)
    segment_ids = np.zeros(values.size, dtype=np.int64)
    segment_ranges: list[tuple[int, int]] = []
    index = 0
    sid = 0
    while index < len(t):
        if not np.isfinite(values[index]):
            index += 1
            continue
        start = index
        while (
            index + 1 < len(t)
            and np.isfinite(values[index + 1])
            and t[index + 1] - t[index] <= opts["max_gap_hours"]
        ):
            index += 1
        stop = index + 1
        sid += 1
        segment_ids[start:stop] = sid
        smoothed[start:stop] = _smooth_segment(t[start:stop], values[start:stop], opts["smooth_hours"])
        segment_ranges.append((start, stop - 1))
        index += 1

    candidates: list[dict[str, Any]] = []
    for sid, (start, end) in enumerate(segment_ranges, start=1):
        index = start + 1
        while index < end:
            if smoothed[index] >= smoothed[index - 1]:
                index += 1
                continue
            plateau_end = index
            while plateau_end < end and smoothed[plateau_end + 1] == smoothed[index]:
                plateau_end += 1
            if plateau_end >= end or smoothed[plateau_end + 1] <= smoothed[index]:
                index = plateau_end + 1
                continue
            trough = (index + plateau_end) // 2
            index = plateau_end + 1
            support = opts["confirm_hours"] + opts["smooth_hours"] / 2
            if t[trough] - t[start] < support:
                continue
            right_confirm = opts["confirm_hours"]
            endpoint_relaxed = False
            if t[end] - t[trough] < support:
                right_confirm = t[end] - t[trough] - opts["smooth_hours"] / 2
                if (
                    not opts["relax_final_trough"]
                    or end != len(t) - 1
                    or right_confirm < opts["endpoint_confirm_hours"]
                ):
                    continue
                endpoint_relaxed = True
            shoulder_times = t[trough] + np.array([-opts["confirm_hours"], right_confirm])
            shoulder_values = np.interp(shoulder_times, t[start : end + 1], smoothed[start : end + 1])
            rise = min(shoulder_values) - smoothed[trough]
            local = np.flatnonzero(np.abs(t[start : end + 1] - t[trough]) <= opts["local_scale_hours"] / 2) + start
            scale = float(np.max(smoothed[local]) - np.min(smoothed[local]))
            if scale <= 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(smoothed[local])))):
                continue
            left = np.flatnonzero((t[start : end + 1] >= t[trough] - opts["shape_hours"]) & (t[start : end + 1] < t[trough])) + start
            right = np.flatnonzero((t[start : end + 1] <= t[trough] + opts["shape_hours"]) & (t[start : end + 1] > t[trough])) + start
            if not left.size or not right.size:
                continue
            if endpoint_relaxed:
                prominence = float(np.max(smoothed[left]) - smoothed[trough])
            else:
                prominence = float(min(np.max(smoothed[left]), np.max(smoothed[right])) - smoothed[trough])
            fraction = prominence / scale
            if fraction < opts["min_prominence_fraction"] or rise / scale < opts["min_shoulder_fraction"]:
                continue
            level = smoothed[trough] + prominence / 2
            left_edge = trough
            right_edge = trough
            while left_edge > start and smoothed[left_edge - 1] < level:
                left_edge -= 1
            while right_edge < end and smoothed[right_edge + 1] < level:
                right_edge += 1
            width = float(t[right_edge] - t[left_edge])
            if width < opts["min_width_hours"]:
                continue
            refine = np.flatnonzero(np.abs(t[start : end + 1] - t[trough]) <= opts["refine_hours"] + 1e-9) + start
            minimum = np.min(values[refine])
            ties = refine[values[refine] == minimum]
            raw_index = int(ties[np.argmin(np.abs(t[ties] - t[trough]))])
            if raw_index <= start or raw_index >= end or values[raw_index] > values[raw_index - 1] or values[raw_index] > values[raw_index + 1]:
                continue
            candidates.append(
                {
                    "index": raw_index,
                    "detection_index": trough,
                    "segment": sid,
                    "prominence": fraction,
                    "rise": rise,
                    "width": width,
                    "quality": min(1.0, 2 * fraction),
                    "edge": bool(opts["refine_hours"] > 0 and raw_index in (refine[0], refine[-1])),
                    "endpoint_relaxed": endpoint_relaxed,
                    "right_confirmation": right_confirm,
                }
            )

    candidates.sort(key=lambda candidate: (candidate["index"], -candidate["quality"]))
    unique_candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate["index"] not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate["index"])
    candidates = unique_candidates
    candidate_hours = np.array([t[item["index"]] for item in candidates])
    cutoff = t[0] + opts["warmup_hours"]
    eligible = candidate_hours >= cutoff - 1e-9
    eligible_indices = np.flatnonzero(eligible)
    eligible_candidates = [candidates[i] for i in eligible_indices]
    path, margins, path_score = compare_dlmo_sequence(
        candidate_hours[eligible],
        np.array([item["quality"] for item in eligible_candidates]),
        np.array([item["segment"] for item in eligible_candidates]),
        opts,
    )
    chosen = eligible_indices[path]
    selected = np.zeros(len(candidates), dtype=bool)
    selected[chosen] = True

    candidate_table = pd.DataFrame(
        {
            "CBTHour": candidate_hours,
            "DetectionHour": [t[item["detection_index"]] for item in candidates],
            "RawX": [values[item["index"]] for item in candidates],
            "ProminenceFraction": [item["prominence"] for item in candidates],
            "ShoulderRise": [item["rise"] for item in candidates],
            "WidthHours": [item["width"] for item in candidates],
            "QualityScore": [item["quality"] for item in candidates],
            "DataSegment": [item["segment"] for item in candidates],
            "RefinementEdge": [item["edge"] for item in candidates],
            "AfterWarmup": eligible,
            "Selected": selected,
            "EndpointRelaxed": [item["endpoint_relaxed"] for item in candidates],
            "RightConfirmationHours": [item["right_confirmation"] for item in candidates],
        }
    )

    event_hours = candidate_hours[chosen]
    gaps = np.r_[np.nan, np.diff(event_hours)] if len(chosen) else np.empty(0)
    event_status = (
        ["anchor"] + ["confirmed"] * (len(chosen) - 1)
        if len(chosen)
        else []
    )
    track_segment = np.ones(len(chosen), dtype=np.int64)
    for position in range(1, len(chosen)):
        previous, current = candidates[chosen[position - 1]], candidates[chosen[position]]
        if current["segment"] != previous["segment"] or gaps[position] >= opts["restart_hours"]:
            event_status[position] = "reacquired"
            track_segment[position:] += 1
        elif gaps[position] < opts["preferred_min_hours"] or gaps[position] > opts["preferred_max_hours"]:
            event_status[position] = "irregular_interval"
    endpoint = np.array([candidates[i]["endpoint_relaxed"] for i in chosen], dtype=bool)
    event_status = [status + "_endpoint_relaxed" if relaxed else status for status, relaxed in zip(event_status, endpoint, strict=True)]
    ambiguous = margins < opts["ambiguity_margin"]
    event_status = [status + "_ambiguous" if is_ambiguous else status for status, is_ambiguous in zip(event_status, ambiguous, strict=True)]
    cbt_datetimes = [_timestamp(midnight, float(hour)) for hour in event_hours]
    dlmo_datetimes = [value - pd.to_timedelta(opts["offset_hours"], unit="h") for value in cbt_datetimes]
    events = pd.DataFrame(
        {
            "EventID": np.arange(1, len(chosen) + 1),
            "CBTDateTime": cbt_datetimes,
            "DLMODateTime": dlmo_datetimes,
            "CBTHour": event_hours,
            "DLMOHour": [value.hour + value.minute / 60 + value.second / 3600 for value in dlmo_datetimes],
            "IntervalHours": gaps,
            "QualityScore": [candidates[i]["quality"] for i in chosen],
            "AlternativePathMargin": margins,
            "Ambiguous": ambiguous,
            "TrackSegment": track_segment,
            "Status": event_status,
            "EndpointRelaxed": endpoint,
            "RightConfirmationHours": [candidates[i]["right_confirmation"] for i in chosen],
        }
    )

    first_date = (_timestamp(midnight, float(t[0] - opts["offset_hours"]))).normalize()
    last_date = (_timestamp(midnight, float(t[-1] - opts["offset_hours"]))).normalize()
    dates = pd.date_range(first_date, last_date, freq="D")
    daily_hours = np.full(len(dates), np.nan)
    candidate_daily = np.full(len(dates), np.nan)
    event_count = np.zeros(len(dates), dtype=np.int64)
    daily_status = ["no_confirmed_trough"] * len(dates)
    event_dates = pd.DatetimeIndex(dlmo_datetimes).normalize()
    for index, day in enumerate(dates):
        event_ids = np.flatnonzero(event_dates == day)
        event_count[index] = len(event_ids)
        if len(event_ids) == 1:
            event_index = int(event_ids[0])
            candidate_daily[index] = events.loc[event_index, "DLMOHour"]
            if ambiguous[event_index]:
                daily_status[index] = "ambiguous"
            else:
                daily_hours[index] = candidate_daily[index]
                daily_status[index] = event_status[event_index]
            continue
        if len(event_ids) > 1:
            daily_status[index] = "multiple_events"
            continue
        left = (day - midnight).total_seconds() / 3600 + opts["offset_hours"]
        right = left + 24
        if right <= cutoff:
            daily_status[index] = "warmup"
        elif left < cutoff:
            daily_status[index] = "partial_warmup"
        elif right + opts["confirm_hours"] + opts["smooth_hours"] / 2 > t[-1]:
            daily_status[index] = "incomplete_end"
        else:
            segment_indices = np.flatnonzero((t >= left) & (t < right))
            if not segment_indices.size or np.any(~np.isfinite(values[segment_indices])) or np.any(np.diff(t[segment_indices]) > opts["max_gap_hours"]):
                daily_status[index] = "data_gap_no_event"
    daily = pd.DataFrame(
        {
            "DLMODate": dates,
            "DLMOHour": daily_hours,
            "CandidateDLMOHour": candidate_daily,
            "EventCount": event_count,
            "Status": daily_status,
        }
    )
    return {
        "candidates": candidate_table,
        "events": events,
        "daily": daily,
        "options": opts,
        "start_midnight": midnight,
        "pf_start_hour": float(t[0]),
        "tracking_start_hour": float(cutoff),
        "pf_start_datetime": _timestamp(midnight, float(t[0])),
        "tracking_start_datetime": _timestamp(midnight, float(cutoff)),
        "path_score": path_score,
        "schema_version": "recoverable_dlmo_v2_endpoint_relaxed",
    }


def compare_dlmo_targets(
    result: dict[str, Any],
    time_hours: NDArray[np.float64],
    x: NDArray[np.float64],
    baseline: datetime | pd.Timestamp,
    study_days: NDArray[np.int64] | list[int] | None,
    boundary_hour: float,
) -> pd.DataFrame:
    """Assign tracked events to requested days, or every available recording day."""
    t = np.asarray(time_hours, dtype=np.float64)
    values = np.asarray(x, dtype=np.float64)
    base = pd.Timestamp(baseline)
    events = result["events"]
    options = result["options"]
    midnight = result["start_midnight"]
    if study_days is None:
        if not t.size or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError("time_hours must be finite and strictly increasing")
        if not np.isfinite(boundary_hour):
            raise ValueError("boundary_hour must be finite")
        offset = float(options["offset_hours"])
        first_day = int(np.floor((t[0] - boundary_hour - offset) / 24.0))
        last_day = int(np.floor((t[-1] - boundary_hour - offset) / 24.0))
        dates = [
            midnight + pd.to_timedelta(day, unit="D")
            for day in range(first_day, last_day + 1)
        ]
        recording_days = [
            int((evening.normalize() - base.normalize()).days + 1)
            for evening in dates
        ]
    else:
        recording_days = [int(day) for day in study_days]
        dates = [base + pd.to_timedelta(day - 1, unit="D") for day in recording_days]
    rows: list[dict[str, Any]] = []
    event_evenings = (pd.to_datetime(events["DLMODateTime"]) - pd.to_timedelta(boundary_hour, unit="h")).dt.normalize()
    for recording_day, evening in zip(recording_days, dates, strict=True):
        ids = np.flatnonzero(event_evenings == evening)
        hour = np.nan
        candidate = np.nan
        actual: Any = pd.NaT
        status = "no_confirmed_trough"
        if len(ids) == 1:
            event = events.iloc[int(ids[0])]
            candidate = float(event["DLMOHour"])
            actual = pd.Timestamp(event["DLMODateTime"])
            status = "ambiguous" if bool(event["Ambiguous"]) else str(event["Status"])
            if not bool(event["Ambiguous"]):
                hour = candidate
        elif len(ids) > 1:
            status = "multiple_events"
        else:
            left = (evening + pd.to_timedelta(boundary_hour, unit="h") - midnight).total_seconds() / 3600 + options["offset_hours"]
            right = left + 24
            if right <= result["tracking_start_hour"]:
                status = "warmup"
            elif left < result["tracking_start_hour"]:
                status = "partial_warmup"
            elif left > t[-1] or right < t[0]:
                status = "outside_record"
            elif right + options["confirm_hours"] + options["smooth_hours"] / 2 > t[-1]:
                status = "incomplete_end"
            else:
                indices = np.flatnonzero((t >= left) & (t < right))
                if not indices.size or np.any(~np.isfinite(values[indices])) or np.any(np.diff(t[indices]) > options["max_gap_hours"]):
                    status = "data_gap_no_event"
        rows.append(
            {
                "RecordingDay": int(recording_day),
                "EveningDate": evening,
                "DLMOHour": hour,
                "CandidateDLMOHour": candidate,
                "EventCount": len(ids),
                "Status": status,
                "DLMODateTime": actual,
            }
        )
    return pd.DataFrame(rows)
