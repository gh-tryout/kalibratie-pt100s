"""Vergelijk een referentie-temperatuurmeter met maximaal 20 PT100-kanalen."""

from __future__ import annotations

import ast
import csv
import html
import io
import math
import operator
import re
import struct
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_EXTENSIONS = {".csv", ".txt", ".dat", ".dbf"}
OPEN_ABS_LIMIT = 1.0e6
MAX_PT100_CHANNELS = 20


def list_data_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    skip_names = {"requirements.txt", "readme.txt", "gebruik.txt", "runtime.txt"}
    files = [
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in DATA_EXTENSIONS
        and p.name.lower() not in skip_names
    ]
    return sorted(files, key=lambda p: p.name.lower())


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _friendly_ref_name(field: str) -> str:
    match = re.fullmatch(r"CHN(\d+)CELS", field.strip(), flags=re.IGNORECASE)
    if match:
        return f"CHN{match.group(1)}"
    return field.strip()


def _parse_dbf_number(raw: str) -> float:
    text = raw.strip().replace(",", ".")
    if not text or text in {".", "-", "+"}:
        return float("nan")
    try:
        value = float(text)
    except ValueError:
        return float("nan")
    if not np.isfinite(value) or abs(value) >= OPEN_ABS_LIMIT:
        return float("nan")
    return value


def _load_reference_dbf(path: str | Path) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Lees een DBF van de referentie-meter (DATE, TIME, CHNxCELS)."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 33 or data[0] not in (0x03, 0x83, 0x8B, 0x30):
        raise ValueError(f"Geen DBF-referentiebestand: {path.name}")

    nrec = struct.unpack_from("<I", data, 4)[0]
    header_len = struct.unpack_from("<H", data, 8)[0]
    record_len = struct.unpack_from("<H", data, 10)[0]
    if header_len < 33 or record_len < 2 or nrec < 1:
        raise ValueError(f"DBF-header is ongeldig: {path.name}")

    fields: list[tuple[str, int]] = []
    pos = 32
    while pos + 32 <= header_len and data[pos] != 0x0D:
        name = data[pos : pos + 11].split(b"\x00", 1)[0].decode("latin-1", errors="replace").strip()
        length = data[pos + 16]
        if name and length > 0:
            fields.append((name, length))
        pos += 32

    by_name = {name.upper(): (name, length) for name, length in fields}
    if "DATE" not in by_name or "TIME" not in by_name:
        raise ValueError(f"Referentiebestand mist DATE/TIME: {path.name}")

    temp_fields = [(name, length) for name, length in fields if name.upper().endswith("CELS")]
    if not temp_fields:
        raise ValueError(f"Referentiebestand mist temperatuurkanalen (°C): {path.name}")

    labels = { _friendly_ref_name(name): _friendly_ref_name(name) for name, _ in temp_fields }
    # kolomnaam = label; bij een botsing de ruwe veldnaam aanhouden
    col_for_field: list[tuple[str, str]] = []
    used: set[str] = set()
    for name, _length in temp_fields:
        label = _friendly_ref_name(name)
        column = label if label not in used else name
        used.add(column)
        labels[column] = label if column == label else name
        col_for_field.append((name, column))

    date_idx = [name for name, _ in fields].index(by_name["DATE"][0])
    time_idx = [name for name, _ in fields].index(by_name["TIME"][0])

    times: list[str] = []
    columns: dict[str, list[float]] = {column: [] for _name, column in col_for_field}
    offset = header_len
    for _i in range(nrec):
        rec = data[offset : offset + record_len]
        offset += record_len
        if len(rec) < record_len or rec[0:1] in (b"*", b"\x2a"):
            continue
        cursor = 1
        values: list[str] = []
        for _name, length in fields:
            values.append(rec[cursor : cursor + length].decode("latin-1", errors="replace"))
            cursor += length
        times.append(values[date_idx].strip() + " " + values[time_idx].strip())
        field_values = {name: values[idx] for idx, (name, _length) in enumerate(fields)}
        for name, column in col_for_field:
            columns[column].append(_parse_dbf_number(field_values[name]))

    out = pd.DataFrame({"tijd": _parse_datetime_series(pd.Series(times)), **columns})
    channel_names = [column for _name, column in col_for_field]
    valid_idx = np.flatnonzero(out["tijd"].notna().to_numpy()).astype(np.intp)
    if valid_idx.size == 0:
        raise ValueError(f"Referentiebestand heeft geen leesbare tijdstippen: {path.name}")
    out = pd.DataFrame({col: out[col].to_numpy().take(valid_idx) for col in out.columns})
    out = out.sort_values("tijd").drop_duplicates(subset=["tijd"]).reset_index(drop=True)
    return out, channel_names, {name: labels[name] for name in channel_names}


def _channel_labels_from_preamble(lines: list[str], header_idx: int) -> dict[str, str]:
    labels: dict[str, str] = {}
    for i, line in enumerate(lines[:header_idx]):
        if not line.lower().startswith("channel,name"):
            continue
        for raw in lines[i + 1 : header_idx]:
            if not raw.strip() or raw.lower().startswith("scan"):
                break
            parts = next(csv.reader([raw]))
            if not parts or not parts[0].strip().isdigit():
                continue
            channel = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else ""
            labels[channel] = f"{channel} {name}".strip() if name else channel
        break
    return labels


def _is_pt100_text(text: str) -> bool:
    head = text[:8000].lower()
    return (
        "temp 4-wire rtd" in head
        or "34972" in head
        or "scan,time," in head
        or "scan;time;" in head
    )


def _load_pt100_csv(path: str | Path) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Lees een Keysight-scan (max. 20 PT100-kanalen). Open kanalen worden NaN."""
    path = Path(path)
    text = _read_text(path)
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:200]):
        low = line.lower().lstrip()
        if low.startswith("scan,time") or low.startswith("scan;time"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"PT100-bestand mist de scan-kop (Scan, Time, kanalen): {path.name}")

    preamble_labels = _channel_labels_from_preamble(lines, header_idx)
    table = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), sep=None, engine="python")
    table.columns = [str(c).strip() for c in table.columns]
    table = table.loc[:, [c for c in table.columns if c and not str(c).lower().startswith("unnamed")]]

    time_col = None
    for col in table.columns:
        if col.lower() in {"time", "tijd", "datetime"}:
            time_col = col
            break
    if time_col is None:
        raise ValueError(f"PT100-bestand mist een tijdkolom: {path.name}")

    channel_cols: list[tuple[str, str]] = []
    for col in table.columns:
        if col == time_col or col.lower().startswith("alarm") or col.lower() == "scan":
            continue
        match = re.match(r"^\s*(\d+)", col)
        if not match:
            continue
        channel = match.group(1)
        channel_cols.append((channel, col))

    if not channel_cols:
        raise ValueError(f"PT100-bestand mist temperatuurkanalen: {path.name}")
    if len(channel_cols) > MAX_PT100_CHANNELS:
        channel_cols = channel_cols[:MAX_PT100_CHANNELS]

    labels = {channel: preamble_labels.get(channel, channel) for channel, _col in channel_cols}
    tijd = _parse_datetime_series(table[time_col].map(_normalize_clock_text))
    data: dict[str, pd.Series] = {"tijd": tijd}
    for channel, col in channel_cols:
        values = _to_float(table[col])
        values = values.where(values.abs() < OPEN_ABS_LIMIT)
        data[channel] = values

    out = pd.DataFrame(data)
    valid_idx = np.flatnonzero(out["tijd"].notna().to_numpy()).astype(np.intp)
    if valid_idx.size == 0:
        raise ValueError(f"PT100-bestand heeft geen leesbare tijdstippen: {path.name}")
    out = pd.DataFrame({col: out[col].to_numpy().take(valid_idx) for col in out.columns})
    out = out.sort_values("tijd").drop_duplicates(subset=["tijd"]).reset_index(drop=True)
    channels = [channel for channel, _col in channel_cols]
    return out, channels, labels


def classify_file(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".dbf":
        return "referentie"
    try:
        text = _read_text(path)
    except OSError as exc:
        raise ValueError(f"Kan bestand niet lezen: {path.name}") from exc
    if _is_pt100_text(text):
        return "pt100"
    return "onbekend"


def load_measurement(path: str | Path) -> tuple[str, pd.DataFrame, list[str], dict[str, str]]:
    """Laad een meetbestand als (soort, dataframe, kanalen, labels)."""
    path = Path(path)
    kind = classify_file(path)
    if kind == "referentie":
        frame, channels, labels = _load_reference_dbf(path)
        return kind, frame, channels, labels
    if kind == "pt100":
        frame, channels, labels = _load_pt100_csv(path)
        return kind, frame, channels, labels
    raise ValueError(
        f"Bestandstype niet herkend: {path.name}. "
        "Bestand 1 is een DBF van de referentie-meter. "
        "Bestand 2 is een PT100-scan (CSV) of een tweede referentie-DBF."
    )


def load_measurement_bytes(filename: str, data: bytes) -> tuple[str, pd.DataFrame, list[str], dict[str, str]]:
    safe_name = Path(filename).name or "upload.bin"
    with tempfile.TemporaryDirectory(prefix="pt100_") as tmpdir:
        path = Path(tmpdir) / safe_name
        path.write_bytes(data)
        return load_measurement(path)


def combine_pt100_measurements(
    parts: list[tuple[pd.DataFrame, list[str], dict[str, str]]],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Voeg PT100-scans samen op tijdstip.

    Hetzelfde kanaalnummer is hetzelfde kanaal. Bij een gelijk tijdstip blijft
    de eerste eindige waarde staan. Kanalen die alleen in een later bestand
    zitten, komen erbij.
    """
    if not parts:
        return pd.DataFrame(columns=["tijd"]), [], {}
    if len(parts) == 1:
        frame, channels, labels = parts[0]
        return frame, list(channels), dict(labels)

    channels: list[str] = []
    labels: dict[str, str] = {}
    frames: list[pd.DataFrame] = []
    for frame, frame_channels, frame_labels in parts:
        for channel in frame_channels:
            if channel not in labels:
                channels.append(channel)
                labels[channel] = frame_labels.get(channel, channel)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    for channel in channels:
        if channel not in combined.columns:
            combined[channel] = np.nan
    ns = combined["tijd"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    order = np.argsort(ns, kind="mergesort")
    ns = ns[order]
    taken = {col: combined[col].to_numpy().take(order) for col in ["tijd", *channels]}
    if len(ns) <= 1 or bool(np.all(np.diff(ns) != 0)):
        return pd.DataFrame(taken), channels, labels

    starts = np.flatnonzero(np.r_[True, np.diff(ns) != 0]).astype(np.intp)
    ends = np.r_[starts[1:], len(ns)].astype(np.intp)
    reduced: dict[str, np.ndarray] = {}
    for channel in channels:
        values = pd.to_numeric(pd.Series(taken[channel]), errors="coerce").to_numpy(dtype=float)
        picked = np.empty(len(starts), dtype=float)
        for index, (start, end) in enumerate(zip(starts, ends)):
            finite_idx = np.flatnonzero(np.isfinite(values[start:end]))
            picked[index] = values[start + int(finite_idx[0])] if finite_idx.size else np.nan
        reduced[channel] = picked
    out = pd.DataFrame({"tijd": taken["tijd"][starts], **reduced})
    return out, channels, labels


def as_single_channel_measurement(
    frame: pd.DataFrame,
    channel: str,
    labels: dict[str, str],
    channel_id: str,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Maak van een referentie-DBF één meetkanaal voor de vergelijking."""
    if channel not in frame.columns:
        raise ValueError(f"Kanaal {channel} ontbreekt in de meting.")
    source_label = labels.get(channel, channel)
    display = f"{channel_id} ({source_label})"
    out = pd.DataFrame(
        {
            "tijd": frame["tijd"].to_numpy(),
            channel_id: pd.to_numeric(frame[channel], errors="coerce").to_numpy(dtype=float),
        }
    )
    return out, [channel_id], {channel_id: display}


def valid_channels(df: pd.DataFrame, channels: list[str]) -> list[str]:
    """Kanalen die echt meelopen. Een handvol open-kanaalpieken telt niet mee."""
    kept: list[str] = []
    n_rows = len(df)
    for channel in channels:
        if channel not in df.columns or n_rows == 0:
            continue
        values = pd.to_numeric(df[channel], errors="coerce").to_numpy(dtype=float)
        n_ok = int(np.isfinite(values).sum())
        if n_ok >= 10 and n_ok / n_rows >= 0.02:
            kept.append(channel)
    return kept


def _to_float(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_clock_text(value) -> str:
    text = str(value).strip()
    # Keysight: 1-2-2024 13:25:29:341 → seconden.milliseconden
    return re.sub(r"(\d{1,2}:\d{2}:\d{2}):(\d{1,3})$", r"\1.\2", text)


def _parse_datetime_series(values: pd.Series) -> pd.Series:
    """Parse datetimes with fixed formats — dateutil fallback crashes on Python 3.14."""
    text = values.map(_normalize_clock_text)
    text = text.str.replace(
        r"^(\d{1,2})-(\d{1,2})-(\d{4})(.*)$",
        lambda match: (
            f"{int(match.group(1)):02d}-{int(match.group(2)):02d}-{match.group(3)}{match.group(4)}"
        ),
        regex=True,
    )
    lower = text.str.lower()
    invalid = text.isin(["", "nan", "None", "NaT", "NaN"]) | lower.str.startswith("date")
    text = text.mask(invalid)

    formats = (
        "%d-%m-%Y %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%Y-%m-%d %H:%M",
    )
    best: pd.Series | None = None
    best_count = -1
    target = int((~invalid).sum())
    for fmt in formats:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        count = int(parsed.notna().sum())
        if count > best_count:
            best = parsed
            best_count = count
        if best_count == target and best_count > 0:
            break
    if best is None:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    return best.astype("datetime64[ns]")


def _ts_ns(value) -> int:
    """Timestamp as int64 nanoseconds (avoids pandas.Timedelta on Python 3.14)."""
    return int(pd.Timestamp(value).value)


def shift_datetime_series(series: pd.Series, seconds: float) -> pd.Series:
    """Tijdoffset zonder pandas.Timedelta."""
    if not seconds:
        return series
    ns = series.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    ns = ns + int(float(seconds) * 1_000_000_000)
    return pd.to_datetime(ns, unit="ns")


def filter_time_range(df: pd.DataFrame, t_start, t_end) -> pd.DataFrame:
    """Selecteer rijen binnen [t_start, t_end] zonder pandas boolean-indexing."""
    if df.empty:
        return df.copy()
    ns = df["tijd"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    a = _ts_ns(t_start)
    b = _ts_ns(t_end)
    if a > b:
        a, b = b, a
    idxs = np.flatnonzero((ns >= a) & (ns <= b)).astype(np.intp)
    if idxs.size == 0:
        return df.iloc[0:0].copy()
    return pd.DataFrame({col: df[col].to_numpy().take(idxs) for col in df.columns})


def _nearest_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str,
    max_delta_seconds: float | None = None,
) -> pd.DataFrame:
    """Nearest-time join without pandas.merge_asof (crashes on Python 3.14)."""
    if left.empty or right.empty:
        return left.iloc[0:0].copy()

    left = left.sort_values(on).reset_index(drop=True)
    right = right.sort_values(on).reset_index(drop=True)

    left_ns = left[on].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    right_ns = right[on].to_numpy(dtype="datetime64[ns]").astype(np.int64)

    idx = np.searchsorted(right_ns, left_ns, side="left")
    idx_lo = np.clip(idx - 1, 0, len(right_ns) - 1)
    idx_hi = np.clip(idx, 0, len(right_ns) - 1)
    delta_lo = np.abs(left_ns - right_ns[idx_lo])
    delta_hi = np.abs(left_ns - right_ns[idx_hi])
    use_hi = delta_hi < delta_lo
    best = np.where(use_hi, idx_hi, idx_lo)
    best_delta = np.where(use_hi, delta_hi, delta_lo)
    if max_delta_seconds is None:
        matched = np.ones(len(left_ns), dtype=bool)
    else:
        tol_ns = int(float(max_delta_seconds) * 1_000_000_000)
        matched = best_delta <= tol_ns

    if not matched.any():
        return left.iloc[0:0].copy()

    left_idx = np.flatnonzero(matched).astype(np.intp)
    best_idx = best[matched].astype(np.intp)
    out = pd.DataFrame({col: left[col].to_numpy().take(left_idx) for col in left.columns})
    for col in right.columns:
        if col == on:
            continue
        out[col] = right[col].to_numpy().take(best_idx)
    return out


def merge_measurements(
    reference: pd.DataFrame,
    pt100: pd.DataFrame,
    channels: list[str],
    *,
    pt100_offset_seconds: float = 0.0,
    max_delta_seconds: float | None = None,
) -> pd.DataFrame:
    """Koppel elke PT100-scan aan de dichtstbijzijnde referentiemeting."""
    left_cols = ["tijd", *[ch for ch in channels if ch in pt100.columns]]
    left = pt100[left_cols].copy().sort_values("tijd")
    left["tijd"] = shift_datetime_series(left["tijd"], pt100_offset_seconds)

    right = reference[["tijd", "t"]].copy().sort_values("tijd")
    right = right.rename(columns={"t": "t_ref"})

    merged = _nearest_join(
        left,
        right,
        on="tijd",
        max_delta_seconds=max_delta_seconds,
    )
    if merged.empty:
        return merged
    for channel in channels:
        if channel not in merged.columns:
            continue
        sensor = merged[channel].to_numpy(dtype=float)
        ref = merged["t_ref"].to_numpy(dtype=float)
        delta = sensor - ref
        delta[~np.isfinite(sensor) | ~np.isfinite(ref)] = np.nan
        merged[f"d_{channel}"] = delta
    return merged.reset_index(drop=True)


def nearest_setpoint(temp: float, step: float, tolerance: float) -> float | None:
    if pd.isna(temp) or not np.isfinite(temp) or step <= 0:
        return None
    setpoint = float(np.round(float(temp) / float(step)) * float(step))
    setpoint = float(np.round(setpoint, 6))
    if setpoint == 0:
        setpoint = 0.0
    if abs(float(temp) - setpoint) <= float(tolerance) + 1e-9:
        return setpoint
    return None


def _rate_k_per_min(tijd_ns: np.ndarray, temp: np.ndarray, lookback_s: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Verandering van de referentie over de voorgaande minuut, in K/min."""
    lookback_ns = int(float(lookback_s) * 1_000_000_000)
    earlier = np.searchsorted(tijd_ns, tijd_ns - lookback_ns, side="right") - 1
    earlier = np.clip(earlier, 0, len(tijd_ns) - 1)
    dt_min = (tijd_ns - tijd_ns[earlier]) / 1_000_000_000 / 60.0
    delta = temp - temp[earlier]
    rate = np.full(len(temp), np.nan, dtype=float)
    usable = (dt_min >= 0.5) & np.isfinite(temp) & np.isfinite(temp[earlier])
    rate[usable] = delta[usable] / dt_min[usable]
    return rate, dt_min


def detect_stable_blocks(
    merged: pd.DataFrame,
    channels: list[str],
    *,
    temp_step: float = 1.0,
    temp_tolerance: float = 0.5,
    max_rate_k_per_min: float = 0.1,
    min_stable_minutes: float = 5.0,
    calc_start_before_minutes: float = 15.0,
    calc_end_before_minutes: float = 2.0,
    progress=None,
) -> pd.DataFrame:
    """Vind temperatuurplateaus en gemiddel over een venster vóór het einde.

    De temperatuurverandering en de minimale duur bepalen het tijdstip waarop
    het systeem naar het volgende setpoint gaat. Het gemiddelde gebruikt alleen
    het stuk van ``calc_start_before_minutes`` tot ``calc_end_before_minutes``
    vóór dat tijdstip.
    """
    if merged.empty or "t_ref" not in merged.columns:
        return pd.DataFrame()

    work = merged.sort_values("tijd").reset_index(drop=True)
    tijd_ns = work["tijd"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    ref_all = work["t_ref"].to_numpy(dtype=float)
    rate, _dt_min = _rate_k_per_min(tijd_ns, ref_all)
    stable_point = np.isfinite(ref_all) & np.isfinite(rate) & (np.abs(rate) < float(max_rate_k_per_min))
    gap_min = np.zeros(len(work), dtype=float)
    if len(work) > 1:
        gap_min[1:] = (tijd_ns[1:] - tijd_ns[:-1]) / 1_000_000_000 / 60.0

    def _tick(fraction: float, detail: str) -> None:
        if progress is None:
            return
        progress(float(fraction), detail)

    runs: list[tuple[int, int]] = []
    i = 0
    n = len(work)
    scan_step = max(n // 12, 1)
    _tick(0.02, "punten nalopen")
    while i < n:
        if progress is not None and i % scan_step == 0:
            _tick(0.05 + 0.25 * (i / max(n, 1)), "punten nalopen")
        if not stable_point[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and stable_point[j + 1] and gap_min[j + 1] <= 3.0:
            j += 1
        duration_min = (tijd_ns[j] - tijd_ns[i]) / 1_000_000_000 / 60.0
        if duration_min + 1e-9 >= float(min_stable_minutes) and (j - i + 1) >= 3:
            runs.append((i, j))
        i = j + 1

    start_before_ns = int(float(calc_start_before_minutes) * 60 * 1_000_000_000)
    end_before_ns = int(float(calc_end_before_minutes) * 60 * 1_000_000_000)
    blocks: list[dict] = []
    accepted_setpoints: list[float] = []
    run_total = max(len(runs), 1)

    for run_index, (i0, i1) in enumerate(runs):
        _tick(0.35 + 0.6 * (run_index / run_total), "tijdblokken")
        run = work.iloc[i0 : i1 + 1]
        run_ref = run["t_ref"].to_numpy(dtype=float)
        if int(np.isfinite(run_ref).sum()) < 3:
            continue
        if i1 + 1 < n:
            anchor_ns = int(tijd_ns[i1 + 1])
            change_at = work["tijd"].iloc[i1 + 1]
        else:
            anchor_ns = int(tijd_ns[i1])
            change_at = work["tijd"].iloc[i1]
        calc_start_ns = anchor_ns - start_before_ns
        calc_end_ns = anchor_ns - end_before_ns
        if calc_end_ns <= calc_start_ns:
            continue
        run_ns = tijd_ns[i0 : i1 + 1]
        selected = np.flatnonzero((run_ns >= calc_start_ns) & (run_ns <= calc_end_ns)).astype(np.intp)
        if selected.size < 3:
            continue
        stable = pd.DataFrame({col: run[col].to_numpy().take(selected) for col in run.columns})
        ref = stable["t_ref"].to_numpy(dtype=float)
        ref_ok = np.isfinite(ref)
        if int(ref_ok.sum()) < 3:
            continue
        t_ref_mean = float(ref[ref_ok].mean())
        run_mean = float(run_ref[np.isfinite(run_ref)].mean())
        sp = nearest_setpoint(t_ref_mean, temp_step, temp_tolerance)
        if sp is None:
            sp = nearest_setpoint(run_mean, temp_step, temp_tolerance)
        if sp is None:
            continue
        t_ref_std = float(ref[ref_ok].std(ddof=1)) if int(ref_ok.sum()) > 1 else 0.0
        s0_ns = _ts_ns(stable["tijd"].iloc[0])
        s1_ns = _ts_ns(stable["tijd"].iloc[-1])
        stable_min = (s1_ns - s0_ns) / 1_000_000_000 / 60.0
        hold_min = (int(tijd_ns[i1]) - int(tijd_ns[i0])) / 1_000_000_000 / 60.0

        prev_sp = accepted_setpoints[-1] if accepted_setpoints else None
        if prev_sp is None:
            richting = "start"
        elif sp > prev_sp:
            richting = "opwaarts"
        elif sp < prev_sp:
            richting = "neerwaarts"
        else:
            richting = "herhaal"
        accepted_setpoints.append(sp)

        common = {
            "setpoint": sp,
            "richting": richting,
            "blok_start": run["tijd"].iloc[0],
            "blok_einde": change_at,
            "volgende_blok": change_at,
            "stabiel_van": stable["tijd"].iloc[0],
            "stabiel_tot": stable["tijd"].iloc[-1],
            "blokduur_min": round(hold_min, 1),
            "stabiel_min": round(stable_min, 1),
            "n_venster": int(len(stable)),
            "t_ref": t_ref_mean,
            "std_t_ref": t_ref_std,
        }

        for channel in channels:
            if channel not in stable.columns:
                continue
            sensor = stable[channel].to_numpy(dtype=float)
            ok = np.isfinite(sensor) & ref_ok
            n_ok = int(ok.sum())
            if n_ok < 3:
                continue
            delta = sensor[ok] - ref[ok]
            blocks.append(
                {
                    **common,
                    "kanaal": channel,
                    "n": n_ok,
                    "t_pt100": float(sensor[ok].mean()),
                    "d_t": float(delta.mean()),
                    "std_d_t": float(delta.std(ddof=1)) if n_ok > 1 else 0.0,
                    "std_t_pt100": float(sensor[ok].std(ddof=1)) if n_ok > 1 else 0.0,
                }
            )

    _tick(1.0, "tijdblokken")
    return pd.DataFrame(blocks)


DIFF_COLUMNS = [
    ("setpoint", "T set [°C]"),
    ("richting", "Richting"),
    ("kanaal", "PT100"),
    ("stabiel_van", "Stabiel van"),
    ("stabiel_tot", "Stabiel tot"),
    ("stabiel_min", "Stabiel [min]"),
    ("n", "n"),
    ("t_ref", "T ref [°C]"),
    ("t_pt100", "T PT100 [°C]"),
    ("d_t", "ΔT (PT100−ref) [°C]"),
    ("std_d_t", "Std ΔT [°C]"),
    ("std_t_ref", "Std T ref [°C]"),
]

BLOCK_COLUMNS = [
    ("setpoint", "T set [°C]"),
    ("richting", "Richting"),
    ("kanaal", "PT100"),
    ("stabiel_van", "Stabiel van"),
    ("stabiel_tot", "Stabiel tot"),
    ("volgende_blok", "Volgende setpoint"),
    ("blokduur_min", "Blok [min]"),
    ("stabiel_min", "Stabiel [min]"),
    ("n", "n"),
    ("t_ref", "T ref [°C]"),
    ("std_t_ref", "Std T ref [°C]"),
    ("t_pt100", "T PT100 [°C]"),
    ("d_t", "ΔT (PT100−ref) [°C]"),
    ("std_d_t", "Std ΔT [°C]"),
    ("std_t_pt100", "Std T PT100 [°C]"),
]


def coupled_columns(channels: list[str], labels: dict[str, str]) -> list[tuple[str, str]]:
    columns = [("tijd", "Tijd"), ("t_ref", "T referentie [°C]")]
    for channel in channels:
        label = labels.get(channel, channel)
        columns.append((channel, f"T {label} [°C]"))
        columns.append((f"d_{channel}", f"ΔT {label} [°C]"))
    return columns


def format_for_display(df: pd.DataFrame, columns: list[tuple[str, str]]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[label for _, label in columns])
    out = pd.DataFrame()
    for src, label in columns:
        if src not in df.columns:
            continue
        series = df[src]
        if src == "setpoint":
            numeric = pd.to_numeric(series, errors="coerce")
            whole = np.isfinite(numeric.to_numpy(dtype=float)).all() and np.allclose(
                numeric.to_numpy(dtype=float) % 1, 0, atol=1e-6
            )
            if whole:
                out[label] = numeric.round(0).astype("Int64")
            else:
                out[label] = numeric
        elif src == "kanaal":
            out[label] = series.astype(str)
        else:
            out[label] = series
    return out


def delta_matrix(
    blocks: pd.DataFrame,
    channels: list[str],
    labels: dict[str, str],
) -> pd.DataFrame:
    """Matrix: rijen chronologisch op gemiddelde T ref, kolommen de PT100-kanalen (ΔT)."""
    if blocks.empty or not channels:
        return pd.DataFrame(columns=["T ref [°C]", *[labels.get(ch, ch) for ch in channels]])

    grouped: dict[tuple, list[int]] = {}
    for idx in range(len(blocks)):
        key = (blocks.iloc[idx]["stabiel_van"], blocks.iloc[idx]["stabiel_tot"])
        grouped.setdefault(key, []).append(idx)

    ordered = sorted(grouped, key=lambda key: _ts_ns(key[0]))
    rows: list[dict] = []
    for key in ordered:
        idxs = grouped[key]
        first = blocks.iloc[idxs[0]]
        by_channel = {str(blocks.iloc[idx]["kanaal"]): blocks.iloc[idx]["d_t"] for idx in idxs}
        row: dict = {"T ref [°C]": float(first["t_ref"])}
        for channel in channels:
            row[labels.get(channel, channel)] = by_channel.get(str(channel), float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def wide_delta_table(blocks: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Eén rij per stabiel setpoint, met ΔT per PT100-kanaal."""
    if blocks.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    keys = ["setpoint", "richting", "stabiel_van", "stabiel_tot"]
    grouped: dict[tuple, list[int]] = {}
    for idx in range(len(blocks)):
        key = tuple(blocks.iloc[idx][col] for col in keys)
        grouped.setdefault(key, []).append(idx)
    for key, idxs in grouped.items():
        first = blocks.iloc[idxs[0]]
        row: dict = {
            "T set [°C]": first["setpoint"],
            "Richting": first["richting"],
            "Stabiel van": first["stabiel_van"],
            "Stabiel tot": first["stabiel_tot"],
            "T ref [°C]": first["t_ref"],
            "n": first["n_venster"],
        }
        for idx in idxs:
            rec = blocks.iloc[idx]
            label = labels.get(str(rec["kanaal"]), str(rec["kanaal"]))
            row[f"ΔT {label} [°C]"] = rec["d_t"]
        rows.append(row)
    return pd.DataFrame(rows)


class UncertaintyFormulaError(ValueError):
    """De ingevoerde onzekerheidsformule is niet bruikbaar."""


class UncertaintyFormula:
    def __init__(self, expression: str, parameters: tuple[str, ...], error: str = ""):
        self.expression = expression
        self.parameters = parameters
        self.error = error

    @property
    def ok(self) -> bool:
        return not self.error


class AssessmentTables:
    def __init__(
        self,
        delta: pd.DataFrame,
        u_measurement: pd.DataFrame,
        u_total: pd.DataFrame,
        judged: pd.DataFrame,
        delta_columns: list[str],
        uncertainty_columns: list[str],
        judged_columns: list[str],
        formula_was_negative: bool,
        formula_not_finite: bool,
        row_ids: list[str] | None = None,
        formula_u: list[float] | None = None,
        formula_columns: list[str] | None = None,
    ):
        self.delta = delta
        self.u_measurement = u_measurement
        self.u_total = u_total
        self.judged = judged
        self.delta_columns = delta_columns
        self.uncertainty_columns = uncertainty_columns
        self.judged_columns = judged_columns
        self.formula_was_negative = formula_was_negative
        self.formula_not_finite = formula_not_finite
        self.row_ids = row_ids or []
        self.formula_u = formula_u or []
        self.formula_columns = formula_columns or []


_FORMULA_FUNCTIONS = {
    "exp": math.exp,
    "sqrt": math.sqrt,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pow": pow,
}
_FORMULA_CONSTANTS = {"e": math.e, "E": math.e, "pi": math.pi}
_FORMULA_RESERVED = set(_FORMULA_FUNCTIONS) | set(_FORMULA_CONSTANTS) | {"T"}
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_JUDGEMENT_FAIL = "fail"
_JUDGEMENT_BORDER = "border"
FORMULA_U_COLUMN = "u formule [°C]"
_FAIL_BG = "#8B0000"
_FAIL_FG = "#ffffff"
_BORDER_BG = "#f6b26b"
_BORDER_FG = "#1a1a1a"


def _normalize_uncertainty_expression(raw: str) -> str:
    text = raw.strip()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("×", "*").replace("·", "*")
    text = re.sub(r"^[uU]\s*=\s*", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    placeholders: list[str] = []

    def hold(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"#{len(placeholders) - 1}#"

    text = re.sub(r"\d+(?:\.\d+)?[eE][+-]?\d+", hold, text)
    text = text.replace("^", "**")
    names = "|".join(sorted(_FORMULA_FUNCTIONS, key=len, reverse=True))
    text = re.sub(rf"\b(?:{names})\b", hold, text)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", text)
        text = re.sub(r"(?<=\))(?=[A-Za-z\d(])", "*", text)
        text = re.sub(r"(?<=[A-Za-z])(?=\()", "*", text)
        text = re.sub(r"(?<=[A-Za-z])(?=T\b)", "*", text)
    for index, original in enumerate(placeholders):
        text = text.replace(f"#{index}#", original)
    return text


def _formula_parameter_names(tree: ast.AST) -> tuple[str, ...]:
    names: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in _FORMULA_RESERVED and node.id not in names:
                names.append(node.id)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return tuple(names)


def parse_uncertainty_formula(raw: str) -> UncertaintyFormula:
    """Lees een formule u(T). Leeg betekent u = 0.

    T is de temperatuur. e is het grondtal van de natuurlijke logaritme.
    aT+b en a*e^(-b*T) zijn allebei geldig.
    """
    if len(raw) > 300:
        return UncertaintyFormula("", (), "De formule is te lang.")
    if not raw.strip():
        return UncertaintyFormula("0", (), "")
    expression = _normalize_uncertainty_expression(raw)
    if not expression:
        return UncertaintyFormula("0", (), "")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return UncertaintyFormula("", (), "De formule is niet geldig. Voorbeelden: a*T + b en a*e^(-b*T).")
    parameters = _formula_parameter_names(tree)
    try:
        _eval_uncertainty(tree, {"T": 0.0, **_FORMULA_CONSTANTS, **{name: 0.0 for name in parameters}})
    except UncertaintyFormulaError as exc:
        return UncertaintyFormula("", (), str(exc))
    except (ZeroDivisionError, OverflowError, ValueError):
        pass
    return UncertaintyFormula(expression, parameters, "")


def _eval_uncertainty(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_uncertainty(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UncertaintyFormulaError("Alleen getallen zijn toegestaan.")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise UncertaintyFormulaError(f"Onbekende naam '{node.id}'.")
        return float(env[node.id])
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return float(_UNARYOPS[type(node.op)](_eval_uncertainty(node.operand, env)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_uncertainty(node.left, env)
        right = _eval_uncertainty(node.right, env)
        return float(_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.Call):
        if node.keywords or not isinstance(node.func, ast.Name) or node.func.id not in _FORMULA_FUNCTIONS:
            raise UncertaintyFormulaError("Deze functie is niet toegestaan. Gebruik exp, ln, log, log10, sqrt of abs.")
        args = [_eval_uncertainty(arg, env) for arg in node.args]
        try:
            return float(_FORMULA_FUNCTIONS[node.func.id](*args))
        except (TypeError, ValueError) as exc:
            raise UncertaintyFormulaError(f"{node.func.id}() kan zo niet worden uitgerekend.") from exc
    raise UncertaintyFormulaError("Dit onderdeel van de formule is niet toegestaan.")


def evaluate_uncertainty(formula: UncertaintyFormula, temperature: float, parameters: dict[str, float]) -> float:
    """Reken de formule uit bij één temperatuur. Het teken blijft behouden."""
    if formula.error:
        raise UncertaintyFormulaError(formula.error)
    env = {"T": float(temperature), **_FORMULA_CONSTANTS}
    for name in formula.parameters:
        if name not in parameters:
            raise UncertaintyFormulaError(f"Coëfficiënt {name} ontbreekt.")
        env[name] = float(parameters[name])
    tree = ast.parse(formula.expression, mode="eval")
    try:
        value = _eval_uncertainty(tree, env)
    except ZeroDivisionError:
        return float("nan")
    except OverflowError:
        return float("nan")
    if not np.isfinite(value):
        return float("nan")
    return float(value)


def uncertainty_formula_text(formula: UncertaintyFormula, parameters: dict[str, float]) -> str:
    """Formule met ingevulde coëfficiënten, geschikt om te tonen."""
    if not formula.ok:
        return ""
    text = formula.expression
    for name in sorted(formula.parameters, key=len, reverse=True):
        value = float(parameters.get(name, float("nan")))
        text = re.sub(rf"\b{re.escape(name)}\b", f"({value:.6g})", text)
    return "u = " + text.replace("**", "^")


def judgement_mark(value, tolerance: float) -> str:
    """'fail' boven de tolerantie, 'border' vanaf de helft tot en met de tolerantie."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    limit = float(tolerance)
    magnitude = abs(number)
    if limit <= 0:
        return _JUDGEMENT_FAIL if magnitude > 1e-12 else ""
    if magnitude > limit + 1e-9:
        return _JUDGEMENT_FAIL
    if magnitude + 1e-9 >= 0.5 * limit:
        return _JUDGEMENT_BORDER
    return ""


def _format_setpoint_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "n" in out.columns:
        out["n"] = pd.to_numeric(out["n"], errors="coerce").round(0).astype("Int64")
    if "T set [°C]" not in out.columns:
        return out
    numeric = pd.to_numeric(out["T set [°C]"], errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if len(values) and np.isfinite(values).all() and np.allclose(values % 1, 0, atol=1e-6):
        out["T set [°C]"] = numeric.round(0).astype("Int64")
    return out


def _assessment_row_id(first: pd.Series, index: int) -> str:
    return f"{float(first['setpoint']):.6g}|{first['richting']}|{int(_ts_ns(first['stabiel_van']))}|{index}"


def _grouped_block_rows(blocks: pd.DataFrame) -> list[tuple[pd.Series, dict[str, pd.Series]]]:
    keys = ["setpoint", "richting", "stabiel_van", "stabiel_tot"]
    grouped: dict[tuple, list[int]] = {}
    for idx in range(len(blocks)):
        key = tuple(blocks.iloc[idx][col] for col in keys)
        grouped.setdefault(key, []).append(idx)
    rows: list[tuple[pd.Series, dict[str, pd.Series]]] = []
    for idxs in grouped.values():
        first = blocks.iloc[idxs[0]]
        by_channel = {str(blocks.iloc[idx]["kanaal"]): blocks.iloc[idx] for idx in idxs}
        rows.append((first, by_channel))
    return rows


class AssessmentBase:
    """ΔT en standaardonzekerheid per tijdblok, zonder de kalibratieformule.

    Een formulewijziging of een handmatige cel hoeft dit niet opnieuw te maken.
    """

    def __init__(
        self,
        ident: pd.DataFrame,
        delta: pd.DataFrame,
        u_measurement: pd.DataFrame,
        d_t: np.ndarray,
        u_meas: np.ndarray,
        temperatures: np.ndarray,
        row_ids: list[str],
        channels: list[str],
        delta_columns: list[str],
        uncertainty_columns: list[str],
        formula_columns: list[str],
    ):
        self.ident = ident
        self.delta = delta
        self.u_measurement = u_measurement
        self.d_t = d_t
        self.u_meas = u_meas
        self.temperatures = temperatures
        self.row_ids = row_ids
        self.channels = channels
        self.delta_columns = delta_columns
        self.uncertainty_columns = uncertainty_columns
        self.formula_columns = formula_columns

    @property
    def ready(self) -> bool:
        return len(self.row_ids) > 0 and len(self.channels) > 0


def _assessment_column_names(
    channels: list[str], labels: dict[str, str]
) -> tuple[list[str], list[str], list[str]]:
    delta_columns = [f"ΔT {labels.get(channel, channel)} [°C]" for channel in channels]
    uncertainty_columns = [f"u {labels.get(channel, channel)} [°C]" for channel in channels]
    formula_columns = [f"u formule {labels.get(channel, channel)} [°C]" for channel in channels]
    return delta_columns, uncertainty_columns, formula_columns


def _empty_assessment_tables(
    delta_columns: list[str],
    uncertainty_columns: list[str],
    formula_columns: list[str],
) -> AssessmentTables:
    ident_columns = ["T set [°C]", "Richting", "T ref [°C]"]
    judged_column_order = [column for pair in zip(formula_columns, delta_columns) for column in pair]
    empty = pd.DataFrame(columns=[*ident_columns, *delta_columns])
    empty_u = pd.DataFrame(columns=[*ident_columns, *uncertainty_columns])
    empty_judged = pd.DataFrame(columns=[*ident_columns, *judged_column_order])
    return AssessmentTables(
        empty,
        empty_u,
        empty_u.copy(),
        empty_judged,
        delta_columns,
        uncertainty_columns,
        delta_columns,
        False,
        False,
        formula_columns=formula_columns,
    )


def assessment_measurement_base(
    blocks: pd.DataFrame,
    channels: list[str],
    labels: dict[str, str],
) -> AssessmentBase:
    """Eenmalig per meting: ΔT en u_meting = std(ΔT) / √n. Geen formule."""
    channels = [str(channel) for channel in channels]
    labels = {str(key): str(value) for key, value in labels.items()}
    delta_columns, uncertainty_columns, formula_columns = _assessment_column_names(channels, labels)
    grouped = _grouped_block_rows(blocks) if not blocks.empty and channels else []
    row_count = len(grouped)
    channel_count = len(channels)
    d_t = np.full((row_count, channel_count), np.nan)
    u_meas = np.full((row_count, channel_count), np.nan)
    temperatures = np.full(row_count, np.nan)
    row_ids: list[str] = []
    ident_rows: list[dict] = []
    for index, (first, by_channel) in enumerate(grouped):
        row_ids.append(_assessment_row_id(first, index))
        temperature = float(first["t_ref"])
        temperatures[index] = temperature
        ident_rows.append(
            {
                "T set [°C]": first["setpoint"],
                "Richting": first["richting"],
                "T ref [°C]": temperature,
            }
        )
        for position, channel in enumerate(channels):
            record = by_channel.get(channel)
            if record is None:
                continue
            delta = float(record["d_t"])
            count = float(record["n"])
            std = float(record["std_d_t"])
            d_t[index, position] = delta
            if count > 0 and np.isfinite(std) and np.isfinite(count):
                u_meas[index, position] = std / math.sqrt(count)
    ident = _format_setpoint_frame(
        pd.DataFrame(ident_rows, columns=["T set [°C]", "Richting", "T ref [°C]"])
    )
    delta_frame = ident.copy()
    measurement_frame = ident.copy()
    for position, column in enumerate(delta_columns):
        delta_frame[column] = d_t[:, position] if row_count else np.array([], dtype=float)
    for position, column in enumerate(uncertainty_columns):
        measurement_frame[column] = u_meas[:, position] if row_count else np.array([], dtype=float)
    return AssessmentBase(
        ident,
        delta_frame,
        measurement_frame,
        d_t,
        u_meas,
        temperatures,
        row_ids,
        channels,
        delta_columns,
        uncertainty_columns,
        formula_columns,
    )


def formula_uncertainty_by_row(base: AssessmentBase, u_at_temperature=None) -> tuple[np.ndarray, bool, bool]:
    """u(T) per tijdblok, vóór handmatige aanpassingen. Negatieve uitkomsten worden absoluut."""
    values = np.full(len(base.row_ids), np.nan)
    was_negative = False
    not_finite = False
    if u_at_temperature is None:
        return values, was_negative, not_finite
    for index, temperature in enumerate(base.temperatures):
        if not np.isfinite(temperature):
            continue
        raw = float(u_at_temperature(float(temperature)))
        if not np.isfinite(raw):
            not_finite = True
            continue
        if raw < 0:
            was_negative = True
            raw = abs(raw)
        values[index] = raw
    return values, was_negative, not_finite


def filter_uncertainty_overrides(
    overrides: dict[str, float] | None,
    row_ids: list[str],
    channels: list[str],
) -> dict[str, float]:
    """Bewaar alleen sleutels 'rij|kanaal' die nog in de huidige tabellen staan."""
    valid_rows = set(row_ids)
    valid_channels = {str(channel) for channel in channels}
    kept: dict[str, float] = {}
    for key, value in (overrides or {}).items():
        row_id, separator, channel = str(key).rpartition("|")
        if not separator or row_id not in valid_rows or channel not in valid_channels:
            continue
        try:
            number = abs(float(value))
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            kept[f"{row_id}|{channel}"] = number
    return kept


def apply_uncertainty_overrides(
    base: AssessmentBase,
    formula_per_row: np.ndarray,
    overrides: dict[str, float] | None,
) -> np.ndarray:
    """Matrix (rijen × kanalen) met de formule, en per cel een eventuele handmatige waarde."""
    row_count = len(base.row_ids)
    channel_count = len(base.channels)
    if row_count == 0 or channel_count == 0:
        return np.zeros((row_count, channel_count))
    matrix = np.repeat(np.asarray(formula_per_row, dtype=float).reshape(row_count, 1), channel_count, axis=1)
    for index, row_id in enumerate(base.row_ids):
        for position, channel in enumerate(base.channels):
            manual = (overrides or {}).get(f"{row_id}|{channel}")
            if manual is None:
                continue
            try:
                number = abs(float(manual))
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                matrix[index, position] = number
    return matrix


def changed_uncertainty_rows(previous: np.ndarray | None, current: np.ndarray) -> list[int] | None:
    """Rijnummers waarvan de kalibratie-onzekerheid veranderde.

    None betekent dat elke rij opnieuw moet. Een lege lijst betekent dat er niets veranderde.
    """
    if previous is None or getattr(previous, "shape", None) != current.shape:
        return None
    if current.size == 0:
        return []
    same = np.isclose(previous, current, rtol=0.0, atol=5e-5, equal_nan=True)
    if bool(same.all()):
        return []
    return np.flatnonzero(~np.all(same, axis=1)).tolist()


def _judged_block(d_t: np.ndarray, u_total: np.ndarray) -> np.ndarray:
    judged = np.full(d_t.shape, np.nan, dtype=float)
    valid = np.isfinite(d_t) & np.isfinite(u_total)
    judged = np.where(valid & (d_t > 0), d_t + u_total, judged)
    judged = np.where(valid & (d_t < 0), d_t - u_total, judged)
    judged = np.where(valid & (d_t == 0), 0.0, judged)
    return judged


def _uncertainty_block(u_cal: np.ndarray, u_meas: np.ndarray) -> np.ndarray:
    finite = np.isfinite(u_cal) & np.isfinite(u_meas)
    return np.where(finite, u_cal + u_meas, np.nan)


def _same_assessment_layout(previous: AssessmentTables | None, base: AssessmentBase) -> bool:
    return (
        previous is not None
        and previous.row_ids == base.row_ids
        and previous.formula_columns == base.formula_columns
        and previous.delta_columns == base.delta_columns
        and previous.uncertainty_columns == base.uncertainty_columns
    )


def assemble_assessment(
    base: AssessmentBase,
    u_calibration: np.ndarray,
    formula_per_row: np.ndarray,
    formula_was_negative: bool,
    formula_not_finite: bool,
    previous: AssessmentTables | None = None,
    dirty_rows: list[int] | None = None,
) -> AssessmentTables:
    """Vul totale onzekerheid en beoordeeld verschil.

    dirty_rows beperkt het schrijven tot die rijen. None vult alle rijen.
    """
    if not base.ready:
        return _empty_assessment_tables(base.delta_columns, base.uncertainty_columns, base.formula_columns)
    formula_values = [
        float(value) if np.isfinite(value) else float("nan") for value in np.asarray(formula_per_row, dtype=float)
    ]
    if dirty_rows is not None and len(dirty_rows) == 0 and _same_assessment_layout(previous, base):
        assert previous is not None
        return AssessmentTables(
            previous.delta,
            previous.u_measurement,
            previous.u_total,
            previous.judged,
            base.delta_columns,
            base.uncertainty_columns,
            base.delta_columns,
            formula_was_negative,
            formula_not_finite,
            list(base.row_ids),
            formula_values,
            list(base.formula_columns),
        )

    full = (
        dirty_rows is None
        or not _same_assessment_layout(previous, base)
        or len(dirty_rows) == len(base.row_ids)
    )
    positions = list(range(len(base.row_ids))) if full else [int(index) for index in dirty_rows or []]
    selected = np.asarray(positions, dtype=int)
    u_total = _uncertainty_block(u_calibration[selected], base.u_meas[selected])
    judged = _judged_block(base.d_t[selected], u_total)

    if full:
        total_frame = base.ident.copy()
        judged_frame = base.ident.copy()
        for position, column in enumerate(base.uncertainty_columns):
            total_frame[column] = u_total[:, position]
        for position, (formula_column, delta_column) in enumerate(zip(base.formula_columns, base.delta_columns)):
            judged_frame[formula_column] = u_calibration[:, position]
            judged_frame[delta_column] = judged[:, position]
    else:
        assert previous is not None
        total_frame = previous.u_total.copy()
        judged_frame = previous.judged.copy()
        total_locations = [total_frame.columns.get_loc(column) for column in base.uncertainty_columns]
        formula_locations = [judged_frame.columns.get_loc(column) for column in base.formula_columns]
        delta_locations = [judged_frame.columns.get_loc(column) for column in base.delta_columns]
        for offset, row_index in enumerate(positions):
            for position, location in enumerate(total_locations):
                total_frame.iat[row_index, location] = float(u_total[offset, position])
            for position, location in enumerate(formula_locations):
                judged_frame.iat[row_index, location] = float(u_calibration[row_index, position])
            for position, location in enumerate(delta_locations):
                judged_frame.iat[row_index, location] = float(judged[offset, position])

    return AssessmentTables(
        base.delta,
        base.u_measurement,
        total_frame,
        judged_frame,
        base.delta_columns,
        base.uncertainty_columns,
        base.delta_columns,
        formula_was_negative,
        formula_not_finite,
        list(base.row_ids),
        formula_values,
        list(base.formula_columns),
    )


def build_assessment_tables(
    blocks: pd.DataFrame,
    channels: list[str],
    labels: dict[str, str],
    u_at_temperature=None,
    u_overrides: dict[str, float] | None = None,
) -> AssessmentTables:
    """Vier tabellen: ΔT, u van de meting, totale u, en het beoordeelde verschil.

    Zonder tijden en zonder n. u_meting = std(ΔT) / √n van dat kanaal.
    u_totaal = u_formule + u_meting. Een override vervangt u_formule voor die meetwaarde.
    Het beoordeelde verschil is ΔT ± u_totaal, het teken volgt de afwijking.
    """
    base = assessment_measurement_base(blocks, channels, labels)
    if not base.ready:
        return _empty_assessment_tables(base.delta_columns, base.uncertainty_columns, base.formula_columns)
    formula_per_row, was_negative, not_finite = formula_uncertainty_by_row(base, u_at_temperature)
    kept = filter_uncertainty_overrides(u_overrides, base.row_ids, base.channels)
    matrix = apply_uncertainty_overrides(base, formula_per_row, kept)
    return assemble_assessment(base, matrix, formula_per_row, was_negative, not_finite)


def _report_number(column: str, value) -> str:
    if pd.isna(value):
        return ""
    if column == "Richting":
        return str(value)
    if column == "n":
        return str(int(value))
    number = float(value)
    if not np.isfinite(number):
        return ""
    if column == "T set [°C]" and abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    if column == "T ref [°C]":
        return f"{number:.3f}"
    return f"{number:.4f}"


def _report_table_html(df: pd.DataFrame, marks: dict[tuple[int, str], str] | None = None) -> str:
    if df.empty:
        return "<p><em>Geen gegevens</em></p>"
    marks = marks or {}
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in df.columns)
    body: list[str] = []
    for row_index, row in enumerate(df.itertuples(index=False)):
        cells: list[str] = []
        for column, value in zip(df.columns, row):
            text = html.escape(_report_number(str(column), value))
            mark = marks.get((row_index, str(column)), "")
            if mark == _JUDGEMENT_FAIL:
                cells.append(
                    f'<td style="background-color:{_FAIL_BG};color:{_FAIL_FG}" bgcolor="{_FAIL_BG}">{text}</td>'
                )
            elif mark == _JUDGEMENT_BORDER:
                cells.append(
                    f'<td style="background-color:{_BORDER_BG};color:{_BORDER_FG}" bgcolor="{_BORDER_BG}">{text}</td>'
                )
            else:
                cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table class="data" border="1" cellspacing="0" cellpadding="4">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _judgement_marks(df: pd.DataFrame, columns: list[str], tolerance: float) -> dict[tuple[int, str], str]:
    marks: dict[tuple[int, str], str] = {}
    for row_index in range(len(df)):
        for column in columns:
            if column not in df.columns:
                continue
            mark = judgement_mark(df.iloc[row_index][column], tolerance)
            if mark:
                marks[(row_index, column)] = mark
    return marks


def build_assessment_report_html(
    *,
    path1_name: str,
    path2_name: str,
    ref_label: str,
    range_start,
    range_end,
    laborant: str,
    referentie: str,
    calibrator: str,
    meetmiddel: str,
    formula_text: str,
    diff_tol: float,
    tables: AssessmentTables,
    max_rate: float,
    min_stable: float,
    calc_start: float,
    calc_end: float,
) -> str:
    """HTML-kalibratierapport dat Word kan openen."""

    def meta_line(label: str, value: str) -> str:
        text = value.strip() or "—"
        return f"<div><strong>{html.escape(label)}:</strong> {html.escape(text)}</div>"

    notes: list[str] = []
    if tables.formula_was_negative:
        notes.append("Een negatieve uitkomst van de formule is als absolute waarde gebruikt.")
    if tables.formula_not_finite:
        notes.append("Voor minstens één temperatuur gaf de formule geen eindig getal.")
    note_html = "".join(f"<p>{html.escape(note)}</p>" for note in notes)
    judged_marks = _judgement_marks(tables.judged, tables.judged_columns, diff_tol)
    return f"""<!DOCTYPE html>
<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" lang="nl">
<head>
  <meta charset="utf-8"/>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
  <meta name="ProgId" content="Word.Document"/>
  <title>Kalibratierapport PT100</title>
  <!--[if gte mso 9]>
  <xml><w:WordDocument><w:View>Print</w:View><w:Zoom>100</w:Zoom></w:WordDocument></xml>
  <![endif]-->
  <style>
    body {{ font-family: Calibri, Segoe UI, sans-serif; font-size: 11pt; color: #1a1a1a; margin: 24px; }}
    h1 {{ font-size: 18pt; margin-bottom: 4pt; }}
    h2 {{ font-size: 14pt; margin-top: 18pt; margin-bottom: 4pt; }}
    .meta div {{ margin: 1pt 0; }}
    table.data {{ border-collapse: collapse; margin: 6pt 0 14pt; }}
    table.data td, table.data th {{ border: 1px solid #666666; padding: 3pt 6pt; font-size: 10pt; text-align: left; }}
    table.data th {{ background: #f3f3f3; }}
  </style>
</head>
<body>
  <h1>Kalibratierapport PT100</h1>
  <div class="meta">
    {meta_line("Laborant", laborant)}
    {meta_line("Referentie", referentie)}
    {meta_line("Calibrator", calibrator)}
    {meta_line("Gekalibreerd meetmiddel", meetmiddel)}
    <div><strong>Bestand 1 (referentie):</strong> {html.escape(path1_name)} · kanaal {html.escape(ref_label)}</div>
    <div><strong>Meting:</strong> {html.escape(path2_name)}</div>
    <div><strong>Tijdspanne:</strong> {html.escape(str(range_start))} → {html.escape(str(range_end))}</div>
    <div><strong>Tolerantie verschil:</strong> {diff_tol:g} °C</div>
    <div><strong>Tijdblok:</strong> einde zodra de referentie sneller dan {max_rate:g} K/min verandert,
    na minstens {min_stable:g} min stabiel. Gemiddelde van {calc_start:g} tot {calc_end:g} min vóór dat einde.</div>
    <div><strong>Meetonzekerheid kalibratie:</strong> {html.escape(formula_text or "u = 0")}</div>
    <div><strong>Gegenereerd:</strong> {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</div>
  </div>
  {note_html}
  <h2>Gemeten verschil ΔT</h2>
  <p>ΔT = PT100 − referentie, in °C. Eén rij per setpoint, zonder tijden.</p>
  {_report_table_html(tables.delta)}
  <h2>Standaardonzekerheid van de meting</h2>
  <p>Standaarddeviatie van ΔT in het tijdblok, gedeeld door de wortel van het aantal waarnemingen van dat kanaal. Eenheid: °C.</p>
  {_report_table_html(tables.u_measurement)}
  <h2>Totale meetonzekerheid</h2>
  <p>De onzekerheid uit de formule, uitgerekend bij de gemiddelde referentietemperatuur, opgeteld bij de standaardonzekerheid van de meting. Eenheid: °C.</p>
  {_report_table_html(tables.u_total)}
  <h2>Beoordeeld verschil</h2>
  <p>Het gemeten verschil, met de totale meetonzekerheid opgeteld in de richting van de afwijking.
  Een positief verschil wordt hoger en een negatief verschil lager.
  Elke kolom u formule hoort bij de meetwaarde ernaast. Die komt uit de formule, of is met de hand ingevuld.
  Donkerrood: de absolute waarde is groter dan de tolerantie van {diff_tol:g} °C.
  Oranje: grensgeval, vanaf de helft van de tolerantie tot en met de tolerantie.</p>
  {_report_table_html(tables.judged, judged_marks)}
</body>
</html>
"""
