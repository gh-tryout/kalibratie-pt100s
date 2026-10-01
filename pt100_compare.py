"""Vergelijk een referentie-temperatuurmeter met maximaal 20 PT100-kanalen."""

from __future__ import annotations

import csv
import io
import re
import struct
import tempfile
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
        "Bestand 1 is een DBF van de referentie-meter, bestand 2 een PT100-scan (CSV)."
    )


def load_measurement_bytes(filename: str, data: bytes) -> tuple[str, pd.DataFrame, list[str], dict[str, str]]:
    safe_name = Path(filename).name or "upload.bin"
    with tempfile.TemporaryDirectory(prefix="pt100_") as tmpdir:
        path = Path(tmpdir) / safe_name
        path.write_bytes(data)
        return load_measurement(path)


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
    max_delta_seconds: float,
) -> pd.DataFrame:
    """Nearest-time join without pandas.merge_asof (crashes on Python 3.14)."""
    if left.empty or right.empty:
        return left.iloc[0:0].copy()

    left = left.sort_values(on).reset_index(drop=True)
    right = right.sort_values(on).reset_index(drop=True)

    left_ns = left[on].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    right_ns = right[on].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    tol_ns = int(float(max_delta_seconds) * 1_000_000_000)

    idx = np.searchsorted(right_ns, left_ns, side="left")
    idx_lo = np.clip(idx - 1, 0, len(right_ns) - 1)
    idx_hi = np.clip(idx, 0, len(right_ns) - 1)
    delta_lo = np.abs(left_ns - right_ns[idx_lo])
    delta_hi = np.abs(left_ns - right_ns[idx_hi])
    use_hi = delta_hi < delta_lo
    best = np.where(use_hi, idx_hi, idx_lo)
    best_delta = np.where(use_hi, delta_hi, delta_lo)
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
    max_delta_seconds: float = 30.0,
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


def detect_stable_blocks(
    merged: pd.DataFrame,
    channels: list[str],
    *,
    temp_step: float = 1.0,
    temp_tolerance: float = 0.5,
    settle_minutes: float = 5.0,
    min_stable_minutes: float = 5.0,
    end_margin_minutes: float = 2.0,
) -> pd.DataFrame:
    """Vind temperatuurplateaus op de referentie en gemiddel per PT100.

    Stabiel venster, zelfde systematiek als de voorbeeldapp:
    - start na ``settle_minutes`` (inregeltijd, daarna is de waarde stabiel)
    - eindigt uiterlijk ``end_margin_minutes`` vóór het volgende setpoint
      (compensatie voor klokken die niet synchroon lopen)
    """
    if merged.empty or "t_ref" not in merged.columns:
        return pd.DataFrame()

    work = merged.sort_values("tijd").reset_index(drop=True)
    assigned = [
        nearest_setpoint(v, temp_step, temp_tolerance) for v in work["t_ref"].to_numpy(dtype=float)
    ]
    work["setpoint"] = pd.Series(assigned, dtype="object")

    plateaus: list[tuple[int, int]] = []
    start = 0
    n = len(work)
    for i in range(1, n + 1):
        if i == n or work.at[i, "setpoint"] != work.at[start, "setpoint"]:
            plateaus.append((start, i))
            start = i

    blocks: list[dict] = []
    settle_ns = int(float(settle_minutes) * 60 * 1_000_000_000)
    end_margin_ns = int(float(end_margin_minutes) * 60 * 1_000_000_000)
    accepted_setpoints: list[float] = []

    for p_idx, (i0, i1) in enumerate(plateaus):
        sp = work.at[i0, "setpoint"]
        if sp is None or (isinstance(sp, float) and np.isnan(sp)):
            continue
        sp = float(sp)
        block = work.iloc[i0:i1].reset_index(drop=True)
        t0 = block["tijd"].iloc[0]
        t1 = block["tijd"].iloc[-1]
        t0_ns = _ts_ns(t0)
        t1_ns = _ts_ns(t1)
        duration_min = (t1_ns - t0_ns) / 1_000_000_000 / 60.0

        stable_start_ns = t0_ns + settle_ns
        stable_end_ns = t1_ns
        next_start = None
        for q in range(p_idx + 1, len(plateaus)):
            j0, _j1 = plateaus[q]
            next_sp = work.at[j0, "setpoint"]
            if next_sp is None or (isinstance(next_sp, float) and np.isnan(next_sp)):
                continue
            next_start = work.at[j0, "tijd"]
            next_start_ns = _ts_ns(next_start)
            stable_end_ns = min(stable_end_ns, next_start_ns - end_margin_ns)
            break

        if stable_end_ns <= stable_start_ns:
            continue

        tijd_ns = block["tijd"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        idxs = np.flatnonzero((tijd_ns >= stable_start_ns) & (tijd_ns <= stable_end_ns)).astype(np.intp)
        if idxs.size == 0:
            continue

        stable = pd.DataFrame({col: block[col].to_numpy().take(idxs) for col in block.columns})
        s0_ns = _ts_ns(stable["tijd"].iloc[0])
        s1_ns = _ts_ns(stable["tijd"].iloc[-1])
        stable_min = (s1_ns - s0_ns) / 1_000_000_000 / 60.0
        if len(stable) < 3 or stable_min < min_stable_minutes * 0.5:
            continue

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

        ref = stable["t_ref"].to_numpy(dtype=float)
        ref_ok = np.isfinite(ref)
        if int(ref_ok.sum()) < 3:
            continue
        t_ref_mean = float(ref[ref_ok].mean())
        t_ref_std = float(ref[ref_ok].std(ddof=1)) if int(ref_ok.sum()) > 1 else 0.0
        common = {
            "setpoint": sp,
            "richting": richting,
            "blok_start": t0,
            "blok_einde": t1,
            "volgende_blok": next_start,
            "stabiel_van": stable["tijd"].iloc[0],
            "stabiel_tot": stable["tijd"].iloc[-1],
            "blokduur_min": round(duration_min, 1),
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
