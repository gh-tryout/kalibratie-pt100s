"""Streamlit-app: vergelijk PT100-kanalen met een referentie-temperatuurmeter."""

from __future__ import annotations

import html
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pt100_compare import (
    BLOCK_COLUMNS,
    coupled_columns,
    detect_stable_blocks,
    filter_time_range,
    format_for_display,
    list_data_files,
    load_measurement,
    load_measurement_bytes,
    merge_measurements,
    shift_datetime_series,
    valid_channels,
    wide_delta_table,
)

st.set_page_config(
    page_title="PT100-kalibratie",
    page_icon="🌡️",
    layout="wide",
)

st.title("PT100-kalibratie vs referentie")
st.caption(
    "Upload eerst de referentie-temperatuurmeter (bestand 1, DBF), kies een tijdspanne, "
    "daarna de PT100-scan (bestand 2, maximaal 20 kanalen). "
    "Stabiele blokken slaan de inregeltijd over en stoppen vóór het volgende setpoint, "
    "zodat klokken niet synchroon hoeven te lopen."
)


def default_data_folder() -> str:
    env = os.environ.get("PT100_DATA_DIR")
    if env:
        return env
    if getattr(sys, "frozen", False):
        return str(Path(sys.executable).resolve().parent)
    return str(Path(__file__).resolve().parent)


def _column_config(df: pd.DataFrame) -> dict:
    config: dict = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            config[col] = st.column_config.DatetimeColumn(col, format="YYYY-MM-DD HH:mm:ss", width="medium")
        elif pd.api.types.is_integer_dtype(series):
            config[col] = st.column_config.NumberColumn(col, format="%d", width="small")
        elif pd.api.types.is_float_dtype(series):
            config[col] = st.column_config.NumberColumn(col, format="%.3f", width="small")
        else:
            config[col] = st.column_config.TextColumn(col, width="small")
    return config


def _to_py_dt(value) -> datetime:
    ts = pd.Timestamp(value)
    return ts.to_pydatetime().replace(tzinfo=None)


def _df_to_html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>Geen gegevens</em></p>"
    return df.to_html(index=False, border=0, classes="data", escape=True)


def _difference_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).startswith("ΔT")]


# Zelfde volgorde als de categorische kleuren van de overige Streamlit-grafieken.
CHANNEL_COLORS = (
    "#0068c9",
    "#83c9ff",
    "#ff2b2b",
    "#ffabab",
    "#29b09d",
    "#7defa1",
    "#ff8700",
    "#ffd16a",
    "#6d3fc0",
    "#d5dae5",
)


def _channel_color(index: int) -> str:
    return CHANNEL_COLORS[index % len(CHANNEL_COLORS)]


def _mark_differences(df: pd.DataFrame, tolerance: float):
    """Kleur verschilcellen naar de mate van |ΔT| ten opzichte van de tolerantie."""
    columns = _difference_columns(df)
    if df.empty or not columns:
        return df
    limit = float(tolerance)

    def paint(value) -> str:
        if pd.isna(value):
            return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(number):
            return ""
        ratio = abs(number) / limit if limit > 0 else (0.0 if number == 0 else 3.0)
        if ratio < 0.25:
            return "background-color: #c6efce"
        if ratio < 0.5:
            return "background-color: #ffe566"
        if ratio <= 1:
            return "background-color: #f6b26b"
        if ratio <= 2:
            return "background-color: #e67c73"
        return "background-color: #b71c1c; color: #ffffff"

    return df.style.map(paint, subset=columns)


def _ask_directory(initial: str) -> str:
    """Open de Windows-mapkiezer. Lege string betekent dat de gebruiker annuleert."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        start = initial if Path(initial).is_dir() else str(Path.home())
        return filedialog.askdirectory(
            title="Kies de datamap",
            initialdir=start,
            mustexist=True,
        )
    finally:
        root.destroy()


def _choose_data_folder() -> None:
    try:
        chosen = _ask_directory(st.session_state.get("data_folder", default_data_folder()))
    except Exception as exc:  # noqa: BLE001
        st.session_state["data_folder_error"] = str(exc)
        return
    st.session_state["data_folder_error"] = ""
    if chosen:
        st.session_state["data_folder"] = str(Path(chosen))


def _hover_bind_script(div_id: str) -> str:
    """Licht bij hover één kanaallijn uit en dim de overige."""
    return f"""
<script>
(function() {{
  function bind() {{
    const gd = document.getElementById({div_id!r});
    if (!gd || typeof gd.on !== "function") {{
      setTimeout(bind, 50);
      return;
    }}
    let active = null;
    let hoverAt = 0;
    function paint(index) {{
      const n = gd.data.length;
      const opacity = [];
      const width = [];
      const color = [];
      for (let i = 0; i < n; i++) {{
        opacity.push(index === null || i === index ? 1 : 0.12);
        width.push(i === index ? 3.5 : 1.6);
        const line = gd.data[i].line || {{}};
        const marker = gd.data[i].marker || {{}};
        color.push(line.color || marker.color);
      }}
      Plotly.restyle(gd, {{
        opacity: opacity,
        "line.width": width,
        "line.color": color,
        "marker.color": color,
      }});
    }}
    gd.on("plotly_hover", function(ev) {{
      if (!ev.points || !ev.points.length) return;
      hoverAt = Date.now();
      const index = ev.points[0].curveNumber;
      if (index === active) return;
      active = index;
      paint(index);
    }});
    gd.on("plotly_unhover", function() {{
      if (Date.now() - hoverAt < 80) return;
      if (active === null) return;
      active = null;
      paint(null);
    }});
  }}
  bind();
}})();
</script>
"""


def _plotly_hover_chart(fig: go.Figure, *, div_id: str, height: int) -> None:
    chart = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=div_id,
        config={"responsive": True, "displayModeBar": True},
    )
    st.components.v1.html(chart + _hover_bind_script(div_id), height=height, scrolling=False)


def _unique_windows(block_df: pd.DataFrame) -> pd.DataFrame:
    if block_df.empty:
        return block_df
    return block_df.drop_duplicates(subset=["setpoint", "stabiel_van", "stabiel_tot"]).reset_index(drop=True)


def build_report_html(
    *,
    path1_name: str,
    path2_name: str,
    ref_label: str,
    range_start,
    range_end,
    settle_min: float,
    end_margin: float,
    fig_main: go.Figure,
    fig_diff: go.Figure,
    fig_blocks: go.Figure | None,
    merged_display: pd.DataFrame,
    block_display: pd.DataFrame,
    wide_display: pd.DataFrame,
    diff_tol: float,
) -> str:
    chart_main = fig_main.to_html(full_html=False, include_plotlyjs="cdn")
    chart_diff = fig_diff.to_html(full_html=False, include_plotlyjs=False)
    chart_blocks = ""
    if fig_blocks is not None:
        chart_blocks = fig_blocks.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="delta_setpoints",
        ) + _hover_bind_script("delta_setpoints")
    if not wide_display.empty:
        blocks_section = f"""
        <h2>Stabiele verschillen</h2>
        <p>Start na {settle_min:g} min inregeltijd; einde uiterlijk {end_margin:g} min
        vóór het volgende setpoint. ΔT = PT100 − referentie.</p>
        <h3>Overzicht ΔT per kanaal</h3>
        <p>Kleur volgens |ΔT| t.o.v. de tolerantie van {diff_tol:g} °C:
        lichtgroen &lt; 0,25×, geel 0,25–0,5×, oranje 0,5–1×, rood 1–2×, donkerrood &gt; 2×.</p>
        {_mark_differences(wide_display, diff_tol).to_html(index=False, border=0, classes="data")}
        <h3>Details per blok</h3>
        {_df_to_html_table(block_display)}
        <h3>ΔT per setpoint</h3>
        {chart_blocks}
        """
    else:
        blocks_section = "<h2>Stabiele verschillen</h2><p><em>Geen stabiele blokken gevonden.</em></p>"

    return f"""<!DOCTYPE html>
<html lang="nl">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>PT100-kalibratie</title>
  <style>
    body {{ font-family: Segoe UI, system-ui, sans-serif; margin: 24px; color: #1a1a1a; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 0.2rem; }}
    h2 {{ margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
    h3 {{ margin-top: 1.4rem; }}
    .meta {{ color: #444; margin-bottom: 1.5rem; }}
    table.data {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.5rem 0 1.5rem; }}
    table.data th, table.data td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    table.data th {{ background: #f3f3f3; }}
    table.data tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>PT100-kalibratie vs referentie</h1>
  <div class="meta">
    <div><strong>Bestand 1 (referentie):</strong> {html.escape(path1_name)} · kanaal {html.escape(ref_label)}</div>
    <div><strong>Bestand 2 (PT100):</strong> {html.escape(path2_name)}</div>
    <div><strong>Tijdspanne:</strong> {html.escape(str(range_start))} → {html.escape(str(range_end))}</div>
    <div><strong>Gegenereerd:</strong> {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</div>
  </div>
  <h2>Temperatuur</h2>
  {chart_main}
  <h2>Verschillen (PT100 − referentie)</h2>
  {chart_diff}
  <h2>Gekoppelde tabel</h2>
  {_df_to_html_table(merged_display)}
  {blocks_section}
</body>
</html>
"""


@st.cache_data(show_spinner="Bestand laden…")
def cached_load(path_str: str, mtime: float):
    return load_measurement(path_str)


@st.cache_data(show_spinner="Bestand laden…")
def cached_load_bytes(filename: str, data: bytes):
    return load_measurement_bytes(filename, data)


def _sync_selection_order(checked_names: list[str]) -> list[str]:
    order: list[str] = list(st.session_state.get("selection_order", []))
    order = [name for name in order if name in checked_names]
    for name in checked_names:
        if name not in order:
            order.append(name)
    st.session_state.selection_order = order
    return order


def _clear_upload_keys() -> None:
    for key in ("upload_file_1", "upload_file_2"):
        if key in st.session_state:
            del st.session_state[key]


def add_block_shading(fig, block_df: pd.DataFrame) -> None:
    colors = ["rgba(31,119,180,0.12)", "rgba(255,127,14,0.12)"]
    windows = _unique_windows(block_df)
    for i, rec in enumerate(windows.itertuples()):
        fig.add_vrect(
            x0=rec.stabiel_van,
            x1=rec.stabiel_tot,
            fillcolor=colors[i % 2],
            line_width=0,
            annotation_text=f"{rec.setpoint:g} °C",
            annotation_position="top left",
            annotation_font_size=10,
        )


def _running_on_cloud() -> bool:
    """True op Streamlit Community Cloud en Hugging Face Spaces."""
    if os.environ.get("SPACE_ID") or os.environ.get("SPACE_HOST"):
        return True
    return Path("/mount/src").is_dir()


with st.sidebar:
    st.header("Bestanden")
    on_spaces = _running_on_cloud()
    source_mode = st.radio(
        "Bron",
        options=["Uploaden"] if on_spaces else ["Uploaden", "Lokale map"],
        index=0,
        help="Op Streamlit Cloud: bestanden uploaden. Lokaal kun je ook een map kiezen.",
    )

    files: list[Path] = []
    folder_path = Path(default_data_folder())

    if st.session_state.pop("reset_selection", False):
        st.session_state["selection_order"] = []
        _clear_upload_keys()
        for path in list_data_files(folder_path) if folder_path.is_dir() else []:
            st.session_state[f"file_{path.name}"] = False

    if st.button("Opnieuw beginnen", help="Wis uploads/selectie en start opnieuw."):
        st.session_state["reset_selection"] = True
        st.rerun()

    name1 = ""
    name2 = ""
    data1: bytes | None = None
    data2: bytes | None = None
    selected_paths: list[Path] = []

    if source_mode == "Uploaden":
        st.caption("Upload eerst bestand 1 (referentie), kies een tijdspanne, daarna bestand 2 (PT100).")
        up1 = st.file_uploader(
            "Bestand 1 — referentie",
            type=["dbf", "csv", "txt", "dat"],
            key="upload_file_1",
        )
        up2 = st.file_uploader(
            "Bestand 2 — PT100-scan",
            type=["csv", "txt", "dat", "dbf"],
            key="upload_file_2",
        )
        if up1 is not None:
            name1 = up1.name
            data1 = up1.getvalue()
        if up2 is not None:
            name2 = up2.name
            data2 = up2.getvalue()
    else:
        if "data_folder" not in st.session_state:
            st.session_state["data_folder"] = default_data_folder()
        folder = st.text_input(
            "Datamap",
            key="data_folder",
            help="Map met de meetbestanden. Met Bladeren kies je die map op deze computer.",
        )
        st.button(
            "Bladeren",
            on_click=_choose_data_folder,
            help="Open een venster om de map te kiezen.",
        )
        if st.session_state.get("data_folder_error"):
            st.error(st.session_state["data_folder_error"])
        folder_path = Path(folder)
        if not folder_path.is_dir():
            st.error("Deze map bestaat niet.")
        else:
            files = list_data_files(folder_path)
            if not files:
                st.warning("Geen .dbf, .csv of .txt bestanden gevonden.")

        st.caption("Vink eerst de referentie aan, kies een tijdspanne, daarna de PT100-scan.")
        checked: list[str] = []
        for path in files:
            if st.checkbox(path.name, key=f"file_{path.name}"):
                checked.append(path.name)
        selection_order = _sync_selection_order(checked)
        selected_paths = [folder_path / name for name in selection_order]

    st.divider()
    st.header("Koppeling")
    offset_s = st.number_input(
        "Tijdoffset bestand 2 [s]",
        value=0.0,
        step=1.0,
        help="Tel deze offset op bij de tijden van de PT100-scan als de klokken verschillen.",
    )

    st.divider()
    st.header("Stabiele blokken")
    temp_step = st.number_input(
        "Temperatuurstap setpoint [°C]",
        value=1.0,
        min_value=0.1,
        step=0.5,
        help="Referentietemperaturen worden aan het dichtstbijzijnde veelvoud van deze stap gekoppeld.",
    )
    temp_tol = st.number_input(
        "Temperatuurtolerantie [°C]",
        value=0.5,
        min_value=0.05,
        step=0.05,
        help="Alleen metingen binnen deze afstand van een setpoint horen bij dat blok.",
    )
    settle_min = st.number_input(
        "Inregeltijd overslaan [min]",
        value=5.0,
        min_value=0.0,
        step=1.0,
        help="Het stabiele venster begint pas na deze inregeltijd.",
    )
    end_margin = st.number_input(
        "Einde vóór volgende setpoint [min]",
        value=2.0,
        min_value=0.0,
        step=0.5,
        help="Het stabiele venster eindigt uiterlijk zoveel minuten vóór het volgende setpoint "
        "(compensatie voor niet-synchrone klokken).",
    )
    min_stable = st.number_input("Minimale stabiele duur [min]", value=5.0, min_value=1.0, step=1.0)
    diff_tol = st.number_input(
        "Tolerantie verschil [°C]",
        value=0.05,
        min_value=0.0,
        step=0.01,
        format="%.2f",
        help="In de overzichtstabel worden verschilwaarden gemarkeerd waarvan de absolute waarde groter is dan deze tolerantie.",
    )


if source_mode == "Uploaden":
    if data1 is None:
        st.info("Upload in de zijbalk het referentiebestand (DBF) om te beginnen.")
        st.stop()
    try:
        kind1, df1_all, ref_channels, ref_labels = cached_load_bytes(name1, data1)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{name1}: {exc}")
        st.stop()
    label1_name = name1
else:
    if not selected_paths:
        st.info("Selecteer in de zijbalk een datamap en vink het referentiebestand aan.")
        st.stop()
    if len(selected_paths) > 2:
        st.warning("Maximaal twee bestanden. Alleen de eerste twee worden gebruikt.")
        selected_paths = selected_paths[:2]
    path1 = selected_paths[0]
    try:
        kind1, df1_all, ref_channels, ref_labels = cached_load(str(path1), path1.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{path1.name}: {exc}")
        st.stop()
    label1_name = path1.name

if kind1 != "referentie":
    st.error(
        f"{label1_name} is geen referentie-temperatuurmeter (DBF met °C-kanalen). "
        "Kies dat bestand als bestand 1."
    )
    st.stop()
if df1_all.empty or not ref_channels:
    st.error(f"{label1_name} bevat geen bruikbare temperatuurmetingen.")
    st.stop()

st.subheader(f"1. Referentie — {label1_name}")
st.caption(
    f"{df1_all['tijd'].iloc[0]} → {df1_all['tijd'].iloc[-1]} · {len(df1_all)} punten · "
    f"kanalen: {', '.join(ref_labels.get(ch, ch) for ch in ref_channels)}"
)

ref_channel = ref_channels[0]
if len(ref_channels) > 1:
    ref_channel = st.selectbox(
        "Kanaal bestand 1",
        options=ref_channels,
        index=0,
        format_func=lambda ch: ref_labels.get(ch, ch),
        help="Standaard wordt het eerste kanaal getoond. Kies hier het andere kanaal.",
    )
ref_label = ref_labels.get(ref_channel, ref_channel)
df1 = df1_all[["tijd", ref_channel]].rename(columns={ref_channel: "t"})

t_min = _to_py_dt(df1["tijd"].iloc[0])
t_max = _to_py_dt(df1["tijd"].iloc[-1])
if t_min == t_max:
    st.warning("Dit bestand heeft maar één tijdstip; de tijdspanne kan niet worden begrensd.")
    range_start, range_end = t_min, t_max
else:
    range_start, range_end = st.slider(
        "Tijdspanne (op basis van bestand 1)",
        min_value=t_min,
        max_value=t_max,
        value=(t_min, t_max),
        format="YYYY-MM-DD HH:mm:ss",
        help="Alleen data binnen dit venster wordt gebruikt voor de vergelijking.",
    )

df1_win = filter_time_range(df1, range_start, range_end)
df1_preview = filter_time_range(df1_all, range_start, range_end)
if df1_win.empty:
    st.error("Geen punten van de referentie in de gekozen tijdspanne.")
    st.stop()

ref_vals = pd.to_numeric(df1_win["t"], errors="coerce").to_numpy(dtype=float)
ref_has_data = bool(np.isfinite(ref_vals).any())
if not ref_has_data:
    st.warning(f"{ref_label} bevat in dit bestand geen temperatuurmetingen.")

tab_preview, tab_table1 = st.tabs(["Voorbeeld referentie", "Tabel referentie"])
with tab_preview:
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=df1_preview["tijd"],
            y=df1_preview[ref_channel],
            name=ref_label,
            mode="lines",
        )
    )
    fig1.update_layout(
        height=480,
        yaxis_title="T [°C]",
        xaxis_title="Tijd",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig1, width="stretch")

with tab_table1:
    preview_cols = [("tijd", "Tijd"), (ref_channel, f"{ref_label} [°C]")]
    display1 = format_for_display(df1_preview, preview_cols)
    st.dataframe(
        display1,
        width="stretch",
        hide_index=True,
        height=420,
        column_config=_column_config(display1),
    )

if not ref_has_data:
    st.stop()

if source_mode == "Uploaden":
    if data2 is None:
        st.info("Upload nu de PT100-scan in de zijbalk om te vergelijken.")
        st.stop()
    try:
        kind2, df2, pt_channels, pt_labels = cached_load_bytes(name2, data2)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{name2}: {exc}")
        st.stop()
    label2_name = name2
else:
    if len(selected_paths) < 2:
        st.info("Vink nu de PT100-scan aan om te vergelijken binnen de gekozen tijdspanne.")
        st.stop()
    path2 = selected_paths[1]
    try:
        kind2, df2, pt_channels, pt_labels = cached_load(str(path2), path2.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{path2.name}: {exc}")
        st.stop()
    label2_name = path2.name

if kind2 != "pt100":
    st.error(
        f"{label2_name} is geen PT100-scan. "
        "Bestand 2 moet de Keysight-export zijn met maximaal 20 temperatuurkanalen."
    )
    st.stop()

usable = valid_channels(df2, pt_channels)
skipped = [ch for ch in pt_channels if ch not in usable]
if len(usable) > 20:
    st.warning("Meer dan 20 geldige PT100-kanalen; de eerste 20 worden gebruikt.")
    usable = usable[:20]

st.divider()
st.subheader(f"2. Vergelijking met {label2_name}")
st.caption(
    f"PT100-scan: {df2['tijd'].iloc[0]} → {df2['tijd'].iloc[-1]} · {len(df2)} scans"
)
if skipped:
    st.caption(
        "Overgeslagen (geen geldige meting, open of overrange): "
        + ", ".join(pt_labels.get(ch, ch) for ch in skipped)
    )
if not usable:
    st.error("Bestand 2 bevat geen geldige PT100-temperaturen.")
    st.stop()

selected_channels = st.multiselect(
    "PT100-kanalen in de vergelijking",
    options=usable,
    default=usable,
    format_func=lambda ch: pt_labels.get(ch, ch),
    help="Maximaal 20 kanalen. Open kanalen staan hier niet bij.",
)
if not selected_channels:
    st.warning("Kies minstens één PT100-kanaal.")
    st.stop()

df2_work = df2.copy()
df2_work["tijd"] = shift_datetime_series(df2_work["tijd"], float(offset_s))
df2_in_range = filter_time_range(df2_work, range_start, range_end)
if df2_in_range.empty:
    st.error(
        "Geen PT100-scans in de gekozen tijdspanne. "
        "Pas de tijdspanne of de tijdoffset aan."
    )
    st.stop()

try:
    merged = merge_measurements(
        df1_win,
        df2_in_range,
        selected_channels,
        pt100_offset_seconds=0.0,
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"Koppelen van de metingen is mislukt: {exc}")
    st.stop()

if merged.empty:
    st.error(
        "Geen overlappende tijdstippen. Pas de tijdspanne of de tijdoffset aan."
    )
    st.stop()

blocks = detect_stable_blocks(
    merged,
    selected_channels,
    temp_step=float(temp_step),
    temp_tolerance=float(temp_tol),
    settle_minutes=float(settle_min),
    min_stable_minutes=float(min_stable),
    end_margin_minutes=float(end_margin),
)
n_windows = 0 if blocks.empty else int(_unique_windows(blocks).shape[0])

info1, info2, info3, info4 = st.columns(4)
info1.metric("Referentie", ref_label)
info2.metric("PT100-kanalen", f"{len(selected_channels)}")
info3.metric("Gekoppelde scans", f"{len(merged)}")
info4.metric("Stabiele blokken", f"{n_windows}")
st.caption(
    f"Tijdspanne: {range_start} → {range_end} · "
    f"PT100-scans in venster: {len(df2_in_range)} · "
    f"ΔT = PT100 − {ref_label}"
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=merged["tijd"],
        y=merged["t_ref"],
        name=f"Referentie {ref_label}",
        mode="lines",
        line=dict(color="#111111", width=2.5),
    )
)
for channel in selected_channels:
    fig.add_trace(
        go.Scatter(
            x=merged["tijd"],
            y=merged[channel],
            name=pt_labels.get(channel, channel),
            mode="lines",
        )
    )
if not blocks.empty:
    add_block_shading(fig, blocks)
fig.update_layout(
    height=680,
    yaxis_title="T [°C]",
    xaxis_title="Tijd",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    hovermode="x unified",
    margin=dict(t=80, b=40),
)

fig_diff = go.Figure()
for channel in selected_channels:
    fig_diff.add_trace(
        go.Scatter(
            x=merged["tijd"],
            y=merged[f"d_{channel}"],
            name=f"ΔT {pt_labels.get(channel, channel)}",
            mode="lines",
        )
    )
fig_diff.add_hline(y=0, line_width=1, line_color="gray")
fig_diff.update_layout(
    height=420,
    yaxis_title="ΔT [°C]",
    xaxis_title="Tijd",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    hovermode="x unified",
    margin=dict(t=80, b=40),
)

display_df = format_for_display(merged, coupled_columns(selected_channels, pt_labels))
block_display = format_for_display(blocks, BLOCK_COLUMNS) if not blocks.empty else pd.DataFrame()
wide_display = wide_delta_table(blocks, pt_labels) if not blocks.empty else pd.DataFrame()

fig_blocks: go.Figure | None = None
if not blocks.empty:
    fig_blocks = go.Figure()
    for index, channel in enumerate(selected_channels):
        part = blocks[blocks["kanaal"].astype(str) == str(channel)]
        if part.empty:
            continue
        color = _channel_color(index)
        fig_blocks.add_trace(
            go.Scatter(
                x=part["setpoint"],
                y=part["d_t"],
                mode="markers+lines",
                name=pt_labels.get(channel, channel),
                text=part["richting"],
                hovertemplate="%{fullData.name}<br>%{x:g} °C<br>ΔT %{y:.3f} °C<extra></extra>",
                line=dict(color=color, width=1.6),
                marker=dict(color=color, size=8),
            )
        )
    fig_blocks.add_hline(y=0, line_width=1, line_color="gray")
    fig_blocks.update_layout(
        xaxis_title="Temperatuursetpoint [°C]",
        yaxis_title="Gem. ΔT (PT100 − referentie) [°C]",
        height=420,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=80, b=40),
    )

report_html = build_report_html(
    path1_name=label1_name,
    path2_name=label2_name,
    ref_label=ref_label,
    range_start=range_start,
    range_end=range_end,
    settle_min=float(settle_min),
    end_margin=float(end_margin),
    fig_main=fig,
    fig_diff=fig_diff,
    fig_blocks=fig_blocks,
    merged_display=display_df,
    block_display=block_display,
    wide_display=wide_display,
    diff_tol=float(diff_tol),
)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
st.download_button(
    "Download grafieken & tabellen (HTML)",
    data=report_html.encode("utf-8"),
    file_name=f"pt100_kalibratie_{stamp}.html",
    mime="text/html",
    help="Eén HTML-bestand met interactieve grafieken en alle tabellen.",
)

tab_grafiek, tab_tabel, tab_blokken = st.tabs(
    ["Temperatuur", "Gekoppelde tabel", "Stabiele verschillen"]
)

with tab_grafiek:
    st.plotly_chart(fig, width="stretch")
    st.subheader("Verschillen (PT100 − referentie)")
    st.plotly_chart(fig_diff, width="stretch")

with tab_tabel:
    st.markdown(
        "Elke **PT100-scan** is gekoppeld aan de dichtstbijzijnde **referentiemeting** "
        "binnen de gekozen tijdspanne."
    )
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        height=520,
        column_config=_column_config(display_df),
    )
    st.download_button(
        "Download gekoppelde tabel (CSV)",
        data=merged.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name="vergelijking_pt100.csv",
        mime="text/csv",
    )

with tab_blokken:
    st.markdown(
        "Verschil **ΔT = PT100 − referentie** over het **stabiele** deel van elk temperatuursetpoint:\n"
        f"- start na **{settle_min:g} min** inregeltijd\n"
        f"- eindigt uiterlijk **{end_margin:g} min** vóór het volgende setpoint "
        "(klokken niet synchroon)"
    )
    if blocks.empty:
        st.warning(
            "Geen stabiele blokken gevonden. Verlaag de inregeltijd of de eindmarge, "
            "of vergroot de temperatuurtolerantie of de temperatuurstap."
        )
    else:
        st.subheader("Overzicht ΔT per kanaal")
        st.caption(
            f"Kleur volgens |ΔT| ten opzichte van {float(diff_tol):g} °C: "
            "lichtgroen onder 0,25×, geel van 0,25× tot 0,5×, oranje van 0,5× tot 1×, "
            "rood van 1× tot 2×, donkerrood daarboven."
        )
        st.dataframe(
            _mark_differences(wide_display, float(diff_tol)),
            width="stretch",
            hide_index=True,
            column_config=_column_config(wide_display),
        )
        st.download_button(
            "Download overzicht ΔT per kanaal (CSV)",
            data=wide_display.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name="overzicht_delta_t.csv",
            mime="text/csv",
            key="download_overview_dt",
        )

        st.subheader("Details per blok")
        st.dataframe(
            block_display,
            width="stretch",
            hide_index=True,
            column_config=_column_config(block_display),
        )

        if fig_blocks is not None:
            st.caption("Beweeg over een lijn om dat kanaal uit te lichten.")
            _plotly_hover_chart(fig_blocks, div_id="delta_setpoints_app", height=500)
