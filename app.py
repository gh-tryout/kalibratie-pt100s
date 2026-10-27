"""Streamlit-app: vergelijk PT100-kanalen met een referentie-temperatuurmeter."""

from __future__ import annotations

import html
import importlib
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Streamlit Cloud houdt een eerder geladen module soms in het geheugen terwijl
# app.py al wel de nieuwe versie is. Opnieuw laden voorkomt dan
# "cannot import name ... from pt100_compare".
try:
    import pt100_compare as _pt100_compare

    _pt100_compare = importlib.reload(_pt100_compare)
    BLOCK_COLUMNS = _pt100_compare.BLOCK_COLUMNS
    classify_file = _pt100_compare.classify_file
    as_single_channel_measurement = _pt100_compare.as_single_channel_measurement
    combine_pt100_measurements = _pt100_compare.combine_pt100_measurements
    coupled_columns = _pt100_compare.coupled_columns
    detect_stable_blocks = _pt100_compare.detect_stable_blocks
    filter_time_range = _pt100_compare.filter_time_range
    format_for_display = _pt100_compare.format_for_display
    list_data_files = _pt100_compare.list_data_files
    load_measurement = _pt100_compare.load_measurement
    load_measurement_bytes = _pt100_compare.load_measurement_bytes
    merge_measurements = _pt100_compare.merge_measurements
    shift_datetime_series = _pt100_compare.shift_datetime_series
    valid_channels = _pt100_compare.valid_channels
    wide_delta_table = _pt100_compare.wide_delta_table
    parse_uncertainty_formula = _pt100_compare.parse_uncertainty_formula
    evaluate_uncertainty = _pt100_compare.evaluate_uncertainty
    uncertainty_formula_text = _pt100_compare.uncertainty_formula_text
    build_assessment_tables = _pt100_compare.build_assessment_tables
    assessment_measurement_base = _pt100_compare.assessment_measurement_base
    formula_uncertainty_by_row = _pt100_compare.formula_uncertainty_by_row
    filter_uncertainty_overrides = _pt100_compare.filter_uncertainty_overrides
    apply_uncertainty_overrides = _pt100_compare.apply_uncertainty_overrides
    changed_uncertainty_rows = _pt100_compare.changed_uncertainty_rows
    assemble_assessment = _pt100_compare.assemble_assessment
    judgement_mark = _pt100_compare.judgement_mark
    build_assessment_report_html = _pt100_compare.build_assessment_report_html
    UncertaintyFormulaError = _pt100_compare.UncertaintyFormulaError
except Exception:
    st.set_page_config(page_title="PT100-kalibratie", layout="wide")
    st.error("pt100_compare.py kon niet geladen worden.")
    st.code(traceback.format_exc())
    st.stop()

st.set_page_config(
    page_title="PT100-kalibratie",
    page_icon="🌡️",
    layout="wide",
)

st.title("PT100-kalibratie vs referentie")
st.caption(
    "Upload eerst de referentie-temperatuurmeter (bestand 1, DBF), kies een tijdspanne, "
    "daarna een of meer PT100-scans of een tweede referentie (DBF, één kanaal). "
    "Het tijdblok van elk setpoint stel je in met twee tijden vóór het volgende setpoint, "
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


def _replace_single_spikes(times, values, limit_c: float = 5.0) -> tuple[np.ndarray, int]:
    """Vervang één losse uitschieter door lineaire interpolatie van de buren.

    Een punt telt als spike als het meer dan ``limit_c`` °C van beide buren afwijkt
    en die buren zelf hoogstens ``limit_c`` °C uit elkaar liggen.
    """
    y = np.asarray(values, dtype=float).copy()
    n = len(y)
    if n < 3:
        return y, 0
    t = pd.to_datetime(pd.Series(times)).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    spike = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        left, mid, right = y[i - 1], y[i], y[i + 1]
        if not (np.isfinite(left) and np.isfinite(mid) and np.isfinite(right)):
            continue
        if abs(left - right) > limit_c:
            continue
        if abs(mid - left) > limit_c and abs(mid - right) > limit_c:
            spike[i] = True
    isolated = spike.copy()
    if n > 2:
        isolated[1:-1] &= ~spike[:-2] & ~spike[2:]
    count = 0
    for i in np.flatnonzero(isolated):
        t0 = t[i - 1]
        t1 = t[i + 1]
        if t1 == t0:
            y[i] = 0.5 * (y[i - 1] + y[i + 1])
        else:
            frac = (t[i] - t0) / (t1 - t0)
            y[i] = y[i - 1] + frac * (y[i + 1] - y[i - 1])
        count += 1
    return y, int(count)


def _to_py_dt(value) -> datetime:
    ts = pd.Timestamp(value)
    return ts.to_pydatetime().replace(tzinfo=None)


def _df_to_html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>Geen gegevens</em></p>"
    return df.to_html(index=False, border=0, classes="data", escape=True)


def _difference_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).startswith("ΔT")]


def _round_differences_for_html(df: pd.DataFrame) -> pd.DataFrame:
    """Rond kolommen met ΔT af op 0,01 °C voor het HTML-rapport."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if "ΔT" not in str(col):
            continue
        numbers = pd.to_numeric(out[col], errors="coerce")
        out[col] = ["" if pd.isna(value) else f"{float(value):.2f}" for value in numbers]
    return out


def _figure_with_rounded_differences(fig: go.Figure) -> go.Figure:
    """Kopie van een ΔT-grafiek met y-waarden op 0,01 °C."""
    clone = go.Figure(fig)
    for trace in clone.data:
        y = getattr(trace, "y", None)
        if y is None:
            continue
        values = pd.to_numeric(pd.Series(list(y)), errors="coerce").to_numpy(dtype=float)
        trace.y = np.round(values, 2)
        template = getattr(trace, "hovertemplate", None)
        if isinstance(template, str):
            trace.hovertemplate = template.replace(".3f", ".2f")
    return clone


def _csv_bytes(df: pd.DataFrame, difference_columns: list[str]) -> bytes:
    """CSV met punt als decimaalteken; ΔT afgerond op 0,01 °C."""
    out = df.copy()
    for col in difference_columns:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    return out.to_csv(index=False, sep=";", decimal=".").encode("utf-8-sig")


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


def _assessment_column_config(df: pd.DataFrame, value_format: str = "%.4f") -> dict:
    config: dict = {}
    for col in df.columns:
        if col == "n" or (col == "T set [°C]" and pd.api.types.is_integer_dtype(df[col])):
            config[col] = st.column_config.NumberColumn(col, format="%d", width="small")
        elif col in ("T set [°C]", "T ref [°C]"):
            config[col] = st.column_config.NumberColumn(col, format="%.3f", width="small")
        elif pd.api.types.is_float_dtype(df[col]):
            config[col] = st.column_config.NumberColumn(col, format=value_format, width="small")
        else:
            config[col] = st.column_config.TextColumn(col, width="small")
    return config


def _format_remaining(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 5:
        return "enkele seconden"
    if seconds < 60:
        whole = int(round(seconds))
        return "enkele seconden" if whole < 5 else f"{whole} seconden"
    minutes = max(1, int(round(seconds / 60.0)))
    if minutes == 1:
        return "1 minuut"
    return f"{minutes} minuten"


def _style_judgement(df: pd.DataFrame, columns: list[str], tolerance: float):
    def paint(value) -> str:
        mark = judgement_mark(value, tolerance)
        if mark == "fail":
            return "background-color: #8B0000; color: #ffffff"
        if mark == "border":
            return "background-color: #f6b26b"
        return ""

    return df.style.map(paint, subset=columns)


def _clear_formula_overrides() -> None:
    """Callback: wist handmatige onzekerheden vóór de fragment-run."""
    st.session_state["formula_u_override"] = {}
    st.session_state["judgement_editor_generation"] = int(st.session_state.get("judgement_editor_generation", 0)) + 1


def _edited_rows_from_state(state) -> dict[int, dict]:
    if not isinstance(state, dict):
        return {}
    raw = state.get("edited_rows") or {}
    parsed: dict[int, dict] = {}
    for key, changes in raw.items():
        if isinstance(changes, dict):
            parsed[int(key)] = changes
    return parsed


def _overrides_with_editor(base, formula_per_row: np.ndarray, stored, edits: dict[int, dict]):
    """Handmatige celwaarden bovenop de formule. Cellen gelijk aan de formule vallen weg."""
    overrides = filter_uncertainty_overrides(stored, base.row_ids, base.channels)
    dropped: dict[int, list[str]] = {}
    for row_index, changes in edits.items():
        if row_index < 0 or row_index >= len(base.row_ids):
            continue
        row_id = base.row_ids[row_index]
        formula_value = float(formula_per_row[row_index])
        formula_known = bool(np.isfinite(formula_value))
        for channel, column in zip(base.channels, base.formula_columns):
            if column not in changes:
                continue
            cell_id = f"{row_id}|{channel}"
            raw = changes[column]
            number = None
            try:
                if raw is not None and not (isinstance(raw, str) and not str(raw).strip()) and not pd.isna(raw):
                    number = abs(float(raw))
            except (TypeError, ValueError):
                number = None
            if number is None or not np.isfinite(number):
                overrides.pop(cell_id, None)
                dropped.setdefault(row_index, []).append(column)
                continue
            if formula_known and abs(number - formula_value) <= 5e-5:
                overrides.pop(cell_id, None)
                dropped.setdefault(row_index, []).append(column)
                continue
            overrides[cell_id] = number
    return overrides, dropped


def _drop_editor_cells(editor_key: str, dropped: dict[int, list[str]]) -> None:
    if not dropped or editor_key not in st.session_state:
        return
    state = st.session_state.get(editor_key)
    if not isinstance(state, dict):
        return
    edited = {
        int(key): dict(changes)
        for key, changes in (state.get("edited_rows") or {}).items()
        if isinstance(changes, dict)
    }
    changed = False
    for row_index, columns in dropped.items():
        row = edited.get(int(row_index))
        if not row:
            continue
        for column in columns:
            if column in row:
                row.pop(column, None)
                changed = True
        if not row:
            edited.pop(int(row_index), None)
    if changed:
        st.session_state[editor_key] = {
            "edited_rows": edited,
            "added_rows": list(state.get("added_rows") or []),
            "deleted_rows": list(state.get("deleted_rows") or []),
        }


@st.fragment
def _show_beoordeling() -> None:
    """Alleen dit blok draait opnieuw bij een formule of een cel in de beoordeling."""
    view = st.session_state.get("beoordeling_view") or {}
    base = st.session_state.get("assessment_base")
    token = st.session_state.get("assessment_token")

    st.subheader("Meetonzekerheid van de kalibratie")
    st.caption(
        "Schrijf de onzekerheid als functie van de temperatuur T, in °C. "
        "Voorbeelden: a*T + b en a*e^(-b*T). "
        "T is de gemiddelde referentietemperatuur van het tijdblok. "
        "e is het grondtal van de natuurlijke logaritme."
    )
    formula_text = st.text_input(
        "Formule",
        value="a*T + b",
        key="uncertainty_formula",
        help="T is de temperatuur. Coëfficiënten zoals a en b vul je hieronder in. "
        "Ook geldig: aT+b, a*exp(-b*T) en 0.02*T+0.05.",
    )
    parsed_formula = parse_uncertainty_formula(formula_text)
    coefficients: dict[str, float] = {}
    if not parsed_formula.ok:
        st.error(parsed_formula.error)
    elif parsed_formula.parameters:
        coef_cols = st.columns(min(len(parsed_formula.parameters), 4))
        for index, name in enumerate(parsed_formula.parameters):
            coefficients[name] = coef_cols[index % len(coef_cols)].number_input(
                name,
                value=0.0,
                step=0.001,
                format="%.6f",
                key=f"unc_coef_{name}",
                help="Coëfficiënt in de formule.",
            )
    formula_shown = uncertainty_formula_text(parsed_formula, coefficients) if parsed_formula.ok else ""
    if formula_shown:
        st.caption(formula_shown)

    if not view.get("time_block_valid", False):
        st.error("De berekening is ongeldig. Zet de start verder vóór het einde van het setpoint dan het einde.")
        return
    if base is None or not base.ready:
        st.warning("Geen stabiele blokken gevonden. Zonder tijdblokken is er niets te beoordelen.")
        return

    generation = int(st.session_state.get("judgement_editor_generation", 0))
    if st.session_state.get("assessment_editor_token") != token:
        generation += 1
        st.session_state["judgement_editor_generation"] = generation
        st.session_state["assessment_editor_token"] = token
    editor_key = f"judgement_u_editor_{generation}"

    formula_ready = parsed_formula.ok
    was_negative = False
    not_finite = False
    try:
        formula_per_row, was_negative, not_finite = formula_uncertainty_by_row(
            base, (lambda temperature: evaluate_uncertainty(parsed_formula, temperature, coefficients)) if formula_ready else None
        )
    except UncertaintyFormulaError as exc:
        formula_ready = False
        formula_per_row = np.full(len(base.row_ids), np.nan)
        st.error(str(exc))

    stored_overrides = dict(st.session_state.get("formula_u_override") or {})
    edits = _edited_rows_from_state(st.session_state.get(editor_key) if editor_key in st.session_state else None)
    overrides, dropped = _overrides_with_editor(base, formula_per_row, stored_overrides, edits)
    _drop_editor_cells(editor_key, dropped)
    st.session_state["formula_u_override"] = overrides

    matrix = apply_uncertainty_overrides(base, formula_per_row, overrides)
    cache = st.session_state.get("assessment_assembly")
    previous_u = None
    previous_tables = None
    if isinstance(cache, dict) and cache.get("token") == token:
        previous_u = cache.get("u_cal")
        previous_tables = cache.get("tables")
    assessment = assemble_assessment(
        base,
        matrix,
        formula_per_row,
        was_negative,
        not_finite,
        previous_tables,
        changed_uncertainty_rows(previous_u, matrix),
    )
    st.session_state["assessment_assembly"] = {"token": token, "u_cal": matrix, "tables": assessment}

    if assessment.formula_was_negative:
        st.warning("Een negatieve uitkomst van de formule is als absolute waarde gebruikt.")
    if assessment.formula_not_finite:
        st.warning("De formule geeft voor minstens één temperatuur geen eindig getal.")

    st.subheader("Gemeten verschil ΔT")
    st.caption("Zelfde overzicht als ΔT per kanaal, zonder de tijden. ΔT = PT100 − referentie.")
    st.dataframe(
        assessment.delta,
        width="stretch",
        hide_index=True,
        column_config=_assessment_column_config(assessment.delta),
    )

    st.subheader("Standaardonzekerheid van de meting")
    st.caption(
        "Per tijdblok: de standaarddeviatie van ΔT gedeeld door de wortel van het aantal waarnemingen van dat kanaal."
    )
    st.dataframe(
        assessment.u_measurement,
        width="stretch",
        hide_index=True,
        column_config=_assessment_column_config(assessment.u_measurement),
    )

    if not formula_ready:
        return

    diff_tol = float(view.get("diff_tol", 0.0))
    st.subheader("Totale meetonzekerheid")
    st.caption(
        "De onzekerheid uit de formule, of de waarde die je in de onderste tabel met de hand invult, "
        "opgeteld bij de standaardonzekerheid van de meting."
    )
    st.dataframe(
        assessment.u_total,
        width="stretch",
        hide_index=True,
        column_config=_assessment_column_config(assessment.u_total),
    )

    st.subheader("Beoordeeld verschil")
    st.caption(
        "Het gemeten verschil, met de totale meetonzekerheid opgeteld in de richting van de afwijking. "
        "Een positief verschil wordt hoger, een negatief verschil lager. "
        f"Donkerrood: groter dan de tolerantie van {diff_tol:g} °C. "
        "Oranje: grensgeval, vanaf de helft van de tolerantie tot en met de tolerantie. "
        "Elke kolom u formule staat bij de meetwaarde ernaast en komt uit de formule. "
        "Pas een cel aan om die meetwaarde een eigen onzekerheid te geven."
    )
    if overrides:
        st.button(
            "Zet onzekerheden terug naar de formule",
            key="reset_formula_u",
            on_click=_clear_formula_overrides,
        )
    judged_config = _assessment_column_config(assessment.judged)
    for column in assessment.formula_columns:
        judged_config[column] = st.column_config.NumberColumn(
            column,
            format="%.4f",
            min_value=0.0,
            step=0.0001,
            help="Onzekerheid uit de formule bij deze meetwaarde. Typ een ander getal om hem met de hand te vullen.",
        )
    st.data_editor(
        _style_judgement(assessment.judged, assessment.judged_columns, diff_tol),
        width="stretch",
        hide_index=True,
        column_config=judged_config,
        disabled=[column for column in assessment.judged.columns if column not in assessment.formula_columns],
        key=editor_key,
    )

    report_assessment_html = build_assessment_report_html(
        path1_name=str(view.get("label1_name", "")),
        path2_name=str(view.get("label2_name", "")),
        ref_label=str(view.get("ref_label", "")),
        range_start=view.get("range_start"),
        range_end=view.get("range_end"),
        laborant=str(view.get("laborant", "")),
        referentie=str(view.get("referentie", "")),
        calibrator=str(view.get("calibrator", "")),
        meetmiddel=str(view.get("meetmiddel", "")),
        formula_text=formula_shown,
        diff_tol=diff_tol,
        tables=assessment,
        max_rate=float(view.get("max_rate", 0.0)),
        min_stable=float(view.get("min_stable", 0.0)),
        calc_start=float(view.get("calc_start", 0.0)),
        calc_end=float(view.get("calc_end", 0.0)),
    )
    report_bytes = report_assessment_html.encode("utf-8-sig")
    stamp = str(view.get("stamp") or datetime.now().strftime("%Y%m%d_%H%M%S"))
    html_report_col, word_report_col = st.columns(2)
    with html_report_col:
        st.download_button(
            "Download kalibratierapport (HTML)",
            data=report_bytes,
            file_name=f"pt100_kalibratierapport_{stamp}.html",
            mime="text/html",
            key="download_assessment_html",
            help="HTML met de beoordeling. In Word open je dit via Bestand → Openen.",
        )
    with word_report_col:
        st.download_button(
            "Download kalibratierapport (Word)",
            data=report_bytes,
            file_name=f"pt100_kalibratierapport_{stamp}.doc",
            mime="application/msword",
            key="download_assessment_doc",
            help="Hetzelfde rapport. Dit bestand opent direct in Word.",
        )


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
    max_rate: float,
    min_stable: float,
    calc_start: float,
    calc_end: float,
    fig_main: go.Figure,
    fig_diff: go.Figure,
    fig_blocks: go.Figure | None,
    block_display: pd.DataFrame,
    wide_display: pd.DataFrame,
    diff_tol: float,
    laborant: str,
    referentie: str,
    calibrator: str,
    meetmiddel: str,
) -> str:
    chart_main = fig_main.to_html(full_html=False, include_plotlyjs="cdn")
    chart_diff = _figure_with_rounded_differences(fig_diff).to_html(full_html=False, include_plotlyjs=False)
    chart_blocks = ""
    if fig_blocks is not None:
        chart_blocks = _figure_with_rounded_differences(fig_blocks).to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="delta_setpoints",
        ) + _hover_bind_script("delta_setpoints")
    wide_html = _round_differences_for_html(wide_display)
    block_html = _round_differences_for_html(block_display)
    if not wide_html.empty:
        blocks_section = f"""
        <h2>Gemeten verschillen</h2>
        <p>Het einde van het setpoint is het moment dat de referentie sneller dan {max_rate:g} K/min
        verandert, na minstens {min_stable:g} min stabiel. Het gemiddelde gebruikt
        {calc_start:g} tot {calc_end:g} min vóór dat einde.
        ΔT = PT100 − referentie, afgerond op 0,01 °C.</p>
        <h3>Overzicht ΔT per kanaal</h3>
        <p>Kleur volgens |ΔT| t.o.v. de tolerantie van {diff_tol:g} °C:
        lichtgroen &lt; 0,25×, geel 0,25–0,5×, oranje 0,5–1×, rood 1–2×, donkerrood &gt; 2×.</p>
        {_mark_differences(wide_html, diff_tol).to_html(index=False, border=0, classes="data")}
        <h3>Details per blok</h3>
        {_df_to_html_table(block_html)}
        <h3>ΔT per setpoint</h3>
        {chart_blocks}
        """
    else:
        blocks_section = "<h2>Gemeten verschillen</h2><p><em>Geen stabiele blokken gevonden.</em></p>"

    def meta_line(label: str, value: str) -> str:
        text = value.strip() or "—"
        return f"<div><strong>{html.escape(label)}:</strong> {html.escape(text)}</div>"

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
    {meta_line("Laborant", laborant)}
    {meta_line("Referentie", referentie)}
    {meta_line("Calibrator", calibrator)}
    {meta_line("Gekalibreerd meetmiddel", meetmiddel)}
    <div><strong>Bestand 1 (referentie):</strong> {html.escape(path1_name)} · kanaal {html.escape(ref_label)}</div>
    <div><strong>PT100:</strong> {html.escape(path2_name)}</div>
    <div><strong>Tijdspanne:</strong> {html.escape(str(range_start))} → {html.escape(str(range_end))}</div>
    <div><strong>Gegenereerd:</strong> {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</div>
  </div>
  <h2>Temperatuur</h2>
  {chart_main}
  <h2>Verschillen (PT100 − referentie)</h2>
  {chart_diff}
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


def _measurement_channel_id(source_name: str, used: set[str]) -> str:
    stem = Path(source_name).stem.strip() or "meting"
    candidate = stem
    n = 2
    while candidate in used:
        candidate = f"{stem} ({n})"
        n += 1
    used.add(candidate)
    return candidate


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
    data1: bytes | None = None
    selected_paths: list[Path] = []
    pt100_uploads: list[tuple[str, bytes]] = []
    ref_paths: list[Path] = []
    pt100_paths: list[Path] = []
    ref_error = ""

    if source_mode == "Uploaden":
        st.caption(
            "Upload eerst de referentie, kies een tijdspanne, daarna PT100-scans (CSV) "
            "of een tweede referentie (DBF)."
        )
        up1 = st.file_uploader(
            "Referentie",
            type=["dbf", "csv", "txt", "dat"],
            key="upload_file_1",
        )
        up2 = st.file_uploader(
            "Metingen",
            type=["csv", "txt", "dat", "dbf"],
            key="upload_file_2",
            accept_multiple_files=True,
            help="CSV: PT100-scan. Hetzelfde kanaalnummer wordt één kanaal. "
            "DBF: tweede referentie, verder behandeld als één meetkanaal.",
        )
        if up1 is not None:
            name1 = up1.name
            data1 = up1.getvalue()
        if up2:
            pt100_uploads = [(item.name, item.getvalue()) for item in up2]
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

        st.caption(
            "Vink eerst de referentie (DBF) aan. Daarna een of meer PT100-scans, "
            "of een extra DBF als tweede referentie. Die telt als één meetkanaal. "
            "Hetzelfde kanaalnummer uit CSV-bestanden wordt één kanaal."
        )
        checked: list[str] = []
        for path in files:
            if st.checkbox(path.name, key=f"file_{path.name}"):
                checked.append(path.name)
        selection_order = _sync_selection_order(checked)
        selected_paths = [folder_path / name for name in selection_order]
        unknown: list[str] = []
        for path in selected_paths:
            try:
                kind = classify_file(path)
            except Exception as exc:  # noqa: BLE001
                st.error(f"{path.name}: {exc}")
                continue
            if kind == "referentie":
                if not ref_paths:
                    ref_paths.append(path)
                else:
                    pt100_paths.append(path)
            elif kind == "pt100":
                pt100_paths.append(path)
            else:
                unknown.append(path.name)
        if unknown:
            st.warning("Niet herkend als referentie of meting: " + ", ".join(unknown))

    st.divider()
    st.header("Koppeling")
    offset_s = st.number_input(
        "Tijdoffset PT100 [s]",
        value=0.0,
        step=1.0,
        help="Tel deze offset op bij de tijden van de meting (PT100-scan of tweede referentie) als de klokken verschillen.",
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
    max_rate = st.number_input(
        "Temperatuurverandering [K/min]",
        value=0.1,
        min_value=0.01,
        step=0.01,
        format="%.2f",
        help="Bepaalt, samen met de minimale duur, het tijdstip waarop het systeem naar het volgende setpoint gaat. "
        "Default 0,1 K/min over de laatste minuut.",
    )
    min_stable = st.number_input(
        "Minimale stabiele duur [min]",
        value=5.0,
        min_value=1.0,
        step=1.0,
        help="Het setpoint-einde telt pas nadat de temperatuur minstens zo lang stabiel is geweest.",
    )
    calc_start_before = st.number_input(
        "Berekening start vóór einde setpoint [min]",
        value=15.0,
        min_value=0.0,
        step=1.0,
        help="Het gemiddelde begint zoveel minuten vóór het einde van het setpoint.",
    )
    calc_end_before = st.number_input(
        "Berekening einde vóór einde setpoint [min]",
        value=2.0,
        min_value=0.0,
        step=0.5,
        help="Het gemiddelde stopt zoveel minuten vóór het einde van het setpoint.",
    )
    time_block_valid = float(calc_start_before) > float(calc_end_before)
    if not time_block_valid:
        st.error("De start van de berekening moet verder vóór het einde liggen dan het einde van de berekening.")
    diff_tol = st.number_input(
        "Tolerantie verschil [°C]",
        value=0.05,
        min_value=0.0,
        step=0.01,
        format="%.2f",
        help="In de overzichtstabel worden verschilwaarden gemarkeerd waarvan de absolute waarde groter is dan deze tolerantie.",
    )

    st.divider()
    st.header("Rapport")
    laborant = st.text_input("Laborant", key="rapport_laborant", help="Optioneel. Komt in het HTML-rapport.")
    referentie = st.text_input(
        "Referentie",
        key="rapport_referentie",
        help="Optioneel. Gebruikte referentie-apparatuur.",
    )
    calibrator = st.text_input(
        "Calibrator",
        key="rapport_calibrator",
        help="Optioneel. Gebruikte calibrator.",
    )
    meetmiddel = st.text_input(
        "Gekalibreerd meetmiddel",
        key="rapport_meetmiddel",
        help="Optioneel. Het meetmiddel dat gekalibreerd wordt.",
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
    if ref_error:
        st.error(ref_error)
        st.stop()
    if not ref_paths:
        st.info("Selecteer in de zijbalk een datamap en vink het referentiebestand aan.")
        st.stop()
    path1 = ref_paths[0]
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
    remove_spikes = st.toggle(
        "Spikes verwijderen",
        key="remove_ref_spikes",
        help="Vervang een enkele meetwaarde die meer dan 5 °C van beide buren afwijkt "
        "door interpolatie. De buren moeten zelf binnen 5 °C van elkaar liggen.",
    )
    n_spikes = 0
    if remove_spikes:
        cleaned, n_spikes = _replace_single_spikes(df1["tijd"], df1["t"])
        df1 = df1.copy()
        df1["t"] = cleaned
        df1_win = filter_time_range(df1, range_start, range_end)
        st.caption(f"{n_spikes} losse uitschieters vervangen door interpolatie tussen de buren.")
    preview_frame = df1_win.rename(columns={"t": ref_channel})
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=preview_frame["tijd"],
            y=preview_frame[ref_channel],
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
    display1 = format_for_display(preview_frame, preview_cols)
    st.dataframe(
        display1,
        width="stretch",
        hide_index=True,
        height=420,
        column_config=_column_config(display1),
    )

if not ref_has_data:
    st.stop()

pt100_parts: list[tuple[pd.DataFrame, list[str], dict[str, str]]] = []
pt100_names: list[str] = []
used_channel_ids: set[str] = set()
if source_mode == "Uploaden":
    if not pt100_uploads:
        st.info("Upload nu een PT100-scan (CSV) of een tweede referentie (DBF) in de zijbalk.")
        st.stop()
    sources = pt100_uploads
else:
    if not pt100_paths:
        st.info("Vink nu een PT100-scan of een tweede referentie (DBF) aan.")
        st.stop()
    sources = [(path.name, path) for path in pt100_paths]

for source in sources:
    if source_mode == "Uploaden":
        source_name, source_data = source
        try:
            kind2, frame, channels, labels = cached_load_bytes(source_name, source_data)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{source_name}: {exc}")
            st.stop()
    else:
        source_name, source_path = source
        try:
            kind2, frame, channels, labels = cached_load(str(source_path), source_path.stat().st_mtime)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{source_name}: {exc}")
            st.stop()
    if kind2 == "referentie":
        if len(channels) > 1:
            chosen = st.selectbox(
                f"Kanaal {source_name}",
                options=channels,
                index=0,
                format_func=lambda ch, names=labels: names.get(ch, ch),
                key=f"meas_dbf_channel_{source_name}",
                help="Dit DBF-bestand telt als één meetkanaal. Kies welk kanaal met de referentie wordt vergeleken.",
            )
        else:
            chosen = channels[0]
        channel_id = _measurement_channel_id(source_name, used_channel_ids)
        frame, channels, labels = as_single_channel_measurement(frame, chosen, labels, channel_id)
    elif kind2 != "pt100":
        st.error(
            f"{source_name} is geen PT100-scan en geen referentie-DBF. "
            "Gebruik een Keysight-export of een tweede referentiebestand."
        )
        st.stop()
    else:
        used_channel_ids.update(channels)
    pt100_names.append(source_name)
    pt100_parts.append((frame, channels, labels))

rows_before = sum(len(frame) for frame, _channels, _labels in pt100_parts)
df2, pt_channels, pt_labels = combine_pt100_measurements(pt100_parts)
label2_name = ", ".join(pt100_names)
usable: list[str] = []
for frame, channels, _labels in pt100_parts:
    for channel in valid_channels(frame, channels):
        if channel not in usable:
            usable.append(channel)
skipped = [channel for channel in pt_channels if channel not in usable]

st.divider()
if len(pt100_names) == 1:
    st.subheader(f"2. Vergelijking met {pt100_names[0]}")
else:
    st.subheader(f"2. Vergelijking — {len(pt100_names)} meetbestanden")
st.caption(
    f"{label2_name} · {df2['tijd'].iloc[0]} → {df2['tijd'].iloc[-1]} · {len(df2)} scans"
)
if rows_before > len(df2):
    st.caption(
        f"{rows_before - len(df2)} scans met hetzelfde tijdstip zijn samengevoegd. "
        "Bij twee waarden op dat tijdstip blijft de eerste eindige waarde staan."
    )
if skipped:
    st.caption(
        "Overgeslagen (geen geldige meting, open of overrange): "
        + ", ".join(pt_labels.get(ch, ch) for ch in skipped)
    )
if not usable:
    st.error("De meetbestanden bevatten geen geldige temperaturen.")
    st.stop()

selected_channels = st.multiselect(
    "Kanalen in de vergelijking",
    options=usable,
    default=usable,
    format_func=lambda ch: pt_labels.get(ch, ch),
    help="Open kanalen staan hier niet bij. Een DBF-meting heeft één kanaal. "
    "Hetzelfde kanaalnummer uit meerdere CSV-bestanden is één kanaal.",
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

work_units = max(len(df1_win) + len(df2_in_range), 1)
previous_seconds = st.session_state.get("last_calc_seconds")
previous_work = st.session_state.get("last_calc_work")
if isinstance(previous_seconds, (int, float)) and isinstance(previous_work, (int, float)) and previous_work > 0:
    expected_seconds = float(previous_seconds) * work_units / float(previous_work)
else:
    expected_seconds = None
if expected_seconds is not None and expected_seconds >= 1:
    st.toast(f"Bezig met rekenen… nog ongeveer {_format_remaining(expected_seconds)}")
else:
    st.toast("Bezig met rekenen…")

calc_popup = st.empty()
calc_started = time.perf_counter()


def _calc_progress(fraction: float, detail: str = "") -> None:
    fraction = min(max(float(fraction), 0.0), 1.0)
    elapsed = time.perf_counter() - calc_started
    if 0.05 < fraction < 0.995 and elapsed >= 0.4:
        remaining = elapsed * (1.0 - fraction) / fraction
        text = f"Bezig met rekenen… nog ongeveer {_format_remaining(remaining)}"
    elif expected_seconds is not None and expected_seconds >= 1 and fraction < 0.995:
        text = f"Bezig met rekenen… nog ongeveer {_format_remaining(expected_seconds * (1.0 - fraction))}"
    else:
        text = "Bezig met rekenen…"
    if detail:
        text = f"{text} — {detail}"
    calc_popup.progress(fraction, text=text)


_calc_progress(0.05, "koppelen")
try:
    merged = merge_measurements(
        df1_win,
        df2_in_range,
        selected_channels,
        pt100_offset_seconds=0.0,
    )
except Exception as exc:  # noqa: BLE001
    calc_popup.empty()
    st.error(f"Koppelen van de metingen is mislukt: {exc}")
    st.stop()

if merged.empty:
    calc_popup.empty()
    st.error(
        "Geen overlappende tijdstippen. Pas de tijdspanne of de tijdoffset aan."
    )
    st.stop()

_calc_progress(0.4, "stabiele blokken")
if time_block_valid:
    blocks = detect_stable_blocks(
        merged,
        selected_channels,
        temp_step=float(temp_step),
        temp_tolerance=float(temp_tol),
        max_rate_k_per_min=float(max_rate),
        min_stable_minutes=float(min_stable),
        calc_start_before_minutes=float(calc_start_before),
        calc_end_before_minutes=float(calc_end_before),
        progress=lambda fraction, detail: _calc_progress(0.4 + 0.45 * float(fraction), detail),
    )
else:
    blocks = pd.DataFrame()
_calc_progress(0.9, "grafieken")
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

_calc_progress(1.0, "")
calc_elapsed = time.perf_counter() - calc_started
st.session_state["last_calc_seconds"] = calc_elapsed
st.session_state["last_calc_work"] = work_units
calc_popup.empty()
if calc_elapsed >= 2:
    st.toast(f"Berekening klaar, in {_format_remaining(calc_elapsed)}.")

report_html = build_report_html(
    path1_name=label1_name,
    path2_name=label2_name,
    ref_label=ref_label,
    range_start=range_start,
    range_end=range_end,
    max_rate=float(max_rate),
    min_stable=float(min_stable),
    calc_start=float(calc_start_before),
    calc_end=float(calc_end_before),
    fig_main=fig,
    fig_diff=fig_diff,
    fig_blocks=fig_blocks,
    block_display=block_display,
    wide_display=wide_display,
    diff_tol=float(diff_tol),
    laborant=laborant,
    referentie=referentie,
    calibrator=calibrator,
    meetmiddel=meetmiddel,
)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_col, html_col = st.columns(2)
with csv_col:
    st.download_button(
        "Download gekoppelde tabel (CSV)",
        data=_csv_bytes(merged, [col for col in merged.columns if str(col).startswith("d_")]),
        file_name="vergelijking_pt100.csv",
        mime="text/csv",
        key="download_coupled_top",
        help="Alleen de gekoppelde scans. ΔT is afgerond op 0,01 °C.",
    )
with html_col:
    st.download_button(
        "Download grafieken en verschillen (HTML)",
        data=report_html.encode("utf-8"),
        file_name=f"pt100_kalibratie_{stamp}.html",
        mime="text/html",
        key="download_report_html",
        help="Grafieken en gemeten verschillen. ΔT is afgerond op 0,01 °C.",
    )

st.session_state["assessment_token"] = int(st.session_state.get("assessment_token", 0)) + 1
if time_block_valid and not blocks.empty:
    st.session_state["assessment_base"] = assessment_measurement_base(blocks, selected_channels, pt_labels)
else:
    st.session_state["assessment_base"] = None
st.session_state["beoordeling_view"] = {
    "time_block_valid": bool(time_block_valid),
    "diff_tol": float(diff_tol),
    "label1_name": label1_name,
    "label2_name": label2_name,
    "ref_label": ref_label,
    "range_start": range_start,
    "range_end": range_end,
    "laborant": laborant,
    "referentie": referentie,
    "calibrator": calibrator,
    "meetmiddel": meetmiddel,
    "max_rate": float(max_rate),
    "min_stable": float(min_stable),
    "calc_start": float(calc_start_before),
    "calc_end": float(calc_end_before),
    "stamp": stamp,
}

tab_grafiek, tab_tabel, tab_blokken, tab_beoordeling = st.tabs(
    ["Temperatuur", "Gekoppelde tabel", "Stabiele verschillen", "Beoordeling"]
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
        data=_csv_bytes(merged, [col for col in merged.columns if str(col).startswith("d_")]),
        file_name="vergelijking_pt100.csv",
        mime="text/csv",
        key="download_coupled_tab",
    )

with tab_blokken:
    st.markdown(
        "Verschil **ΔT = PT100 − referentie**. Het einde van het setpoint is het moment dat de "
        f"referentie sneller dan **{max_rate:g} K/min** verandert, na minstens **{min_stable:g} min** stabiel. "
        f"Het gemiddelde loopt van **{calc_start_before:g}** tot **{calc_end_before:g} min** vóór dat einde."
    )
    if not time_block_valid:
        st.error(
            "De berekening is ongeldig. Zet de start verder vóór het einde van het setpoint dan het einde."
        )
    elif blocks.empty:
        st.warning(
            "Geen stabiele blokken gevonden. Verlaag de temperatuurverandering of de minimale stabiele duur, "
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
            data=_csv_bytes(wide_display, _difference_columns(wide_display)),
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

with tab_beoordeling:
    _show_beoordeling()
