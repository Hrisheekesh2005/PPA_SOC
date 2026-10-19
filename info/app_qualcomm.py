"""SoC PPA Advisor — transparent, parameter-driven engineering estimator.

Important: This is an early-design decision-support tool, not a replacement for
foundry PDKs, characterized Liberty libraries, synthesis, STA, power analysis,
place-and-route, or signoff flows. It intentionally does not infer silicon area
or power from the marketing name of a process node.
"""

import os
import json
import hashlib
import io
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from imagine import ChatMessage, ImagineClient
except ImportError:
    ChatMessage = None
    ImagineClient = None


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SoC PPA Advisor | Engineering Workbench",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ SoC PPA Advisor — Engineering Workbench")
st.caption(
    "Transparent early-design estimates, auditable assumptions, laptop/system budgeting, "
    "and AI-assisted review. Absolute accuracy depends on the quality of supplied data."
)

st.warning(
    "**Engineering maturity notice:** this application cannot produce signoff-accurate PPA from an application profile or node label alone. "
    "Use characterized library/IP data and synthesis/STA/power/physical-design reports wherever available. "
    "Unknown inputs remain unknown instead of being replaced by invented vendor-like constants."
)


# -----------------------------------------------------------------------------
# Starter architecture templates. Counts are editable planning placeholders,
# not specifications of any Intel, AMD, Qualcomm, TI, or other commercial SoC.
# -----------------------------------------------------------------------------
PROFILE_BLOCKS = {
    "Laptop / Client PC SoC": [
        ("CPU performance cores", 4),
        ("CPU efficiency cores", 8),
        ("Integrated GPU", 1),
        ("NPU / AI accelerator", 1),
        ("Shared cache / uncore", 1),
        ("Memory controller + NoC", 1),
        ("Media / display engine", 1),
        ("Other on-die logic", 1),
    ],
    "Mobile / Tablet SoC": [
        ("CPU performance cores", 1),
        ("CPU mid/efficiency cores", 7),
        ("Integrated GPU", 1),
        ("NPU / AI accelerator", 1),
        ("Shared cache / uncore", 1),
        ("Memory controller + NoC", 1),
        ("Image / media engine", 1),
        ("Other on-die logic", 1),
    ],
    "Embedded MCU / Industrial Control": [
        ("CPU / MCU core", 1),
        ("DSP / accelerator", 1),
        ("SRAM / memory macros", 1),
        ("Memory controller + interconnect", 1),
        ("Peripherals / I/O", 1),
        ("Other on-die logic", 1),
    ],
    "Edge AI / Robotics": [
        ("CPU performance cores", 2),
        ("CPU efficiency cores", 4),
        ("GPU / vision engine", 1),
        ("NPU / AI accelerator", 1),
        ("Shared cache / uncore", 1),
        ("Memory controller + NoC", 1),
        ("Image / sensor I/O", 1),
        ("Other on-die logic", 1),
    ],
    "Datacenter Compute SoC": [
        ("CPU cores", 64),
        ("Vector / matrix accelerator", 1),
        ("Shared cache / uncore", 1),
        ("Memory controllers + NoC", 1),
        ("High-speed I/O", 1),
        ("Other on-die logic", 1),
    ],
    "Custom SoC": [("Custom block 1", 1)],
}

TABLE_COLUMNS = [
    "Block",
    "Instances",
    "Ceff pF/instance",
    "Activity alpha",
    "Voltage V",
    "Frequency GHz",
    "Leakage W/instance",
    "Tool total power W/instance",
    "Std-cell area mm2/instance",
    "Macro area mm2/instance",
    "Data source / conditions",
]


def default_block_table(profile: str) -> pd.DataFrame:
    rows = []
    for name, count in PROFILE_BLOCKS[profile]:
        rows.append(
            {
                "Block": name,
                "Instances": count,
                "Ceff pF/instance": 0.0,
                "Activity alpha": 0.0,
                "Voltage V": 0.0,
                "Frequency GHz": 0.0,
                "Leakage W/instance": 0.0,
                "Tool total power W/instance": 0.0,
                "Std-cell area mm2/instance": 0.0,
                "Macro area mm2/instance": 0.0,
                "Data source / conditions": "Unspecified",
            }
        )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def numeric_column(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0).clip(lower=0.0)


def analyse_components(
    df: pd.DataFrame,
    power_method: str,
    target_frequency_ghz: float,
) -> Dict[str, Any]:
    """Compute block totals from explicitly supplied inputs; no node scaling.

    For the physics method, Pdyn[W] = alpha * Ceff[pF] * V[V]^2 * f[GHz] / 1000.
    The /1000 converts pF*GHz to watts. Inputs are per instance; Instances scales them.
    """
    work = df.copy()
    if "Block" not in work:
        work["Block"] = "Unnamed block"
    work["Block"] = work["Block"].fillna("").astype(str).str.strip()
    work.loc[work["Block"] == "", "Block"] = "Unnamed block"

    numeric_names = [c for c in TABLE_COLUMNS if c not in ("Block", "Data source / conditions")]
    if "Data source / conditions" not in work:
        work["Data source / conditions"] = "Unspecified"
    work["Data source / conditions"] = work["Data source / conditions"].fillna("Unspecified").astype(str)
    for name in numeric_names:
        if name not in work:
            work[name] = 0.0
        work[name] = numeric_column(work, name)

    counts = work["Instances"]
    work["Dynamic power W"] = 0.0
    work["Leakage power W"] = 0.0
    work["Estimated / reported block power W"] = 0.0

    if power_method == "Physics model: alpha-C-V-squared-f + leakage":
        per_instance_dynamic = (
            work["Ceff pF/instance"]
            * work["Activity alpha"].clip(upper=1.0)
            * work["Voltage V"] ** 2
            * work["Frequency GHz"]
            / 1000.0
        )
        work["Dynamic power W"] = counts * per_instance_dynamic
        work["Leakage power W"] = counts * work["Leakage W/instance"]
        work["Estimated / reported block power W"] = (
            work["Dynamic power W"] + work["Leakage power W"]
        )
    else:
        # Direct input is expected to be average total power per instance from a
        # tool report, validated estimate, or measurement. Do not add leakage again.
        work["Estimated / reported block power W"] = (
            counts * work["Tool total power W/instance"]
        )

    work["Std-cell area mm2"] = counts * work["Std-cell area mm2/instance"]
    work["Macro area mm2"] = counts * work["Macro area mm2/instance"]

    physics_inputs_present = bool(
        (
            (work["Instances"] > 0)
            & (work["Ceff pF/instance"] > 0)
            & (work["Voltage V"] > 0)
            & (work["Frequency GHz"] > 0)
        ).any()
        or (work["Leakage W/instance"] > 0).any()
    )
    direct_inputs_present = bool((work["Tool total power W/instance"] > 0).any())
    power_inputs_present = (
        physics_inputs_present
        if power_method == "Physics model: alpha-C-V-squared-f + leakage"
        else direct_inputs_present
    )

    return {
        "table": work,
        "dynamic_power_w": float(work["Dynamic power W"].sum()),
        "leakage_power_w": float(work["Leakage power W"].sum()),
        "soc_power_w": float(work["Estimated / reported block power W"].sum()),
        "stdcell_area_mm2": float(work["Std-cell area mm2"].sum()),
        "macro_area_mm2": float(work["Macro area mm2"].sum()),
        "power_inputs_present": power_inputs_present,
    }


def area_estimate(
    stdcell_area_mm2: float,
    macro_area_mm2: float,
    target_utilization_pct: float,
    floorplan_reserve_pct: float,
    die_periphery_pct: float,
) -> Dict[str, float]:
    """First-order planning footprint, not a detailed floorplan/die estimator.

    Core footprint approximation = (standard-cell area / target cell utilization) + macro area.
    It excludes detailed macro halos, routing channels, pin planning and other blockages unless
    represented by the additional reserve input. Die-periphery overhead is a separate user input.
    """
    util = max(1.0, min(95.0, float(target_utilization_pct))) / 100.0
    base_core = stdcell_area_mm2 / util + macro_area_mm2
    reserve = base_core * max(0.0, floorplan_reserve_pct) / 100.0
    core_footprint = base_core + reserve
    die_proxy = core_footprint * (1.0 + max(0.0, die_periphery_pct) / 100.0)
    placement_whitespace = max(0.0, stdcell_area_mm2 / util - stdcell_area_mm2)
    return {
        "base_core_area_mm2": base_core,
        "floorplan_reserve_mm2": reserve,
        "core_footprint_mm2": core_footprint,
        "die_area_proxy_mm2": die_proxy,
        "placement_whitespace_mm2": placement_whitespace,
    }


def make_ai_client(api_key: str):
    endpoint = os.environ.get(
        "INFERENCE_CLOUD_ENDPOINT",
        "https://aisuite.cirrascale.com/apis/v2",
    )
    if ImagineClient is None:
        raise RuntimeError(
            "The Imagine SDK is not installed. Run: python -m pip install python-imagine-sdk"
        )
    return ImagineClient(endpoint=endpoint, api_key=api_key)


# -----------------------------------------------------------------------------
# Sidebar: service, scenario and design constraints
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 AI Engine")
    api_key_input = st.text_input(
        "Qualcomm Inference Cloud API key",
        type="password",
        value=os.environ.get("INFERENCE_CLOUD_API_KEY", ""),
        help="Cirrascale Qualcomm AI Inference Cloud key. Do not commit it to source control.",
    )
    model_selection = st.selectbox(
        "Analysis model",
        [
            "DeepSeek-R1-Distill-Llama-70B",
            "Qwen-QwQ-32B",
            "Llama-3.3-70B",
            "Llama-3.1-8B",
        ],
        index=0,
        help="Availability and quotas are determined by your Cirrascale account.",
    )

    st.header("🧩 Architecture scenario")
    app_profile = st.selectbox("Platform profile", list(PROFILE_BLOCKS.keys()), index=0)
    process_label = st.selectbox(
        "Process / library label (metadata only)",
        [
            "Not specified",
            "Custom foundry PDK / standard-cell library",
            "28 nm-class process",
            "16 nm-class process",
            "7 nm-class process",
            "5 nm-class process",
            "3 nm-class process",
            "Other / proprietary",
        ],
        help=(
            "This label does not change calculations. Process-node names alone do not define cell area, "
            "capacitance, leakage, SRAM density, routing or timing."
        ),
    )

    st.header("📏 Constraints")
    power_budget_w = st.number_input(
        "SoC average / sustained power budget (W)", min_value=0.0, max_value=2000.0,
        value=15.0, step=1.0,
        help="A user-specified design target, not a vendor default or a measured limit.",
    )
    area_budget_mm2 = st.number_input(
        "Maximum planning-footprint budget (mm²)", min_value=0.0, max_value=100000.0,
        value=90.0, step=5.0,
        help="This check is against a first-order core/floorplan footprint unless you select the die proxy below.",
    )
    target_frequency_ghz = st.number_input(
        "Target clock frequency (GHz)", min_value=0.01, max_value=20.0,
        value=2.2, step=0.1,
        help="A target constraint; frequency closure is not inferred from this input.",
    )

st.markdown("## 1. Power and area data")
scenario_name = st.text_input("Scenario / workload name", value="Unspecified workload", help="Name the workload and operating state used for the estimates.")
workload_notes = st.text_area(
    "Workload / operating-point notes",
    value="",
    placeholder="Example: sustained compile workload; steady-state; activity from RTL VCD/SAIF; voltage and temperature corner; laptop balanced mode.",
    help="Record operating state, activity trace, duty cycle, workload duration and measurement setup if known.",
)

st.caption(
    "The starter block counts are editable examples. Populate per-instance parameters from a characterized library, "
    "synthesis/power report, prior silicon measurement, or a clearly identified reference design. "
    "Record each source/PVT/workload in the last column. Set Instances to zero for blocks absent from the candidate."
)

power_method = st.radio(
    "Power input method",
    [
        "Physics model: alpha-C-V-squared-f + leakage",
        "Tool/reference average power per block",
    ],
    horizontal=True,
    help=(
        "Physics mode calculates Pdynamic = alpha × Ceff × V² × f and adds supplied leakage. "
        "Direct mode sums per-instance total average power from reports or measurements and does not add leakage again."
    ),
)

# Keep editable data separate from the Streamlit widget-state key.
editor_data_key = f"block_data_{app_profile}"
editor_key = f"block_editor_{app_profile}"
if editor_data_key not in st.session_state:
    st.session_state[editor_data_key] = default_block_table(app_profile)
template_csv = default_block_table(app_profile).to_csv(index=False)
template_col, upload_col = st.columns([1, 2])
with template_col:
    st.download_button(
        "Download block-input CSV template",
        data=template_csv,
        file_name=f"soc_block_inputs_{app_profile.lower().replace(' ', '_').replace('/', '_')}.csv",
        mime="text/csv",
        help="Fill in per-instance power/area data from reports, a characterized library, or measurements, then upload it here.",
    )
with upload_col:
    uploaded_block_csv = st.file_uploader(
        "Load normalized block-input CSV",
        type=["csv"],
        key=f"block_csv_upload_{app_profile}",
        help="Use the template format. This does not parse proprietary raw tool reports automatically; normalize report data into these columns first.",
    )
if uploaded_block_csv is not None:
    csv_bytes = uploaded_block_csv.getvalue()
    upload_signature = hashlib.sha256(csv_bytes).hexdigest()
    signature_key = f"block_csv_signature_{app_profile}"
    if st.session_state.get(signature_key) != upload_signature:
        try:
            imported_df = pd.read_csv(io.BytesIO(csv_bytes))
            if "Block" not in imported_df.columns or "Instances" not in imported_df.columns:
                st.error("CSV needs at least 'Block' and 'Instances' columns. Download the template and preserve its column names.")
            else:
                for col in TABLE_COLUMNS:
                    if col not in imported_df.columns:
                        if col == "Block":
                            imported_df[col] = "Imported block"
                        elif col == "Data source / conditions":
                            imported_df[col] = "Unspecified — imported CSV"
                        else:
                            imported_df[col] = 0.0
                imported_df = imported_df[TABLE_COLUMNS].copy()
                if editor_key in st.session_state:
                    del st.session_state[editor_key]
                st.session_state[editor_data_key] = imported_df
                st.session_state[signature_key] = upload_signature
                st.success(f"Loaded {len(imported_df)} block rows. Review them in the editor below before running calculations.")
        except Exception as import_exc:
            st.error(f"Could not read that CSV: {import_exc}")

column_config = {
    "Block": st.column_config.TextColumn("Block / power domain", help="Editable block name."),
    "Instances": st.column_config.NumberColumn("Instances", min_value=0, step=1, format="%d"),
    "Ceff pF/instance": st.column_config.NumberColumn(
        "Ceff (pF) / instance", min_value=0.0, step=0.01, format="%.4f",
        help="Effective switched capacitance per instance from a library/activity model or a calibrated estimate. Decimal inputs are accepted; use units consistently. Not the sum of all gate capacitances unless that matches your alpha convention."
    ),
    "Activity alpha": st.column_config.NumberColumn(
        "Activity α", min_value=0.0, max_value=1.0, step=0.001, format="%.4f",
        help="Average switching-activity factor used in P = α C V² f. Up to four decimal places are displayed. Define it consistently with Ceff."
    ),
    "Voltage V": st.column_config.NumberColumn("Voltage (V)", min_value=0.0, step=0.01, format="%.4f"),
    "Frequency GHz": st.column_config.NumberColumn("Frequency (GHz)", min_value=0.0, step=0.01, format="%.4f"),
    "Leakage W/instance": st.column_config.NumberColumn(
        "Leakage (W) / instance", min_value=0.0, step=0.0001, format="%.4f",
        help="Obtain from characterized cells/IP or an applicable power report at the relevant process, voltage and temperature. Decimal inputs down to four displayed places are supported."
    ),
    "Tool total power W/instance": st.column_config.NumberColumn(
        "Tool total power (W) / instance", min_value=0.0, step=0.01, format="%.4f",
        help="Direct mode only: average total power per instance, including leakage if the source report includes it. Decimal inputs are supported."
    ),
    "Std-cell area mm2/instance": st.column_config.NumberColumn(
        "Std-cell area (mm²) / instance", min_value=0.0, step=0.01, format="%.4f",
        help="Standard-cell area from synthesis/library reports; do not enter floorplan area here. Decimal inputs are supported."
    ),
    "Macro area mm2/instance": st.column_config.NumberColumn(
        "Hard-macro area (mm²) / instance", min_value=0.0, step=0.01, format="%.4f",
        help="SRAM/ROM/analog/other hard-macro area from the relevant compiler or IP documentation. Decimal inputs are supported."
    ),
    "Data source / conditions": st.column_config.TextColumn(
        "Data source / PVT / workload notes",
        help="Record report/file/tool version, Liberty/PDK/IP version, voltage, temperature, activity source, workload and operating state where relevant."
    ),
}

# IMPORTANT: Keep the source dataframe stable between reruns. Streamlit stores
# pending cell edits/additions/deletions in the data_editor widget state. Writing
# the returned edited dataframe back into the widget's input on every rerun can
# make the editor rebase its own pending edits, which can cause values to snap
# back, keyboard focus to jump, and the horizontal scroll position to reset.
# Only replace editor_data_key intentionally (initialization/profile/CSV import).
block_df = st.data_editor(
    st.session_state[editor_data_key].copy(deep=True),
    key=editor_key,
    column_config=column_config,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
)
# Use the returned edited dataframe for all calculations below. Do not assign it
# back to st.session_state[editor_data_key] here; widget state owns live edits.

col_complete_pwr, col_complete_area = st.columns(2)
with col_complete_pwr:
    power_coverage_confirmed = st.checkbox(
        "I have supplied power inputs for all included blocks/domains",
        value=False,
        key=f"power_coverage_{app_profile}",
        help="Only mark this after accounting for each in-scope block and workload state. This is a user assertion, not automated verification.",
    )
with col_complete_area:
    area_coverage_confirmed = st.checkbox(
        "I have supplied area inputs for all included blocks/macros",
        value=False,
        key=f"area_coverage_{app_profile}",
        help="Only mark this after accounting for all standard-cell and hard-macro areas in the candidate.",
    )

st.markdown("### Floorplan assumptions")
area_col1, area_col2, area_col3 = st.columns(3)
with area_col1:
    target_utilization_pct = st.slider(
        "Target standard-cell utilization (%)", min_value=20, max_value=90, value=65, step=1,
        help="A floorplanning target to test, not a universal industry standard. High utilization can increase congestion and timing risk."
    )
with area_col2:
    floorplan_reserve_pct = st.number_input(
        "Extra floorplan reserve (%)", min_value=0.0, max_value=200.0, value=0.0, step=2.5,
        help="Optional user-supplied allowance for halos, routing channels, blockages, pin planning and other effects not explicitly modeled."
    )
with area_col3:
    die_periphery_pct = st.number_input(
        "Die / I/O periphery allowance (%)", min_value=0.0, max_value=300.0, value=0.0, step=5.0,
        help="Optional user-supplied die-edge/I/O/periphery allowance. Zero means no die-area proxy is available."
    )
area_constraint_basis = st.radio(
    "Compare area budget against",
    ["Core/floorplan planning footprint", "Die-area proxy (requires periphery allowance)"],
    horizontal=True,
)

calc = analyse_components(block_df, power_method, target_frequency_ghz)
plot_df = calc["table"]
area = area_estimate(
    calc["stdcell_area_mm2"],
    calc["macro_area_mm2"],
    target_utilization_pct,
    floorplan_reserve_pct,
    die_periphery_pct,
)

# Only show feasibility conclusions when the user has declared complete coverage and
# the relevant input category contains nonzero data. Partial sums are still shown as partial.
# Coverage check: every active/in-scope block needs a usable source value.
# This prevents one populated row from accidentally making a partial subtotal appear complete.
power_missing_domains = []
for _, row in plot_df.iterrows():
    if float(row["Instances"]) <= 0:
        continue
    block_name = str(row["Block"])
    if power_method == "Physics model: alpha-C-V-squared-f + leakage":
        dynamic_parameters_present = (
            float(row["Ceff pF/instance"]) > 0
            and float(row["Voltage V"]) > 0
            and float(row["Frequency GHz"]) > 0
        )
        leakage_only_present = float(row["Leakage W/instance"]) > 0
        if not (dynamic_parameters_present or leakage_only_present):
            power_missing_domains.append(block_name)
    elif float(row["Tool total power W/instance"]) <= 0:
        power_missing_domains.append(block_name)

area_missing_domains = []
for _, row in plot_df.iterrows():
    if float(row["Instances"]) <= 0:
        continue
    if (
        float(row["Std-cell area mm2/instance"]) <= 0
        and float(row["Macro area mm2/instance"]) <= 0
    ):
        area_missing_domains.append(str(row["Block"]))

power_complete = (
    power_coverage_confirmed
    and calc["power_inputs_present"]
    and calc["soc_power_w"] > 0
    and not power_missing_domains
)
area_complete = (
    area_coverage_confirmed
    and (calc["stdcell_area_mm2"] + calc["macro_area_mm2"] > 0)
    and not area_missing_domains
)
area_value_for_check = (
    area["core_footprint_mm2"]
    if area_constraint_basis == "Core/floorplan planning footprint"
    else (area["die_area_proxy_mm2"] if die_periphery_pct > 0 else None)
)
area_check_available = area_complete and area_value_for_check is not None

# -----------------------------------------------------------------------------
# Summary dashboard
# -----------------------------------------------------------------------------
st.markdown("## 2. PPA summary")
metric_cols = st.columns(4)
if power_complete:
    metric_cols[0].metric("Modeled SoC power", f"{calc['soc_power_w']:.3f} W")
    metric_cols[1].metric("Power budget headroom", f"{power_budget_w - calc['soc_power_w']:.3f} W")
else:
    metric_cols[0].metric("Power subtotal", f"{calc['soc_power_w']:.3f} W", help="Partial or unverified until full block coverage is confirmed.")
    metric_cols[1].metric("Power feasibility", "Not verified")
if area_complete:
    metric_cols[2].metric("Estimated core/floorplan", f"{area['core_footprint_mm2']:.3f} mm²")
else:
    metric_cols[2].metric("Area footprint", "Incomplete inputs")
if area_check_available:
    metric_cols[3].metric("Area budget headroom", f"{area_budget_mm2 - float(area_value_for_check):.3f} mm²")
else:
    metric_cols[3].metric("Area feasibility", "Not verified")

if power_missing_domains:
    st.caption("Power inputs still missing for active rows: " + ", ".join(power_missing_domains[:12]) + (" …" if len(power_missing_domains) > 12 else ""))
if area_missing_domains:
    st.caption("Area inputs still missing for active rows: " + ", ".join(area_missing_domains[:12]) + (" …" if len(area_missing_domains) > 12 else ""))

if power_complete:
    if calc["soc_power_w"] <= power_budget_w:
        st.success(f"Power budget check: user-supplied model is {power_budget_w - calc['soc_power_w']:.3f} W below the configured budget.")
    else:
        st.error(f"Power budget check: modeled power exceeds the configured budget by {calc['soc_power_w'] - power_budget_w:.3f} W.")
else:
    st.warning("Power feasibility is **not verified**. The displayed total is a partial subtotal or no valid power inputs have been supplied.")

if area_check_available:
    if float(area_value_for_check) <= area_budget_mm2:
        st.success(f"Area budget check: estimated footprint is {area_budget_mm2 - float(area_value_for_check):.3f} mm² below the configured budget.")
    else:
        st.error(f"Area budget check: estimated footprint exceeds the configured budget by {float(area_value_for_check) - area_budget_mm2:.3f} mm².")
else:
    if area_constraint_basis.startswith("Die-area") and die_periphery_pct <= 0:
        st.warning("Die area cannot be checked: provide a justified die/I/O periphery allowance or select the core/floorplan check.")
    else:
        st.warning("Area feasibility is **not verified**. Supply complete standard-cell and hard-macro area inputs from synthesis/library/IP data.")

st.caption(
    f"Process/library label: **{process_label}** (metadata only; it does not scale the equations). "
    f"Target period: **{1.0 / target_frequency_ghz:.4f} ns** at {target_frequency_ghz:.3f} GHz. "
    "A target clock is not evidence of timing closure."
)

with st.expander("Calculation basis and input provenance", expanded=False):
    st.markdown(
        "**Physics power mode**  \n"
        "For each block instance: `Pdynamic (W) = alpha × Ceff (pF) × V² × f (GHz) / 1000`. "
        "The conversion factor follows from pF × GHz = mW. The modeled domain power is the sum of dynamic power and the separately supplied leakage power. "
        "This is a first-order lumped-capacitance estimate; it does not model glitches, internal cell power, clock-tree power, power-gating transitions, rail losses, or temperature dependence unless represented in the supplied inputs.\n\n"
        "**Direct reference mode**  \n"
        "Per-block average total power is summed from user-supplied tool/reference/measurement values. Do not mix operating points or double-count leakage.\n\n"
        "**Area mode**  \n"
        "First-order floorplan footprint = `standard-cell area / target utilization + macro area`, plus the user-entered reserve. "
        "It is not a die-area prediction unless a separately justified I/O/periphery allowance is supplied, and does not replace placement, routing, macro halo, or congestion analysis.\n\n"
        "**Process node**  \n"
        "The node label is not used as a multiplier. Foundry, device architecture, threshold-voltage options, library cells, SRAM compiler, voltage, temperature, routing and design rules materially affect PPA."
    )

# -----------------------------------------------------------------------------
# Optional timing / thermal validation from user-provided tool/model results
# -----------------------------------------------------------------------------
st.markdown("## 3. Timing and thermal checks")
timing_col, thermal_col = st.columns(2)
with timing_col:
    st.subheader("Static timing analysis (STA)")
    has_sta = st.checkbox("I have a STA report for this candidate", value=False)
    if has_sta:
        wns_ns = st.number_input(
            "Worst setup slack at the target clock (ns)", min_value=-100.0, max_value=100.0,
            value=0.0, step=0.01,
            help="Enter the reported setup slack for the relevant mode/corner after reading the timing report. This single value is not a substitute for multi-corner signoff."
        )
        if wns_ns >= 0:
            st.success(f"Reported setup slack is non-negative ({wns_ns:.3f} ns) for the supplied report context.")
        else:
            st.error(f"Reported setup slack is negative ({wns_ns:.3f} ns); the analyzed timing context misses setup.")
    else:
        wns_ns = None
        st.info("Timing closure: not checked. Use STA with the actual netlist, Liberty libraries, SDC constraints and extracted parasitics as appropriate.")

with thermal_col:
    st.subheader("Steady-state thermal estimate")
    has_thermal_model = st.checkbox("I have a validated effective thermal resistance for this system", value=False)
    if has_thermal_model:
        ambient_c = st.number_input("Ambient / inlet temperature (°C)", min_value=-40.0, max_value=100.0, value=25.0, step=1.0)
        theta_c_per_w = st.number_input(
            "Effective system thermal resistance (°C/W)", min_value=0.001, max_value=100.0,
            value=1.0, step=0.1,
            help="Use data for this package, board, chassis/cooling system and workload; a generic value is not portable across laptops."
        )
        max_junction_c = st.number_input("Project junction-temperature limit (°C)", min_value=1.0, max_value=200.0, value=95.0, step=1.0)
        if power_complete:
            tj_est = ambient_c + calc["soc_power_w"] * theta_c_per_w
            st.metric("First-order steady-state junction estimate", f"{tj_est:.1f} °C", f"Limit margin: {max_junction_c - tj_est:.1f} °C")
            if tj_est > max_junction_c:
                st.error("This first-order thermal estimate exceeds the user-entered junction limit.")
            else:
                st.info("This estimate is only as reliable as the supplied effective thermal resistance and average-power inputs.")
        else:
            st.info("Thermal result unavailable until power inputs are complete.")
    else:
        st.info("Thermal estimate withheld unless an applicable thermal model/measurement is supplied.")

# -----------------------------------------------------------------------------
# Laptop / system-level analysis
# -----------------------------------------------------------------------------
if app_profile == "Laptop / Client PC SoC":
    st.markdown("## 4. Laptop system power and battery runtime")
    st.caption(
        "SoC power alone is not laptop power. Enter average power for off-SoC loads at the same workload and operating point. "
        "Battery runtime is an energy-budget estimate, not a battery-life guarantee."
    )
    batt_col, load_col = st.columns(2)
    with batt_col:
        battery_capacity_wh = st.number_input("Battery nameplate capacity (Wh)", min_value=0.0, max_value=500.0, value=0.0, step=1.0)
        usable_battery_pct = st.slider("Usable battery-energy fraction (%)", min_value=1, max_value=100, value=85, step=1)
        battery_path_eff_pct = st.slider("Battery-to-load conversion efficiency (%)", min_value=1, max_value=100, value=90, step=1)
    with load_col:
        display_power_w = st.number_input("Display / panel average power (W)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
        dram_power_w = st.number_input("External DRAM average power (W)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
        storage_network_fan_w = st.number_input("Storage + Wi-Fi + fans + other average power (W)", min_value=0.0, max_value=200.0, value=0.0, step=0.5)
        system_loads_covered = st.checkbox(
            "I have accounted for all significant non-SoC system loads",
            value=False,
            help="Use same-workload average values or measurements for display, DRAM, storage, wireless, fans and other significant system loads.",
        )

    non_soc_power_w = display_power_w + dram_power_w + storage_network_fan_w
    laptop_system_power_w = calc["soc_power_w"] + non_soc_power_w
    battery_runtime_h = None
    if power_complete and system_loads_covered and battery_capacity_wh > 0 and laptop_system_power_w > 0:
        usable_energy_wh = battery_capacity_wh * (usable_battery_pct / 100.0) * (battery_path_eff_pct / 100.0)
        battery_runtime_h = usable_energy_wh / laptop_system_power_w

    l1, l2, l3 = st.columns(3)
    l1.metric("Entered non-SoC system power", f"{non_soc_power_w:.2f} W")
    l2.metric(
        "Modeled average laptop system power",
        f"{laptop_system_power_w:.2f} W" if (power_complete and system_loads_covered) else "Incomplete load data",
    )
    l3.metric("Idealized runtime estimate", f"{battery_runtime_h:.2f} h" if battery_runtime_h is not None else "Complete power + load inputs")

    if battery_runtime_h is not None:
        st.caption(
            f"Runtime uses {battery_capacity_wh:.1f} Wh × {usable_battery_pct}% usable fraction × "
            f"{battery_path_eff_pct}% conversion efficiency, divided by modeled average system power. "
            "Workload changes, battery ageing, sleep states, display brightness and thermal throttling are not modeled."
        )

# -----------------------------------------------------------------------------
# Graphs — only graph entered/calculated data, with assumptions clearly disclosed
# -----------------------------------------------------------------------------
st.markdown("## 5. Engineering plots")
plot_df = calc["table"]
if calc["power_inputs_present"] and calc["soc_power_w"] > 0:
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.subheader("Power by block / domain")
        fig_power = go.Figure()
        if power_method == "Physics model: alpha-C-V-squared-f + leakage":
            fig_power.add_trace(go.Bar(x=plot_df["Block"], y=plot_df["Dynamic power W"], name="Dynamic (modeled)"))
            fig_power.add_trace(go.Bar(x=plot_df["Block"], y=plot_df["Leakage power W"], name="Leakage (input)"))
            fig_power.update_layout(barmode="stack")
        else:
            fig_power.add_trace(go.Bar(x=plot_df["Block"], y=plot_df["Estimated / reported block power W"], name="Tool/reference average total power"))
        fig_power.update_layout(
            xaxis_title="Block / domain", yaxis_title="Power (W)",
            margin=dict(t=20, b=100, l=35, r=20), height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig_power.update_xaxes(tickangle=-35)
        st.plotly_chart(fig_power, use_container_width=True)

    with p_col2:
        st.subheader("Power budget usage")
        p_use_pct = (calc["soc_power_w"] / power_budget_w * 100.0) if power_budget_w > 0 else None
        fig_budget = go.Figure()
        fig_budget.add_trace(go.Bar(
            x=["Modeled / reported SoC power"], y=[calc["soc_power_w"]],
            name="Power", text=[f"{calc['soc_power_w']:.3f} W"], textposition="outside"
        ))
        if power_budget_w > 0:
            fig_budget.add_hline(y=power_budget_w, line_dash="dash", annotation_text=f"Budget {power_budget_w:.2f} W")
        fig_budget.update_layout(
            yaxis_title="Power (W)", margin=dict(t=45, b=35, l=35, r=20), height=420,
        )
        st.plotly_chart(fig_budget, use_container_width=True)
        if power_complete and p_use_pct is not None:
            st.caption(f"Power-budget use: {p_use_pct:.1f}%. This status is valid only if all in-scope power domains and the operating scenario are represented.")
        else:
            st.caption("Partial subtotal only; do not treat the bar as a whole-SoC power estimate until block coverage is complete.")
else:
    st.info("Power plots will populate after you enter valid power values for at least one block. Feasibility still requires complete block coverage.")

if calc["stdcell_area_mm2"] + calc["macro_area_mm2"] > 0:
    st.subheader("Area composition and planning footprint")
    base = area["core_footprint_mm2"]
    cell = calc["stdcell_area_mm2"]
    macros = calc["macro_area_mm2"]
    whitespace = area["placement_whitespace_mm2"]
    extra_reserve = area["floorplan_reserve_mm2"]
    area_labels = ["Standard cells", "Hard macros", "Placement whitespace", "Extra floorplan reserve"]
    area_values = [cell, macros, whitespace, extra_reserve]
    fig_area = go.Figure(go.Bar(
        x=area_labels, y=area_values,
        text=[f"{v:.3f} mm²" for v in area_values], textposition="outside",
    ))
    area_plot_title = "Area estimate from supplied area data" if area_complete else "Partial area subtotal — coverage incomplete"
    fig_area.update_layout(
        title=area_plot_title,
        xaxis_title="Area category", yaxis_title="Area (mm²)",
        margin=dict(t=55, b=55, l=35, r=20), height=400,
    )
    st.plotly_chart(fig_area, use_container_width=True)
    if area_complete:
        st.caption(
            f"Core/floorplan planning footprint: {base:.3f} mm². "
            + (f"User-margin die proxy: {area['die_area_proxy_mm2']:.3f} mm²." if die_periphery_pct > 0 else "Die area is not inferred because no die/I/O periphery allowance was supplied.")
        )
    else:
        st.caption(
            f"Partial subtotal only: {base:.3f} mm² derived from the area rows entered. "
            "Do not treat this as a complete design footprint until all block and macro areas are supplied."
        )
else:
    st.info("Area plot withheld until standard-cell and/or hard-macro area inputs are entered.")

if app_profile == "Laptop / Client PC SoC" and power_complete and system_loads_covered:
    system_df = pd.DataFrame(
        {
            "Subsystem": ["SoC", "Display", "External DRAM", "Storage + Wi-Fi + fans + other"],
            "Average power (W)": [calc["soc_power_w"], display_power_w, dram_power_w, storage_network_fan_w],
        }
    )
    st.subheader("Laptop system power budget")
    fig_sys = go.Figure(go.Pie(
        labels=system_df["Subsystem"], values=system_df["Average power (W)"], hole=0.42,
        hovertemplate="%{label}<br>%{value:.2f} W<extra></extra>",
    ))
    fig_sys.update_layout(margin=dict(t=25, b=20, l=20, r=20), height=380)
    st.plotly_chart(fig_sys, use_container_width=True)

if power_method == "Physics model: alpha-C-V-squared-f + leakage" and power_complete:
    with st.expander("Optional: first-order frequency sensitivity graph", expanded=False):
        st.caption(
            "This is a what-if calculation only: it scales every entered block frequency proportionally to the selected target while holding voltage, activity, effective capacitance and leakage fixed. "
            "Real DVFS, timing feasibility and leakage can change with operating point; this is not a timing/power signoff curve."
        )
        freq_grid = np.linspace(max(0.01, target_frequency_ghz * 0.25), target_frequency_ghz * 1.75, 21)
        rows = []
        active = plot_df[plot_df["Instances"] > 0]
        for f_target in freq_grid:
            dyn = 0.0
            leak = 0.0
            for _, r in active.iterrows():
                instances = float(r["Instances"])
                scale = float(f_target) / target_frequency_ghz if target_frequency_ghz > 0 else 1.0
                dyn += (
                    instances * float(r["Ceff pF/instance"]) * min(1.0, float(r["Activity alpha"]))
                    * float(r["Voltage V"]) ** 2 * (float(r["Frequency GHz"]) * scale) / 1000.0
                )
                leak += instances * float(r["Leakage W/instance"])
            rows.append({"Frequency scale target (GHz)": float(f_target), "Modelled power (W)": dyn + leak})
        freq_df = pd.DataFrame(rows)
        fig_freq = go.Figure(go.Scatter(
            x=freq_df["Frequency scale target (GHz)"], y=freq_df["Modelled power (W)"],
            mode="lines+markers", name="First-order model",
        ))
        fig_freq.add_hline(y=power_budget_w, line_dash="dash", annotation_text="Power budget")
        fig_freq.add_vline(x=target_frequency_ghz, line_dash="dot", annotation_text="Selected target")
        fig_freq.update_layout(
            xaxis_title="Frequency scale target (GHz)", yaxis_title="Modeled power (W)",
            margin=dict(t=45, b=35, l=35, r=20), height=390,
        )
        st.plotly_chart(fig_freq, use_container_width=True)

# -----------------------------------------------------------------------------
# AI-assisted engineering review
# -----------------------------------------------------------------------------
st.markdown("## 6. Senior silicon architect review")
st.caption("The AI interprets your inputs; it does not create measured PPA data or replace EDA tools.")

component_snapshot: List[Dict[str, Any]] = []
for _, r in plot_df.iterrows():
    component_snapshot.append(
        {
            "block": str(r["Block"]),
            "instances": int(r["Instances"]),
            # Include source parameters as well as outputs. Omitting these inputs
            # previously caused the AI to incorrectly claim activity alpha was missing.
            "ceff_pf_per_instance": round(float(r["Ceff pF/instance"]), 6),
            "activity_alpha": round(float(r["Activity alpha"]), 6),
            "voltage_v": round(float(r["Voltage V"]), 6),
            "frequency_ghz": round(float(r["Frequency GHz"]), 6),
            "leakage_w_per_instance": round(float(r["Leakage W/instance"]), 6),
            "tool_total_power_w_per_instance": round(float(r["Tool total power W/instance"]), 6),
            "dynamic_power_w": round(float(r["Dynamic power W"]), 6),
            "leakage_power_w": round(float(r["Leakage power W"]), 6),
            "direct_or_total_power_w": round(float(r["Estimated / reported block power W"]), 6),
            "stdcell_area_mm2_per_instance": round(float(r["Std-cell area mm2/instance"]), 6),
            "macro_area_mm2_per_instance": round(float(r["Macro area mm2/instance"]), 6),
            "stdcell_area_mm2": round(float(r["Std-cell area mm2"]), 6),
            "macro_area_mm2": round(float(r["Macro area mm2"]), 6),
            "data_source_conditions": str(r["Data source / conditions"]),
        }
    )

# Compute component shares deterministically so the model need not estimate them.
total_dynamic_for_share = float(calc["dynamic_power_w"])
total_block_power_for_share = float(calc["soc_power_w"])
for component in component_snapshot:
    component["share_of_total_dynamic_power_pct"] = (
        round(component["dynamic_power_w"] / total_dynamic_for_share * 100.0, 3)
        if total_dynamic_for_share > 0 else None
    )
    component["share_of_total_soc_power_pct"] = (
        round(component["direct_or_total_power_w"] / total_block_power_for_share * 100.0, 3)
        if total_block_power_for_share > 0 else None
    )

if power_complete:
    power_headroom = round(power_budget_w - calc["soc_power_w"], 6)
    power_status = "PASS" if power_headroom >= 0 else "VIOLATION"
else:
    power_headroom = None
    power_status = "NOT_VERIFIED_INCOMPLETE_INPUT_COVERAGE"

if area_check_available:
    area_headroom = round(area_budget_mm2 - float(area_value_for_check), 6)
    area_status = "PASS" if area_headroom >= 0 else "VIOLATION"
else:
    area_headroom = None
    area_status = "NOT_VERIFIED_INCOMPLETE_OR_UNAVAILABLE_AREA_BASIS"

active_rows = plot_df[plot_df["Instances"] > 0]
activity_alpha_values = [round(float(v), 6) for v in active_rows["Activity alpha"].tolist()]
activity_alpha_nonzero_all_active = bool(len(active_rows) > 0 and all(v > 0 for v in activity_alpha_values))
cpu_dynamic_w = float(
    active_rows.loc[
        active_rows["Block"].astype(str).str.contains("cpu|processor", case=False, regex=True),
        "Dynamic power W",
    ].sum()
)
cpu_dynamic_share_pct = (
    round(cpu_dynamic_w / total_dynamic_for_share * 100.0, 3)
    if total_dynamic_for_share > 0 else None
)

verified_calculations = {
    "important_note": "These statuses are calculated by the application from the current entered values. They are arithmetic checks on the model, not proof of silicon accuracy.",
    "power_status": power_status,
    "power_budget_w": round(float(power_budget_w), 6),
    "modeled_or_reported_soc_power_w": round(float(calc["soc_power_w"]), 6),
    "power_headroom_w_budget_minus_power": power_headroom,
    "power_coverage_complete": bool(power_complete),
    "area_status": area_status,
    "area_constraint_basis": area_constraint_basis,
    "area_budget_mm2": round(float(area_budget_mm2), 6),
    "area_value_used_for_budget_check_mm2": round(float(area_value_for_check), 6) if area_value_for_check is not None else None,
    "area_headroom_mm2_budget_minus_checked_area": area_headroom,
    "area_coverage_complete": bool(area_complete),
    "core_floorplan_planning_footprint_mm2": round(float(area["core_footprint_mm2"]), 6) if area_complete else None,
    "die_area_planning_proxy_mm2": round(float(area["die_area_proxy_mm2"]), 6) if area_complete and die_periphery_pct > 0 else None,
    "dynamic_power_w": round(float(calc["dynamic_power_w"]), 6),
    "leakage_power_w": round(float(calc["leakage_power_w"]), 6),
    "activity_alpha_inputs_for_active_rows": activity_alpha_values if power_method == "Physics model: alpha-C-V-squared-f + leakage" else "Not applicable to direct total-power mode",
    "all_active_rows_have_nonzero_alpha": activity_alpha_nonzero_all_active if power_method == "Physics model: alpha-C-V-squared-f + leakage" else None,
    "cpu_and_processor_blocks_dynamic_power_share_pct": cpu_dynamic_share_pct,
    "sta_status": ("REPORTED_NONNEGATIVE_WNS" if has_sta and wns_ns is not None and wns_ns >= 0 else "REPORTED_NEGATIVE_WNS" if has_sta and wns_ns is not None else "NOT_CHECKED"),
    "thermal_status": "USER_SUPPLIED_THERMAL_MODEL" if has_thermal_model else "NOT_VERIFIED_NO_VALIDATED_THERMAL_MODEL",
    "laptop_system_power_w": round(float(laptop_system_power_w), 6) if app_profile == "Laptop / Client PC SoC" and power_complete and system_loads_covered else None,
    "laptop_runtime_hours": round(float(battery_runtime_h), 6) if app_profile == "Laptop / Client PC SoC" and battery_runtime_h is not None else None,
}

analysis_context = {
    "verified_calculations_authoritative": verified_calculations,
    "platform_profile": app_profile,
    "scenario_workload_name": scenario_name,
    "workload_operating_point_notes": workload_notes,
    "process_label_metadata_only": process_label,
    "power_method": power_method,
    "target_frequency_ghz": target_frequency_ghz,
    "required_clock_period_ns": 1.0 / target_frequency_ghz,
    "power_budget_w": power_budget_w,
    "power_coverage_confirmed_by_user": bool(power_coverage_confirmed),
    "power_feasibility_available": bool(power_complete),
    "power_total_or_partial_w": round(calc["soc_power_w"], 6),
    "dynamic_power_w": round(calc["dynamic_power_w"], 6),
    "leakage_power_w": round(calc["leakage_power_w"], 6),
    "power_headroom_w_if_complete": round(power_budget_w - calc["soc_power_w"], 6) if power_complete else None,
    "area_coverage_confirmed_by_user": bool(area_coverage_confirmed),
    "area_feasibility_available": bool(area_check_available),
    "stdcell_area_mm2_subtotal": round(calc["stdcell_area_mm2"], 6),
    "macro_area_mm2_subtotal": round(calc["macro_area_mm2"], 6),
    "target_utilization_pct": target_utilization_pct,
    "floorplan_reserve_pct": floorplan_reserve_pct,
    "core_floorplan_area_mm2_if_complete": round(area["core_footprint_mm2"], 6) if area_complete else None,
    "die_area_proxy_mm2_if_periphery_given": round(area["die_area_proxy_mm2"], 6) if area_complete and die_periphery_pct > 0 else None,
    "area_constraint_basis": area_constraint_basis,
    "area_budget_mm2": area_budget_mm2,
    "sta_report_present": bool(has_sta),
    "reported_wns_ns": wns_ns,
    "thermal_model_present": bool(has_thermal_model),
    "laptop_battery_wh": battery_capacity_wh if app_profile == "Laptop / Client PC SoC" else None,
    "laptop_non_soc_power_w": round(non_soc_power_w, 6) if app_profile == "Laptop / Client PC SoC" else None,
    "laptop_non_soc_load_coverage_confirmed": bool(system_loads_covered) if app_profile == "Laptop / Client PC SoC" else None,
    "laptop_runtime_hours_estimate": round(battery_runtime_h, 4) if app_profile == "Laptop / Client PC SoC" and battery_runtime_h is not None else None,
    "components": component_snapshot,
}

with st.expander("Review data sent to AI", expanded=False):
    st.json(analysis_context)

if st.button("Run AI Architectural Trade-off Analysis", type="primary"):
    system_instructions = """You are a rigorous SoC architecture/PPA reviewer with expertise in low-power VLSI, digital design, power estimation, physical design, laptop/client compute SoCs and EDA validation.

Follow these non-negotiable rules:
- FIRST read `verified_calculations_authoritative`. These are deterministic results computed by the application. They are authoritative for arithmetic and configured constraint checks; do not contradict, reinterpret, or reverse their PASS / VIOLATION / NOT_VERIFIED statuses.
- If `power_status` is `VIOLATION`, explicitly say power exceeds budget by the absolute value of `power_headroom_w_budget_minus_power`. Do not say the power budget is met.
- If `area_status` is `VIOLATION`, explicitly say the selected area basis exceeds the area budget by the absolute value of `area_headroom_mm2_budget_minus_checked_area`. Do not say the area budget is sufficient. If the checked basis is core/floorplan planning footprint, state the die proxy separately and do not use the proxy as the checked metric.
- Report the provided per-block `activity_alpha` values when in physics mode. Do not claim alpha is missing if values are present. You may state the workload/activity provenance is synthetic or unvalidated if indicated by the data-source notes.
- Use the supplied deterministic CPU/processor dynamic-power share if available; do not estimate a different percentage from memory.
- The supplied values are user-input parameter estimates or tool/reference values. Do not label them measured unless the user explicitly identifies them as measured.
- Do not invent foundry rules, library metrics, cell area, leakage, toggle rates, cache-miss rates, IPC, TOPS, utilization, routing congestion, slack, transistor counts, benchmarks, battery life, or vendor-specific internal formulas.
- The process label is metadata only and must not be used to scale area or power.
- If coverage flags are false, call values partial subtotals and do not declare PPA feasibility.
- Never infer timing closure from the target frequency. If no STA report/slack is supplied, say timing is unverified.
- Never infer die area from core/floorplan area without the explicitly supplied periphery allowance; even then, call it a planning proxy.
- Distinguish measured/tool-reported data, direct user inputs, calculated values, assumptions, and unknowns.
- Dynamic power formula used in physics mode: alpha*Ceff*V^2*f, with Ceff in pF and f in GHz, divided by 1000 to get watts. This is a lumped first-order model; it omits effects not represented in the input data.
- In direct-power mode, direct component power is presumed to be total average power and leakage must not be added again.
- For laptops, distinguish SoC power from whole-system power; battery runtime is an estimate based on user-entered battery capacity, usable fraction, conversion efficiency and non-SoC loads.
- Do not claim this tool replicates Intel, AMD, Qualcomm, or TI internal methods. Public vendor tools rely on device/board/IP models and detailed scenario inputs; use neutral engineering practices only.
- Use only evidence-supported recommendations. If information is insufficient, state exactly what report or parameter is required next.

Required sections:
1. Input quality and confidence (complete vs partial; data types and missing evidence).
2. Power / area budget checks, only if complete data supports them.
3. Architecture assessment, with observations separated from hypotheses.
4. Three prioritized engineering experiments, each with expected qualitative direction, trade-off, and validation method; do not invent numerical gains.
5. Timing, thermal and (for laptops) system-energy interpretation, only where inputs support it.
6. Validation roadmap, in this order where appropriate: workload/activity characterization; synthesis and cell/macro area; STA with Liberty+SDC and parasitics; power estimation with realistic activity; placement/floorplan and routing; package/thermal measurements or validated model.
7. Final verdict with explicit limitations.

Use clear technical English for a third-year ECE/VLSI student. Be concise but technically careful."""

    prompt = (
        "Review this SoC engineering scenario. Do not treat missing parameters as zero-power or zero-area hardware; "
        "the table's zeros may mean inputs have not yet been supplied.\n\n"
        "CRITICAL: Start your Power / Area Budget Checks by accurately reporting the exact statuses and values in "
        "verified_calculations_authoritative. These application-computed checks take precedence over your own verbal inference. "
        "If an area violation is present, you must call it a violation even if a separate planning proxy has a different value. "
        "Do not claim any explicitly supplied input is missing.\n\n"
        "SCENARIO DATA (JSON):\n"
        + json.dumps(analysis_context, indent=2)
        + "\n\nMake recommendations only from the supplied evidence. Identify the most consequential missing inputs needed to improve accuracy."
    )

    if not api_key_input:
        st.warning("No API key entered. The calculation dashboard remains usable, but cloud AI analysis was not run.")
    elif ImagineClient is None or ChatMessage is None:
        st.error("Imagine SDK is unavailable. Install it with: python -m pip install python-imagine-sdk")
    else:
        with st.spinner(f"Requesting engineering review from {model_selection} via Qualcomm AI Inference Cloud..."):
            try:
                client = make_ai_client(api_key_input)
                response = client.chat(
                    messages=[
                        ChatMessage(role="system", content=system_instructions),
                        ChatMessage(role="user", content=prompt),
                    ],
                    model=model_selection,
                )
                analysis_text = getattr(response, "first_content", None)
                if analysis_text:
                    # Always display the deterministic calculator verdict separately
                    # from generated prose, so a model phrasing error cannot obscure it.
                    st.markdown("### Deterministic calculation verdict (from application inputs)")
                    facts_col1, facts_col2 = st.columns(2)
                    with facts_col1:
                        if power_status == "PASS":
                            st.success(f"Power: PASS — {calc['soc_power_w']:.4f} W against {power_budget_w:.4f} W; headroom {power_headroom:.4f} W.")
                        elif power_status == "VIOLATION":
                            st.error(f"Power: VIOLATION — {calc['soc_power_w']:.4f} W against {power_budget_w:.4f} W; exceeds budget by {-power_headroom:.4f} W.")
                        else:
                            st.warning("Power: NOT VERIFIED — complete power coverage is not available.")
                    with facts_col2:
                        if area_status == "PASS":
                            st.success(f"Area: PASS — checked {area_value_for_check:.4f} mm² against {area_budget_mm2:.4f} mm²; headroom {area_headroom:.4f} mm² ({area_constraint_basis}).")
                        elif area_status == "VIOLATION":
                            st.error(f"Area: VIOLATION — checked {area_value_for_check:.4f} mm² against {area_budget_mm2:.4f} mm²; exceeds budget by {-area_headroom:.4f} mm² ({area_constraint_basis}).")
                        else:
                            st.warning("Area: NOT VERIFIED — complete area inputs or the selected area basis are unavailable.")
                    if power_method == "Physics model: alpha-C-V-squared-f + leakage":
                        alpha_text = ", ".join(f"{r['Block']}: α={float(r['Activity alpha']):.4f}" for _, r in active_rows.iterrows())
                        st.caption("Activity factors from active rows: " + alpha_text)
                    if app_profile == "Laptop / Client PC SoC":
                        if battery_runtime_h is not None:
                            st.caption(f"Laptop estimate: {laptop_system_power_w:.4f} W total system power; idealized runtime {battery_runtime_h:.4f} h. This depends on the entered synthetic/user-supplied inputs.")
                        else:
                            st.caption("Laptop runtime not verified because power coverage, non-SoC load coverage, or battery capacity is incomplete.")
                    st.markdown("### AI interpretation")
                    st.markdown(analysis_text)
                    # Heuristic contradiction warnings are advisory; deterministic facts above remain authoritative.
                    response_lower = analysis_text.lower()
                    contradiction_phrases = []
                    if power_status == "VIOLATION" and any(phrase in response_lower for phrase in ("power budget is sufficient", "power budget is met", "meets the power budget", "power budget is satisfied", "power constraints are met")):
                        contradiction_phrases.append("AI prose appears to claim the power budget is met, but the application computes a power violation.")
                    if area_status == "VIOLATION" and any(phrase in response_lower for phrase in ("area budget seems sufficient", "area budget seems to be sufficient", "area budget is sufficient", "area budget is met", "meets the area budget", "area constraints are met", "area budget is satisfied", "area budget appears sufficient", "area budget appears to be sufficient")):
                        contradiction_phrases.append("AI prose appears to claim the area budget is met, but the application computes an area violation.")
                    if (
                        power_method == "Physics model: alpha-C-V-squared-f + leakage"
                        and activity_alpha_nonzero_all_active
                        and "alpha" in response_lower
                        and any(phrase in response_lower for phrase in (
                            "alpha is missing", "alpha value is missing", "alpha value is unspecified",
                            "missing or unspecified", "not provided", "not supplied",
                        ))
                    ):
                        contradiction_phrases.append("AI prose appears to say alpha is missing or unspecified, but nonzero alpha values are supplied for the active rows.")
                    for contradiction in contradiction_phrases:
                        st.error("Consistency warning: " + contradiction)
                else:
                    st.warning("The API call returned no text. Check model availability and the provider API response.")
            except Exception as exc:
                st.error(f"Qualcomm AI Inference Cloud request failed: {exc}")

# Downloadable audit trail for results and explicit source inputs.
st.markdown("## 7. Export / audit")
export_payload = {
    "application": "SoC PPA Advisor — Engineering Workbench",
    "disclaimer": "Parameter-driven planning estimator, not signoff PPA.",
    "scenario": analysis_context,
    "block_input_table": plot_df.to_dict(orient="records"),
    "area_model": area,
    "method": {
        "power_method": power_method,
        "power_formula": "Pdyn_W = alpha * Ceff_pF * V_V^2 * frequency_GHz / 1000",
        "area_formula": "core_footprint = standard_cell_area / utilization + macro_area + user_reserve",
        "process_label_is_metadata_only": True,
    },
}
st.download_button(
    "Download calculation audit JSON",
    data=json.dumps(export_payload, indent=2, default=float),
    file_name="soc_ppa_audit.json",
    mime="application/json",
)

with st.expander("Industrial-method references and next validation tools", expanded=False):
    st.markdown(
        "These public references illustrate the type of scenario data used in engineering power and timing analysis; they are not claimed to be formulas for proprietary CPU/GPU products.\n\n"
        "- [Intel Power and Thermal Calculator guidance](https://www.intel.com/content/www/us/en/docs/programmable/813741/24-3-1/power-and-thermal-calculator.html) — resource usage, operating frequency, toggle rates and environmental inputs.\n"
        "- [AMD Power Design Manager User Guide](https://docs.amd.com/r/en-US/ug1556-power-design-manager) — device selection, activity, voltage/temperature and characterized device models.\n"
        "- [TI AM62x Power Estimation Tool](https://www.ti.com/tool/AM62X-PET-CALC) — estimates built from measured/simulated data and workload/environment inputs; TI notes actual system verification is needed.\n"
        "- [OpenSTA examples](https://opensta.readthedocs.io/en/latest/Examples.html) — timing analysis with Liberty timing models, Verilog netlists, SDC constraints and parasitics.\n\n"
        "For an open-source RTL-to-layout flow, evaluate Yosys + OpenROAD/OpenSTA with a compatible, legally available platform/library. For a real commercial process, use the foundry PDK, licensed libraries, memory compilers, and the relevant signoff tools/flows."
    )

st.caption(
    "Reproducibility note: retain input provenance, workload/scenario, voltage and temperature, library/PDK version, tool versions, "
    "constraints, activity assumptions and report files alongside any engineering conclusion."
)
