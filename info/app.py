import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from imagine import ChatMessage, ImagineClient

# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="SoC PPA Advisor | Semiconductor Architecture Optimizer",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Power, Performance, Area (PPA) & SoC Architectural Advisor")
st.caption("AI-assisted decision-support system for early-stage semiconductor design space exploration.")

# ---------------------------------------------------------
# 1. SIDEBAR: API KEY & SYSTEM CONSTRAINTS
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔑 Engine Settings")
    api_key_input = st.text_input(
        "Qualcomm Inference Cloud API Key",
        type="password",
        value=os.environ.get("INFERENCE_CLOUD_API_KEY", ""),
        help="Enter the API key from your Cirrascale Qualcomm AI Inference Cloud dashboard."
    )

    st.header("🧠 AI Model")
    model_selection = st.selectbox(
        "Model for architectural analysis",
        [
            "DeepSeek-R1-Distill-Llama-70B",
            "Qwen-QwQ-32B",
            "Llama-3.3-70B",
            "Llama-3.1-8B",
        ],
        index=0,
        help=(
            "Default: DeepSeek-R1-Distill-Llama-70B for reasoning-heavy analysis. "
            "Model names must match those exposed in your Cirrascale portal."
        ),
    )
    st.caption("Recommended default: DeepSeek R1 70B. Validate quality on your own SoC cases.")
    
    st.header("🎯 Workload & Target Application")
    app_profile = st.selectbox(
        "Application Profile",
        [
            "Wearable / Ultra-Low Power IoT",
            "Edge AI Vision / Robotics",
            "Mobile Flagship Subsystem",
            "High-Performance Datacenter Compute"
        ]
    )

    st.header("📐 PPA Design Constraints")
    max_power_budget = st.slider("Max Power Budget (Watts)", min_value=0.5, max_value=250.0, value=15.0, step=0.5)
    max_area_budget = st.slider("Max Silicon Die Area (mm²)", min_value=5.0, max_value=600.0, value=90.0, step=5.0)
    target_frequency = st.slider("Target Frequency (GHz)", min_value=0.5, max_value=4.5, value=2.2, step=0.1)
    
    st.header("🛠️ Technology Node Selection")
    node_selection = st.selectbox(
        "Process Node Strategy",
        ["Auto-Optimize (Best PPA)", "3nm FinFET/GAA", "5nm FinFET", "7nm FinFET", "16nm FinFET", "28nm Planar"]
    )

# ---------------------------------------------------------
# 2. DETERMINISTIC PPA SIMULATION & SCALING ENGINE
# ---------------------------------------------------------
NODE_PARAMS = {
    "3nm FinFET/GAA": {"density_factor": 2.8, "cap_factor": 0.65, "leakage_ratio": 0.28, "cost_idx": 4.5},
    "5nm FinFET":     {"density_factor": 2.0, "cap_factor": 0.78, "leakage_ratio": 0.22, "cost_idx": 3.2},
    "7nm FinFET":     {"density_factor": 1.4, "cap_factor": 0.90, "leakage_ratio": 0.18, "cost_idx": 2.2},
    "16nm FinFET":    {"density_factor": 1.0, "cap_factor": 1.15, "leakage_ratio": 0.12, "cost_idx": 1.5},
    "28nm Planar":    {"density_factor": 0.55, "cap_factor": 1.45, "leakage_ratio": 0.08, "cost_idx": 1.0}
}

def simulate_soc_ppa(app_type, p_limit, a_limit, f_target, selected_node):
    """
    Computes physical resource allocation, dynamic/static power, and area trade-offs.
    """
    # Auto-resolve process node if set to Auto
    if selected_node == "Auto-Optimize (Best PPA)":
        if p_limit <= 3.0 or a_limit <= 25.0:
            target_node = "3nm FinFET/GAA"
        elif p_limit <= 25.0:
            target_node = "5nm FinFET"
        elif p_limit <= 65.0:
            target_node = "7nm FinFET"
        else:
            target_node = "5nm FinFET" if a_limit <= 200 else "7nm FinFET"
    else:
        target_node = selected_node

    n_data = NODE_PARAMS[target_node]
    density = n_data["density_factor"]
    c_factor = n_data["cap_factor"]
    leak_ratio = n_data["leakage_ratio"]

    # Baseline core allocation heuristic based on target application profile
    if "Wearable" in app_type:
        core_type = "Efficiency-Optimized (Cortex-M/A55 class)"
        base_cores = max(2, min(4, int(p_limit / 0.4)))
        l3_cache_mb = 1.0
        npu_tops = 0.5 if p_limit > 1.5 else 0.0
    elif "Edge AI" in app_type:
        core_type = "Heterogeneous (2 Big + 4 LITTLE)"
        base_cores = max(4, min(8, int(p_limit / 1.5)))
        l3_cache_mb = 4.0
        npu_tops = min(16.0, p_limit * 1.2)
    elif "Mobile" in app_type:
        core_type = "Tri-cluster (1 Prime + 3 Perf + 4 Eff)"
        base_cores = 8
        l3_cache_mb = 8.0
        npu_tops = min(45.0, p_limit * 2.0)
    else: # Datacenter
        core_type = "High-Performance Compute Tiles"
        base_cores = max(16, min(128, int(p_limit / 1.8)))
        l3_cache_mb = base_cores * 1.5
        npu_tops = min(300.0, p_limit * 2.5)

    # Area estimation breakdown (mm²)
    core_area = (base_cores * 3.5) / density
    cache_area = (l3_cache_mb * 1.8) / density
    npu_area = (npu_tops * 0.9) / density
    io_noc_area = (12.0 + (base_cores * 0.2)) / (density ** 0.5)
    total_area = core_area + cache_area + npu_area + io_noc_area

    # Power estimation breakdown (W): dynamic = alpha * C * V^2 * f, static = leak_ratio * dynamic
    v_dd = 0.75 if "3nm" in target_node or "5nm" in target_node else 0.9
    dyn_core_power = base_cores * 0.35 * c_factor * (v_dd ** 2) * (f_target / 1.5)
    dyn_cache_power = (l3_cache_mb * 0.05) * (f_target / 1.5)
    dyn_npu_power = (npu_tops * 0.12) * c_factor
    dynamic_power = dyn_core_power + dyn_cache_power + dyn_npu_power
    static_power = dynamic_power * leak_ratio
    total_power = dynamic_power + static_power

    # Compute single-thread & multi-thread performance indices
    perf_single = round(f_target * 1.25, 2)
    perf_multi = round((base_cores ** 0.82) * f_target * (1.1 if npu_tops > 0 else 1.0), 2)

    # Feasibility checks against hard constraints
    power_violation = total_power > p_limit
    area_violation = total_area > a_limit

    return {
        "node": target_node,
        "core_architecture": core_type,
        "core_count": base_cores,
        "l3_cache_mb": l3_cache_mb,
        "npu_tops": npu_tops,
        "total_area_mm2": round(total_area, 2),
        "total_power_w": round(total_power, 2),
        "dynamic_power_w": round(dynamic_power, 2),
        "static_power_w": round(static_power, 2),
        "perf_single": perf_single,
        "perf_multi": perf_multi,
        "power_violation": power_violation,
        "area_violation": area_violation,
        "power_delta": round(p_limit - total_power, 2),
        "area_delta": round(a_limit - total_area, 2),
        "breakdown_area": {"Cores": core_area, "L3 Cache": cache_area, "NPU/Acc": npu_area, "NoC/IO": io_noc_area},
        "breakdown_power": {"Cores": dyn_core_power, "Cache": dyn_cache_power, "NPU": dyn_npu_power, "Leakage": static_power}
    }

metrics = simulate_soc_ppa(app_profile, max_power_budget, max_area_budget, target_frequency, node_selection)

# ---------------------------------------------------------
# 3. DASHBOARD METRICS DISPLAY & VIOLATION WARNINGS
# ---------------------------------------------------------
if metrics["power_violation"] or metrics["area_violation"]:
    st.error(
        f"⚠️ **Constraint Violation Detected!** "
        f"{'Power exceeded by ' + str(abs(metrics['power_delta'])) + 'W. ' if metrics['power_violation'] else ''}"
        f"{'Area exceeded by ' + str(abs(metrics['area_delta'])) + 'mm². ' if metrics['area_violation'] else ''}"
        "Optimization required to avoid timing closure or thermal throttling issues."
    )
else:
    st.success("✅ **Feasible Design Space Located**: Configuration satisfies all hard thermal and area boundaries.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fabrication Node", metrics["node"])
c2.metric("Total Compute Units", f"{metrics['core_count']} Cores", f"{metrics['core_architecture']}")
c3.metric("Silicon Die Footprint", f"{metrics['total_area_mm2']} mm²", f"Headroom: {metrics['area_delta']} mm²")
c4.metric("Total Package Power", f"{metrics['total_power_w']} W", f"Headroom: {metrics['power_delta']} W")

# ---------------------------------------------------------
# 4. PPA VISUALIZATIONS & RESOURCE ALLOCATION BREAKDOWN
# ---------------------------------------------------------
st.markdown("---")
col_chart_1, col_chart_2 = st.columns(2)

with col_chart_1:
    st.subheader("Silicon Die Area Distribution (mm²)")
    fig_area = go.Figure(data=[go.Pie(
        labels=list(metrics["breakdown_area"].keys()),
        values=list(metrics["breakdown_area"].values()),
        hole=0.45,
        marker=dict(colors=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"])
    )])
    fig_area.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
    st.plotly_chart(fig_area, use_container_width=True)

with col_chart_2:
    st.subheader("Power Dissipation Breakdown (W)")
    fig_power = go.Figure(data=[go.Bar(
        x=list(metrics["breakdown_power"].keys()),
        y=list(metrics["breakdown_power"].values()),
        marker_color=["#2b5c8f", "#d95f02", "#7570b3", "#e7298a"]
    )])
    fig_power.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        height=300,
        yaxis_title="Power (Watts)"
    )
    st.plotly_chart(fig_power, use_container_width=True)

# ---------------------------------------------------------
# 5. LLM-POWERED SILICON ARCHITECTURE ADVISOR
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🧠 Senior Silicon Architect Recommendation")

advisor_prompt = f"""
You are a senior semiconductor architect specializing in SoC architecture,
low-power VLSI, RTL design, physical design, and Power-Performance-Area (PPA)
optimization.

Your task is to critically evaluate the following simulated SoC configuration.
Produce technically rigorous, evidence-based recommendations suitable for an
ECE/VLSI engineering project.

IMPORTANT ENGINEERING RULES:
1. Treat the numerical values below as outputs of a simplified heuristic
   simulator, NOT measured silicon results or signoff-quality EDA results.
2. Never invent measurements, technology specifications, utilization values,
   cache miss rates, timing slack, transistor counts, manufacturing costs,
   battery characteristics, or benchmark results.
3. Clearly distinguish:
   - Calculated facts supplied in the configuration.
   - Engineering interpretations inferred from those facts.
   - Assumptions or hypotheses requiring further validation.
4. Do not claim that the design violates a constraint unless the supplied
   metrics or explicitly stated requirements demonstrate that violation.
5. The configured hard constraints are the power and area limits. Frequency
   is a requested target, NOT proof of timing closure.
6. Do not introduce arbitrary thermal limits, battery-life requirements,
   or other constraints that were not supplied.
7. Do not claim that a process node, architecture, or optimization guarantees
   a particular power, area, leakage, or performance improvement without
   supporting data.
8. Avoid false precision. Never present illustrative estimates as verified
   engineering results.
9. Explain relevant limitations in the model, including that actual PPA depends
   on RTL, libraries, voltage, timing constraints, memory macros, synthesis,
   placement, routing, and workload.
10. Be critical but constructive. If evidence is insufficient to diagnose a
    bottleneck, state what additional information is required.
11. Do not describe heuristic area categories as physically measured floorplan regions. In      particular, "NoC/IO" is a combined modeled category, not a verified pad-ring, bump-array, or routed interconnect area.
12. When reporting area breakdowns, calculate component percentages from the supplied component values and the supplied total area. Verify that the component areas sum to the reported total within rounding tolerance.
13. Do not attribute the modeled core area to actual Cortex-M, Cortex-A55, or other commercial processor implementations. The core architecture label is descriptive; the area formula is a simplified heuristic.
14. Do not claim that a specific component is physically oversized, underutilized, or responsible for congestion without corresponding implementation or utilization data.
15. Check numerical consistency before drawing conclusions. If a derived value does not agree with the supplied data, identify the discrepancy instead of silently accepting it.
    
    

DESIGN CONFIGURATION

Application: {app_profile}

Configured constraints:
- Maximum power: {max_power_budget} W
- Maximum die area: {max_area_budget} mm²
- Target frequency: {target_frequency} GHz

Simulated architecture:
- Process node: {metrics['node']}
- Core architecture: {metrics['core_architecture']}
- Core count: {metrics['core_count']}
- L3 cache: {metrics['l3_cache_mb']} MB
- NPU performance: {metrics['npu_tops']} TOPS
- Estimated total die area: {metrics['total_area_mm2']} mm²
- Estimated total power: {metrics['total_power_w']} W
- Estimated dynamic power: {metrics['dynamic_power_w']} W
- Estimated static leakage power: {metrics['static_power_w']} W
- Single-thread performance index: {metrics['perf_single']}
- Multi-thread performance index: {metrics['perf_multi']}

Constraint evaluation:
- Power violation reported by simulator: {metrics['power_violation']}
- Area violation reported by simulator: {metrics['area_violation']}
- Power headroom: {metrics['power_delta']} W
- Area headroom: {metrics['area_delta']} mm²

AREA BREAKDOWN:
{metrics['breakdown_area']}

POWER BREAKDOWN:
{metrics['breakdown_power']}

REQUIRED RESPONSE FORMAT

1. PPA FEASIBILITY
- State whether the simulated design meets the configured power and area
  constraints, using the supplied violation flags and numerical values.
- Report the remaining power and area headroom.
- Explicitly state that timing closure has not been verified.

2. ARCHITECTURAL ASSESSMENT
- Evaluate the balance between compute, cache, and acceleration resources.
- Identify only bottlenecks supported by the available data.
- If cache contention, dark silicon, leakage dominance, thermal issues, or
  memory bandwidth problems cannot be established, describe them as
  possibilities to investigate rather than confirmed problems.
- Interpret dynamic and static power using the supplied estimates.

3. RECOMMENDATIONS
Provide three prioritized recommendations.
For each recommendation, specify:
- The proposed change.
- The engineering reason.
- The expected direction of impact on power, performance, and area.
- The trade-off or downside.
- How the change could be validated.

Do not invent numerical improvements. If the expected magnitude is unknown,
say that synthesis, workload analysis, or physical-design experiments are needed.

4. VALIDATION PLAN
List the most relevant next steps, such as RTL synthesis, static timing analysis,
memory subsystem evaluation, power estimation, and physical-design analysis.
Explain which uncertainties each step would resolve.

5. FINAL VERDICT
Finish with:
- Feasibility: Meets constraints / Violates constraints.
- Primary concern: The most important evidence-supported concern.
- Next action: One specific engineering experiment to perform.

STYLE:
- Use clear technical English appropriate for a third-year ECE student studying
  VLSI design.
- Use concise sections, bullets, and bold key terms.
- Explain specialist terms briefly when first introduced.
- Avoid dramatic claims, unsupported statistics, and unnecessary jargon.
- Keep the response focused on this specific configuration.
"""

def render_response_graphs():
    """Render plots from the local heuristic simulator, never from guessed LLM numbers."""
    st.markdown("---")
    st.subheader("📊 Supporting Graphs for This Analysis")
    st.caption(
        "All values below are generated by the local heuristic simulator. "
        "They are exploratory estimates, not synthesis, STA, or signoff measurements."
    )

    power_use_pct = (metrics["total_power_w"] / max_power_budget * 100.0) if max_power_budget else 0.0
    area_use_pct = (metrics["total_area_mm2"] / max_area_budget * 100.0) if max_area_budget else 0.0
    utilization = pd.DataFrame({
        "Resource": ["Power budget", "Area budget"],
        "Estimated utilization (%)": [power_use_pct, area_use_pct],
    })
    fig_util = go.Figure()
    fig_util.add_trace(go.Bar(
        x=utilization["Resource"],
        y=utilization["Estimated utilization (%)"],
        name="Estimated utilization",
        text=[f"{v:.1f}%" for v in utilization["Estimated utilization (%)"]],
        textposition="outside",
    ))
    fig_util.add_hline(y=100, line_dash="dash", annotation_text="Configured limit (100%)")
    fig_util.update_layout(
        title="Power and Area Budget Utilization",
        yaxis_title="Budget used (%)",
        yaxis_rangemode="tozero",
        margin=dict(t=60, b=30, l=30, r=20),
        height=360,
    )
    st.plotly_chart(fig_util, use_container_width=True)

    col_node, col_freq = st.columns(2)
    with col_node:
        node_rows = []
        for node_name in NODE_PARAMS:
            node_metric = simulate_soc_ppa(
                app_profile, max_power_budget, max_area_budget, target_frequency, node_name
            )
            node_rows.append({
                "Node": node_name,
                "Estimated area (mm²)": node_metric["total_area_mm2"],
                "Estimated power (W)": node_metric["total_power_w"],
            })
        node_df = pd.DataFrame(node_rows)
        fig_nodes = go.Figure()
        fig_nodes.add_trace(go.Scatter(
            x=node_df["Estimated area (mm²)"],
            y=node_df["Estimated power (W)"],
            mode="markers+text",
            text=node_df["Node"],
            textposition="top center",
            marker=dict(size=11),
            name="Modeled process-node options",
            customdata=node_df[["Node"]],
            hovertemplate=(
                "Node: %{customdata[0]}<br>Area: %{x:.2f} mm²<br>Power: %{y:.2f} W<extra></extra>"
            ),
        ))
        fig_nodes.add_vline(x=max_area_budget, line_dash="dash", annotation_text="Area limit")
        fig_nodes.add_hline(y=max_power_budget, line_dash="dash", annotation_text="Power limit")
        fig_nodes.update_layout(
            title="Process-Node Sensitivity (Heuristic)",
            xaxis_title="Estimated area (mm²)",
            yaxis_title="Estimated power (W)",
            margin=dict(t=60, b=35, l=35, r=20),
            height=420,
        )
        st.plotly_chart(fig_nodes, use_container_width=True)

    with col_freq:
        frequency_values = np.round(np.linspace(0.5, 4.5, 17), 2)
        sweep_rows = []
        for freq in frequency_values:
            freq_metric = simulate_soc_ppa(
                app_profile, max_power_budget, max_area_budget, float(freq), node_selection
            )
            sweep_rows.append({
                "Target frequency (GHz)": float(freq),
                "Estimated power (W)": freq_metric["total_power_w"],
            })
        sweep_df = pd.DataFrame(sweep_rows)
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(
            x=sweep_df["Target frequency (GHz)"],
            y=sweep_df["Estimated power (W)"],
            mode="lines+markers",
            name="Estimated power",
        ))
        fig_freq.add_hline(y=max_power_budget, line_dash="dash", annotation_text="Power limit")
        fig_freq.add_vline(
            x=target_frequency,
            line_dash="dot",
            annotation_text=f"Current target: {target_frequency:.1f} GHz",
        )
        fig_freq.update_layout(
            title="Estimated Power vs Target Frequency",
            xaxis_title="Target frequency (GHz)",
            yaxis_title="Estimated power (W)",
            margin=dict(t=60, b=35, l=35, r=20),
            height=420,
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    st.info(
        "Interpretation note: the process-node and frequency plots expose trends in the "
        "current simplified equations only. They do not prove a real process node will "
        "deliver these PPA values or that timing closes at the requested frequency."
    )


if st.button("Run AI Architectural Trade-off Analysis", type="primary"):
    if not api_key_input:
        st.warning("⚠️ API Key not detected. Displaying local offline architectural assessment.")
        fallback_analysis = f"""
**Offline PPA Summary (AI service not called):**

- **Modelled configuration:** {metrics['node']}, {metrics['core_count']} cores, {metrics['l3_cache_mb']} MB L3 cache, and {metrics['npu_tops']} TOPS modelled NPU throughput.
- **Estimated power:** {metrics['total_power_w']} W against a {max_power_budget} W limit; modelled headroom is {metrics['power_delta']} W.
- **Estimated area:** {metrics['total_area_mm2']} mm² against a {max_area_budget} mm² limit; modelled headroom is {metrics['area_delta']} mm².
- **Limitations:** These figures come from simplified heuristic equations. Timing closure, RTL synthesis, physical area, and signoff power have not been verified. No architecture bottleneck is inferred in this offline fallback.
        """
        st.markdown(fallback_analysis)
        render_response_graphs()
    else:
        with st.spinner(f"Requesting architectural analysis from {model_selection} via Qualcomm AI Inference Cloud..."):
            try:
                endpoint = os.environ.get(
                    "INFERENCE_CLOUD_ENDPOINT",
                    "https://aisuite.cirrascale.com/apis/v2"
                )
                client = ImagineClient(
                    endpoint=endpoint,
                    api_key=api_key_input
                )
                response = client.chat(
                    messages=[
                        ChatMessage(
                            role="user",
                            content=advisor_prompt
                        )
                    ],
                    model=model_selection
                )
                analysis_text = response.first_content
                if analysis_text:
                    st.markdown(analysis_text)
                    render_response_graphs()
                else:
                    st.warning("The Qualcomm AI service returned an empty response. Please try again.")
            except Exception as e:
                st.error(f"Error connecting to Qualcomm AI Inference Cloud: {e}")
                