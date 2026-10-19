# SoC PPA Advisor — engineering estimator

## Run

```powershell
cd C:\infosys_hackathon\info
python -m pip install streamlit pandas numpy plotly python-imagine-sdk
streamlit run app_qualcomm.py
```

The Qualcomm AI key is read from `INFERENCE_CLOUD_API_KEY` or entered in the sidebar. The optional endpoint override is `INFERENCE_CLOUD_ENDPOINT`, defaulting to `https://aisuite.cirrascale.com/apis/v2`.

## What this version changes

- Removes fabricated process-node density, capacitance, leakage and cost multipliers. The process label is metadata only.
- Removes heuristic core-count-derived single-thread and multi-thread “performance indices.” Clock target is treated as a timing requirement, not proof of timing closure.
- Adds editable architecture templates for laptop/client PC, mobile, embedded, edge AI, datacenter and custom SoCs. Template block counts are placeholders, not specifications of commercial products.
- Supports either (1) a first-order dynamic-power calculation using `Pdynamic = alpha * Ceff * V^2 * f` plus an input leakage estimate, or (2) per-block total average power supplied from a tool/reference/measurement. It does not add leakage a second time in direct-power mode.
- Adds standard-cell area, hard-macro area, utilization and optional floorplan/periphery allowances. A core/floorplan estimate is not called a measured die area.
- Requires explicit confirmation of complete block coverage before reporting budget feasibility; missing block inputs are identified.
- Adds optional STA slack, thermal resistance, laptop non-SoC power, battery energy/runtime estimate, per-block graphs and JSON audit export.
- Adds a normalized CSV input template and uploader. Proprietary raw reports are not parsed automatically; transcribe/normalize valid report data to the table schema and record source, process/library version, PVT and workload.

## Input definitions

- `Ceff pF/instance`: effective capacitance for one instance consistent with the specified activity-factor convention.
- `Activity alpha`: user-supplied activity factor (0–1 in this UI); do not guess this from the process node. Prefer VCD/SAIF or other workload activity analysis when appropriate.
- `Voltage V`, `Frequency GHz`: operating point for that block/domain.
- `Leakage W/instance`: leakage at the relevant characterized process/voltage/temperature and state.
- `Tool total power W/instance`: direct-mode average total power (include exactly what the source report includes; do not double-count leakage).
- `Std-cell area mm2/instance`: synthesis/library-derived standard-cell area.
- `Macro area mm2/instance`: memory/analog/other hard-macro area from IP/compiler data.
- For aggregate report values, set `Instances = 1` and enter the aggregate value in the relevant per-instance field.

## Accuracy limits

This is an auditable early-stage model, not “hyper-accurate” signoff PPA. The model cannot infer valid absolute area or power from a profile or node label. Better accuracy requires the actual RTL/IP, foundry PDK, characterized Liberty libraries, memory compiler outputs, SDC constraints, realistic activity vectors, voltage and temperature conditions, synthesis, static timing analysis, power analysis, placement/route parasitics, and package/thermal context.

The laptop battery-life figure is a workload-specific energy-budget estimate. It depends on average system power, usable battery energy, conversion efficiency and the quality of external-load estimates. It is not a guarantee of runtime.

## Public methodological references

- Intel Power and Thermal Calculator guidance: https://www.intel.com/content/www/us/en/docs/programmable/813741/24-3-1/power-and-thermal-calculator.html
- AMD Power Design Manager User Guide: https://docs.amd.com/r/en-US/ug1556-power-design-manager
- TI AM62x Power Estimation Tool: https://www.ti.com/tool/AM62X-PET-CALC
- OpenSTA examples (Liberty, Verilog, SDC, SPEF/SDF): https://opensta.readthedocs.io/en/latest/Examples.html

These public tools and guides cover their supported product families; they are references for disciplined inputs and validation, not evidence that this app duplicates any vendor's proprietary CPU/GPU estimation algorithms.
