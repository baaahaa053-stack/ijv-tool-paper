"""
IJV Thermal Comfort Tool
Impinging Jet Ventilation · Thermal Comfort Evaluation
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from thermal_model import ThermalModel, BatchSolver, calc_PD

st.set_page_config(
    page_title="IJV Thermal Comfort Tool",
    layout="wide",
)

# ── Fixed PMV parameters ──────────────────────────────────────────────────────
M    = 58.15   # Metabolic rate (W/m²)
W    = 0.0     # Mechanical work (W/m²)
I_cl = 0.124   # Clothing insulation (m²K/W)
RH   = 50.0    # Relative humidity (%)
U    = 0.3     # Air speed in occupied zone (m/s)

# ── Colour palette (HVAC convention) ─────────────────────────────────────────
C_SUPPLY  = "#1a6faf"   # deep blue  — supply / cold
C_FLOOR   = "#4a9fd4"   # mid blue   — near-floor
C_OCC     = "#2e8b57"   # forest green — occupied zone
C_MIXED   = "#c0392b"   # brick red  — mixed / upper zone
C_EXHAUST = "#922b21"   # dark red   — exhaust / ceiling
C_OK      = "#2e8b57"   # green — pass
C_WARN    = "#c0392b"   # red   — fail
C_NEUTRAL = "#7f8c8d"   # gray  — neutral

# ── Helper functions ──────────────────────────────────────────────────────────
def pmv_color(v):
    if abs(v) <= 0.5: return C_OK
    if abs(v) <= 1.0: return "#e67e22"
    return C_WARN

def pmv_label(v):
    if abs(v) <= 0.5: return "✔  Comfortable (ISO 7730 Category B)"
    if abs(v) <= 1.0: return "△  Slightly uncomfortable"
    return "✘  Uncomfortable"

PROFILE_ZONES = [
    ('tc',  3.60, 'Ceiling',        C_EXHAUST),
    ('te',  3.50, 'Exhaust air',    C_EXHAUST),
    ('tmz', 1.80, 'Mixed zone',     C_MIXED),
    ('toz', 0.60, 'Occupied zone',  C_OCC),
    ('tnf', 0.10, 'Near-floor air', C_FLOOR),
    ('tf',  0.00, 'Floor',          C_SUPPLY),
]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    f"<h2 style='text-align:center; margin-bottom:2px; color:#1a3a5c;'>"
    f"IJV Thermal Comfort Tool</h2>"
    f"<p style='text-align:center; color:{C_NEUTRAL}; margin-top:0; font-size:14px;'>"
    f"Impinging Jet Ventilation &nbsp;·&nbsp; 10-Zone Nonlinear Thermal Model"
    f"&nbsp;·&nbsp; ISO 7730 / ASHRAE 55</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Two-column layout ─────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 2], gap="large")

# ════════════════════════════════════════
# Left column — Input parameters
# ════════════════════════════════════════
with col_in:
    st.markdown(f"<span style='font-size:15px; font-weight:600; "
                f"color:#1a3a5c;'>Input Parameters</span>",
                unsafe_allow_html=True)

    mode = st.radio("Mode", ["Single Case", "Batch Solve"],
                    horizontal=True, label_visibility="collapsed")
    single = (mode == "Single Case")
    st.divider()

    # Room geometry
    st.markdown(f"<b style='color:#1a3a5c;'>Room Geometry</b>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    a  = c1.number_input("Length a (m)",  value=5.0,  step=0.5, format="%.1f")
    b  = c2.number_input("Width b (m)",   value=5.0,  step=0.5, format="%.1f")
    hr = c3.number_input("Height hr (m)", value=3.6,  step=0.1, format="%.1f")

    # Occupants
    st.markdown(f"<b style='color:#1a3a5c;'>Occupants</b>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    N  = c1.number_input("N",        value=32,    step=1)
    Ap = c2.number_input("Ap (m²)",  value=2.56,  step=0.1,  format="%.2f")
    Pt = c3.number_input("Pt (W)",   value=120.0, step=10.0, format="%.0f")
    hp = c4.number_input("hp (m)",   value=1.5,   step=0.05, format="%.2f")

    # Supply air
    st.markdown(f"<b style='color:#1a3a5c;'>Supply Air</b>",
                unsafe_allow_html=True)
    S = st.number_input("Nozzle area S (m²)", value=0.170625, format="%.6f")

    if single:
        c1, c2 = st.columns(2)
        ts = c1.number_input("Supply temp ts (°C)", value=18.0,
                             step=0.5, format="%.2f")
        vs = c2.number_input("Supply vel vs (m/s)", value=1.5,
                             step=0.1, format="%.2f")
    else:
        st.markdown(f"<b style='color:#1a3a5c;'>Supply temperature ts (°C)</b>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        ts_min  = c1.number_input("Min",  value=15.0, step=0.5, format="%.2f")
        ts_max  = c2.number_input("Max",  value=22.0, step=0.5, format="%.2f")
        ts_step = c3.number_input("Step", value=1.0,  step=0.5, format="%.2f",
                                  min_value=0.01)

        st.markdown(f"<b style='color:#1a3a5c;'>Supply velocity vs (m/s)</b>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        vs_min  = c1.number_input("Min",  value=1.0, step=0.1, format="%.2f")
        vs_max  = c2.number_input("Max",  value=2.0, step=0.1, format="%.2f")
        vs_step = c3.number_input("Step", value=0.1, step=0.1, format="%.2f",
                                  min_value=0.01)

    st.divider()
    with st.expander("Fixed PMV Parameters"):
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Metabolic rate M | {M} W/m² |
| Mechanical work W | {W} W/m² |
| Clothing insulation I_cl | {I_cl} m²K/W |
| Relative humidity RH | {RH} % |
| Occupied zone air speed U | {U} m/s |
""")

    run = st.button("▶  Calculate", type="primary", use_container_width=True)

# ════════════════════════════════════════
# Right column — Results
# ════════════════════════════════════════
with col_out:
    if not run:
        st.markdown(f"<span style='font-size:15px; font-weight:600; "
                    f"color:#1a3a5c;'>Results</span>", unsafe_allow_html=True)
        st.info("Set parameters on the left and click **▶ Calculate**.")
        st.stop()

    # ── Single case ───────────────────────────────────────────────────────────
    if single:
        model = ThermalModel()
        model.set_case(a, b, hr, Ap, Pt, N, ts, vs, S, hp,
                       M=M, W=W, I_cl=I_cl, U=U, RH=RH)
        res = model.solve()

        if not res['converged']:
            st.error("⚠️ Solver did not converge. Please adjust the input parameters.")
            st.stop()

        toz_v = res['toz'];  pd_v  = calc_PD(toz_v)
        pmv_v = res['PMV'];  ppd_v = res['PPD']
        e_v   = res['E'];    ach_v = model.ACH
        ok = (24 <= toz_v <= 28 and pd_v <= 20 and abs(pmv_v) <= 0.5)

        # Pass / fail banner
        if ok:
            st.success("✔  All comfort criteria satisfied"
                       "  (toz 24–28 °C  ·  PD ≤ 20 %  ·  PMV ± 0.5)")
        else:
            fails = []
            if not (24 <= toz_v <= 28):
                fails.append(f"toz = {toz_v:.2f} °C  ∉  [24, 28]")
            if not (pd_v <= 20):
                fails.append(f"PD = {pd_v:.1f} %  > 20 %")
            if not (abs(pmv_v) <= 0.5):
                fails.append(f"PMV = {pmv_v:.2f}  ∉  [−0.5, 0.5]")
            st.warning("⚠️  Criteria not met:  " + "   ·   ".join(fails))

        st.markdown(f"**{pmv_label(pmv_v)}**"
                    f"&nbsp;&nbsp;&nbsp;ACH = **{ach_v:.1f} h⁻¹**")
        st.divider()

        # KPI cards
        k1, k2, k3, k4, k5 = st.columns(5)
        def kpi(col, label, val, unit="", ok_range=None):
            if ok_range:
                color = (f"color:{C_OK};" if ok_range[0] <= val <= ok_range[1]
                         else f"color:{C_WARN};")
            else:
                color = f"color:#1a3a5c;"
            col.markdown(
                f"<div style='text-align:center; padding:6px 0;'>"
                f"<div style='font-size:11px; color:{C_NEUTRAL}; "
                f"text-transform:uppercase; letter-spacing:0.5px;'>{label}</div>"
                f"<div style='font-size:26px; font-weight:700; {color}'>"
                f"{val:.2f}<span style='font-size:13px;'>{unit}</span></div>"
                f"</div>", unsafe_allow_html=True)

        kpi(k1, "Occ. Zone Temp", toz_v, " °C", (24, 28))
        kpi(k2, "PMV",            pmv_v, "",     (-0.5, 0.5))
        kpi(k3, "PPD",            ppd_v, " %")
        kpi(k4, "Draft PD",       pd_v,  " %",   (0, 20))
        kpi(k5, "Energy Coeff E", e_v,   "")
        st.markdown("")

        # Charts
        g1, g2 = st.columns([3, 2])

        with g1:
            st.markdown(f"<b style='color:#1a3a5c;'>Temperature Profile</b>",
                        unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=[res[z] for z, *_ in PROFILE_ZONES],
                y=[f"{h:.2f} m  {l}" for _, h, l, _ in PROFILE_ZONES],
                orientation='h',
                marker_color=[c for _, _, _, c in PROFILE_ZONES],
                text=[f"{res[z]:.1f} °C" for z, *_ in PROFILE_ZONES],
                textposition='outside',
                width=0.55,
            ))
            xmin = min(res[z] for z, *_ in PROFILE_ZONES) - 1
            xmax = max(res[z] for z, *_ in PROFILE_ZONES) + 3
            fig.update_layout(
                xaxis=dict(title="Temperature (°C)", range=[xmin, xmax],
                           showgrid=True, gridcolor="#e8ecef"),
                yaxis=dict(title="", showgrid=False),
                height=300,
                margin=dict(l=0, r=70, t=10, b=30),
                showlegend=False,
                font=dict(size=12, family="Arial"),
                plot_bgcolor="#f8f9fa",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            st.markdown(f"<b style='color:#1a3a5c;'>PMV Index</b>",
                        unsafe_allow_html=True)
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pmv_v,
                number={'font': {'size': 30, 'family': 'Arial'},
                        'valueformat': '.2f'},
                gauge={
                    'axis': {'range': [-3, 3], 'tickwidth': 1,
                             'tickvals': [-3,-2,-1,0,1,2,3],
                             'tickcolor': C_NEUTRAL},
                    'bar': {'color': pmv_color(pmv_v), 'thickness': 0.22},
                    'bgcolor': 'white',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [-3,   -0.5], 'color': '#d6eaf8'},
                        {'range': [-0.5,  0.5], 'color': '#d5f5e3'},
                        {'range': [0.5,   3  ], 'color': '#fde8e8'},
                    ],
                    'threshold': {
                        'line': {'color': '#1a3a5c', 'width': 2},
                        'thickness': 0.75, 'value': pmv_v,
                    }
                }
            ))
            fig2.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Arial"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Detailed Results"):
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Supply temperature ts | {ts:.2f} °C |
| Supply velocity vs | {vs:.2f} m/s |
| Air change rate ACH | {ach_v:.2f} h⁻¹ |
| Exhaust temperature te | {res['te']:.2f} °C |
| Floor temperature tf | {res['tf']:.2f} °C |
| Ceiling temperature tc | {res['tc']:.2f} °C |
| Near-floor air temperature tnf | {res['tnf']:.2f} °C |
| Mixed zone temperature tmz | {res['tmz']:.2f} °C |
| Energy utilisation coefficient E | {e_v:.3f} |
""")

    # ── Batch mode ────────────────────────────────────────────────────────────
    else:
        ts_list = [round(ts_min + i*ts_step, 4)
                   for i in range(int(round((ts_max-ts_min)/ts_step))+1)]
        vs_list = [round(vs_min + i*vs_step, 4)
                   for i in range(int(round((vs_max-vs_min)/vs_step))+1)]
        total = len(ts_list) * len(vs_list)

        st.markdown(f"<b style='color:#1a3a5c;'>Batch Results — "
                    f"{total} cases</b>", unsafe_allow_html=True)
        bar = st.progress(0)
        rows = []
        for idx, (ts_i, vs_i) in enumerate(
                [(t, v) for t in ts_list for v in vs_list], 1):
            model = ThermalModel()
            model.set_case(a, b, hr, Ap, Pt, N, ts_i, vs_i, S, hp,
                           M=M, W=W, I_cl=I_cl, U=U, RH=RH)
            res   = model.solve()
            toz_v = res['toz']
            pd_v  = calc_PD(toz_v) if res['converged'] else float('nan')
            pmv_v = res['PMV']
            ok    = (res['converged'] and 24 <= toz_v <= 28
                     and pd_v <= 20 and abs(pmv_v) <= 0.5)
            rows.append({'ts': ts_i, 'vs': vs_i, 'ACH': round(model.ACH, 2),
                         'toz': round(toz_v, 3), 'PMV': round(pmv_v, 3),
                         'PPD': round(res['PPD'], 2), 'PD': round(pd_v, 2),
                         'E': round(res['E'], 3), 'te': round(res['te'], 3),
                         'converged': res['converged'], 'qualified': ok})
            bar.progress(idx / total)
        bar.empty()

        df   = pd.DataFrame(rows)
        df_q = df[df['qualified']]
        if len(df_q):
            st.success(f"✔  {len(df_q)} / {total} cases satisfy all criteria")
        else:
            st.warning(f"⚠️  No cases satisfy all criteria ({total} computed)")

        def make_heatmap(col, title, colorscale, zmid=None):
            pivot = df.pivot(index='vs', columns='ts', values=col)
            kw = dict(zmid=zmid) if zmid is not None else {}
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(),
                y=pivot.index.tolist(), colorscale=colorscale,
                text=np.round(pivot.values, 2), texttemplate="%{text}",
                colorbar=dict(title=col), **kw))
            for _, row in df_q.iterrows():
                fig.add_shape(type='rect',
                    x0=row['ts'] - 0.45*ts_step,
                    x1=row['ts'] + 0.45*ts_step,
                    y0=row['vs'] - 0.45*vs_step,
                    y1=row['vs'] + 0.45*vs_step,
                    line=dict(color='#1a3a5c', width=2))
            fig.update_layout(
                title=dict(text=title, font=dict(size=13, color='#1a3a5c')),
                xaxis_title="Supply temperature ts (°C)",
                yaxis_title="Supply velocity vs (m/s)",
                font=dict(family="Arial", size=12),
                paper_bgcolor="rgba(0,0,0,0)",
                height=420)
            return fig

        tab1, tab2, tab3, tab4 = st.tabs([
            "Occupied Zone Temp", "PMV", "Energy Coeff E", "Data Table"])
        with tab1:
            st.plotly_chart(make_heatmap('toz',
                'Occupied Zone Temperature toz (°C)', 'RdYlGn_r'),
                use_container_width=True)
            st.caption("Dark blue border = all criteria satisfied")
        with tab2:
            st.plotly_chart(make_heatmap('PMV', 'PMV Index',
                'RdBu_r', zmid=0), use_container_width=True)
        with tab3:
            st.plotly_chart(make_heatmap('E',
                'Energy Utilisation Coefficient E', 'Blues'),
                use_container_width=True)
        with tab4:
            st.dataframe(df, use_container_width=True, height=380)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️  Download CSV", csv,
                               file_name="ijv_results.csv", mime="text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "IJV Thermal Comfort Tool  ·  10-zone nonlinear thermal model  ·  "
    "PMV/PPD: ISO 7730 / ASHRAE 55 Fanger equations  ·  "
    "Comfort criteria: toz 24–28 °C  |  Draft PD ≤ 20 %  |  PMV ∈ [−0.5, 0.5]"
)
