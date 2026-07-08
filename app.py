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

# ── 减少页面顶部留白 ───────────────────────────────────────────────
st.markdown("""
<style>
/* 主容器顶部/底部内边距 */
.block-container {
    padding-top: 0.3rem !important;
    padding-bottom: 1rem !important;
}
/* Streamlit 默认头部工具栏（右上角菜单栏）压缩高度，并去掉残留的分割线 */
header[data-testid="stHeader"] {
    height: 0rem;
    min-height: 0rem;
    border-bottom: none !important;
    box-shadow: none !important;
}
/* 隐藏右上角的 Deploy/菜单/GitHub 图标，进一步节省空间（不需要可删除这两行） */
#MainMenu {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}

/* 全局字体改为 Times New Roman */
html, body, [class*="css"], .stMarkdown, .stText,
input, textarea, select, button,
div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
    font-family: "Times New Roman", Times, serif !important;
}

/* 缩小 Mode 单选按钮下方的留白，让上下间距一致 */
div[data-testid="stRadio"] {
    margin-bottom: -18px;
}
</style>
""", unsafe_allow_html=True)

# ── Fixed PMV parameters ──────────────────────────────────────────────────────
M    = 58.15   # Metabolic rate (W/m²)
W    = 0.0     # Mechanical work (W/m²)
I_cl = 0.124   # Clothing insulation (m²K/W)
RH   = 50.0    # Relative humidity (%)
U    = 0.3     # Air speed in occupied zone (m/s)

# ── Colour palette (engineering psychrometric chart convention) ──────────────
C_SUPPLY  = "#0000FF"   # pure blue   — supply / cold
C_FLOOR   = "#4499FF"   # light blue  — near-floor air
C_OCC     = "#00AA44"   # green       — occupied zone
C_MIXED   = "#FF6600"   # orange      — mixed zone
C_EXHAUST = "#FF0000"   # pure red    — exhaust / ceiling
C_OK      = "#00AA44"   # green — pass
C_WARN    = "#FF0000"   # red   — fail
C_NEUTRAL = "#555555"   # gray  — neutral

# ── Helper functions ──────────────────────────────────────────────────────────
def pmv_color(v):
    if abs(v) <= 0.5: return C_OK
    if abs(v) <= 1.0: return "#e67e22"
    return C_WARN

def pmv_label(v):
    if abs(v) <= 0.5: return "✔  Comfortable (ISO 7730 Category B)"
    if abs(v) <= 1.0: return "△  Slightly uncomfortable"
    return "✘  Uncomfortable"

def labeled_input(col, label_html, key, **kwargs):
    """Render a proper HTML label (supports <sub>, upright, correct size)
    above a number_input whose native label is hidden."""
    col.markdown(
        f"<div style='font-size:14px; margin-bottom:2px; line-height:1.4;'>"
        f"{label_html}</div>", unsafe_allow_html=True)
    return col.number_input(label_html, key=key,
                             label_visibility="collapsed", **kwargs)

# 从高到低排列，颜色由绘图时按温度动态计算
PROFILE_ZONES = [
    ('tc',  3.60, 'Ceiling'),
    ('te',  3.50, 'Exhaust zone'),
    ('tmz', 1.80, 'Mixed zone'),
    ('toz', 0.60, 'Occupied zone'),
    ('tnf', 0.10, 'Floor zone'),
    ('tf',  0.00, 'Floor'),
]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='display:flex; align-items:baseline; gap:14px; "
    f"margin:0; padding:6px 0 8px 0;'>"
    f"<span style='font-size:26px; font-weight:700; color:#003399; "
    f"line-height:1;'>IJV Thermal Comfort Tool</span>"
    f"<span style='color:{C_NEUTRAL}; font-size:13px; line-height:1;'>"
    f"Impinging Jet Ventilation &nbsp;·&nbsp; Four-Zonal Model"
    f"&nbsp;·&nbsp; ISO 7730 / ASHRAE 55-2023</span>"
    f"</div>"
    f"<hr style='margin:0 0 12px 0; border:none; border-top:1px solid #e0e0e0;'>",
    unsafe_allow_html=True,
)

# ── Two-column layout ─────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 2], gap="large")

# ════════════════════════════════════════
# Left column — Input parameters
# ════════════════════════════════════════
with col_in:
    st.markdown(f"<span style='font-size:15px; font-weight:600; "
                f"color:#003399;'>Input Parameters</span>",
                unsafe_allow_html=True)

    mode = st.radio("Mode", ["Single Case", "Batch Solve"],
                    horizontal=True, label_visibility="collapsed")
    single = (mode == "Single Case")
    st.markdown("<hr style='margin:4px 0 12px 0; border:none; "
                "border-top:1px solid #e0e0e0;'>", unsafe_allow_html=True)

    # Room geometry
    st.markdown(f"<b style='color:#003399;'>Room Geometry</b>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    a  = labeled_input(c1, "Length a (m)",  "in_a",  value=5.0,  step=0.5, format="%.1f")
    b  = labeled_input(c2, "Width b (m)",   "in_b",  value=5.0,  step=0.5, format="%.1f")
    hr = labeled_input(c3, "Height h<sub>r</sub> (m)", "in_hr", value=3.6,  step=0.1, format="%.1f")

    # Occupants
    st.markdown(f"<b style='color:#003399;'>Occupants</b>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    N  = labeled_input(c1, "N",             "in_N",  value=32,    step=1)
    Ap = labeled_input(c2, "A<sub>p</sub> (m²)",  "in_Ap", value=2.56,  step=0.1,  format="%.2f")
    Pt = labeled_input(c3, "P<sub>t</sub> (W)",   "in_Pt", value=120.0, step=10.0, format="%.0f")
    hp = labeled_input(c4, "h<sub>p</sub> (m)",   "in_hp", value=1.5,   step=0.05, format="%.2f")

    # Supply air
    st.markdown(f"<b style='color:#003399;'>Supply Air</b>",
                unsafe_allow_html=True)
    S = st.number_input("Nozzle area S (m²)", value=0.170625, format="%.6f")

    if single:
        c1, c2 = st.columns(2)
        ts = labeled_input(c1, "Supply temp t<sub>s</sub> (°C)", "in_ts",
                           value=18.0, step=0.5, format="%.2f")
        vs = labeled_input(c2, "Supply vel v<sub>s</sub> (m/s)", "in_vs",
                           value=1.5, step=0.1, format="%.2f")
    else:
        st.markdown(f"<b style='color:#003399;'>Supply temperature t<sub>s</sub> (°C)</b>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        ts_min  = c1.number_input("Min",  value=15.0, step=0.5, format="%.2f")
        ts_max  = c2.number_input("Max",  value=22.0, step=0.5, format="%.2f")
        ts_step = c3.number_input("Step", value=1.0,  step=0.5, format="%.2f",
                                  min_value=0.01)

        st.markdown(f"<b style='color:#003399;'>Supply velocity v<sub>s</sub> (m/s)</b>",
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
                    f"color:#003399;'>Results</span>", unsafe_allow_html=True)
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
        st.markdown("<hr style='margin:4px 0 12px 0; border:none; "
                    "border-top:1px solid #e0e0e0;'>", unsafe_allow_html=True)

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
                f"{val:.2f}<span style='font-size:26px;'>{unit}</span></div>"
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
            st.markdown(f"<b style='color:#003399;'>Temperature Profile</b>",
                        unsafe_allow_html=True)

            # 关键高度节点：Floor(0) - Near-floor(0.1) - Occupied(2.0) -
            # Mixed/Exhaust boundary(hr-0.1) - Ceiling(hr)
            H_01  = 0.10
            H_20  = 2.0
            H_35  = hr - 0.10   # exhaust zone 下边界
            H_top = hr

            # 各区中心高度（地板/天花板取面上，其余取区间中点）
            Y_floor    = 0.0
            Y_floorz   = H_01 / 2
            Y_occ      = (H_01 + H_20) / 2
            Y_mixed    = (H_20 + H_35) / 2
            Y_exhaust  = (H_35 + H_top) / 2
            Y_ceiling  = H_top

            nodes = [
                ('tf',  Y_floor,   res['tf'],  'T_f'),
                ('tnf', Y_floorz,  res['tnf'], 'T_0.1'),
                ('toz', Y_occ,     res['toz'], 'T_2.0'),
                ('tmz', Y_mixed,   res['tmz'], 'T_3.5'),
                ('te',  Y_exhaust, res['te'],  'T_e'),
                ('tc',  Y_ceiling, res['tc'],  'T_c'),
            ]
            # 折线节点顺序：tf -> tnf -> toz -> tmz -> te -> tc
            line_x = [res['tf'], res['tnf'], res['toz'], res['tmz'], res['te'], res['tc']]
            line_y = [Y_floor, Y_floorz, Y_occ, Y_mixed, Y_exhaust, Y_ceiling]

            t_min, t_max = min(line_x), max(line_x)
            # 蓝 → 绿 → 黄 → 红 四段渐变（对标参考图配色）
            _stops = [
                (0.00, (0, 102, 255)),    # 蓝
                (0.33, (0, 200, 90)),     # 绿
                (0.66, (255, 220, 0)),    # 黄
                (1.00, (230, 30, 20)),    # 红
            ]
            def _t2color(t):
                r = 0.5 if t_max == t_min else (t - t_min) / (t_max - t_min)
                r = min(1, max(0, r))
                for i in range(len(_stops) - 1):
                    f0, c0 = _stops[i]
                    f1, c1 = _stops[i+1]
                    if f0 <= r <= f1:
                        local = 0 if f1 == f0 else (r - f0) / (f1 - f0)
                        rr = int(c0[0] + (c1[0]-c0[0]) * local)
                        gg = int(c0[1] + (c1[1]-c0[1]) * local)
                        bb = int(c0[2] + (c1[2]-c0[2]) * local)
                        return f"rgb({rr},{gg},{bb})"
                return f"rgb({_stops[-1][1][0]},{_stops[-1][1][1]},{_stops[-1][1][2]})"

            fig = go.Figure()

            # 背景分区底色（浅色，不挡字）
            fig.add_shape(type='rect', x0=0, x1=1, y0=0, y1=H_01,
                          xref='paper', fillcolor='#dceeff', line=dict(width=0), layer='below')
            fig.add_shape(type='rect', x0=0, x1=1, y0=H_01, y1=H_20,
                          xref='paper', fillcolor='#e8f9ee', line=dict(width=0), layer='below')
            fig.add_shape(type='rect', x0=0, x1=1, y0=H_20, y1=H_35,
                          xref='paper', fillcolor='#fffbe0', line=dict(width=0), layer='below')
            fig.add_shape(type='rect', x0=0, x1=1, y0=H_35, y1=H_top,
                          xref='paper', fillcolor='#ffe3e0', line=dict(width=0), layer='below')

            # 渐变折线：逐段画线，颜色随两端温度渐变
            for i in range(len(line_x) - 1):
                seg_c0 = _t2color(line_x[i])
                seg_c1 = _t2color(line_x[i+1])
                n_seg = 12
                for j in range(n_seg):
                    f0, f1 = j/n_seg, (j+1)/n_seg
                    xs = [line_x[i] + (line_x[i+1]-line_x[i])*f0,
                          line_x[i] + (line_x[i+1]-line_x[i])*f1]
                    ys = [line_y[i] + (line_y[i+1]-line_y[i])*f0,
                          line_y[i] + (line_y[i+1]-line_y[i])*f1]
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode='lines',
                        line=dict(color=_t2color((xs[0]+xs[1])/2), width=4),
                        showlegend=False, hoverinfo='skip',
                    ))

            # 节点圆点 + 标注（标注符号名，偏移方向避开折线，避免文字压线）
            # tf: 移到点的右侧、更靠近点；tc: 位置略微下移、更靠近点
            NODE_LABELS = {
                'tf':  ("t<sub>f</sub>",  dict(xanchor='left',   yanchor='middle', xshift=6,  yshift=0)),
                'tnf': ("t<sub>nf</sub>", dict(xanchor='center', yanchor='bottom', xshift=0,  yshift=6)),
                'toz': ("t<sub>oz</sub>", dict(xanchor='right',  yanchor='bottom', xshift=-5, yshift=5)),
                'tmz': ("t<sub>mz</sub>", dict(xanchor='right',  yanchor='bottom', xshift=-5, yshift=5)),
                'te':  ("t<sub>e</sub>",  dict(xanchor='right',  yanchor='bottom', xshift=-5, yshift=1)),
                'tc':  ("t<sub>c</sub>",  dict(xanchor='left',   yanchor='middle', xshift=6,  yshift=0)),
            }
            for z, y, t, _ in nodes:
                fig.add_trace(go.Scatter(
                    x=[t], y=[y], mode='markers',
                    marker=dict(size=6, color=_t2color(t),
                               line=dict(color='white', width=1.5)),
                    showlegend=False,
                    hovertext=f"{t:.2f} °C @ {y:.2f} m", hoverinfo='text',
                ))
                label_text, pos_kw = NODE_LABELS[z]
                fig.add_annotation(
                    x=t, y=y, text=label_text, showarrow=False,
                    font=dict(size=12, color='#333'), **pos_kw,
                )

            # 分区文字标签（放在图右侧，不挡折线）
            zone_label_x = t_max + (t_max - t_min) * 0.18
            for label, y0, y1 in [
                ('Floor zone',    0.0,  H_01),
                ('Occupied zone', H_01, H_20),
                ('Mixed zone',    H_20, H_35),
                ('Exhaust zone',  H_35, H_top),
            ]:
                fig.add_annotation(
                    x=zone_label_x, y=(y0+y1)/2, text=label,
                    showarrow=False, font=dict(size=11, color='#555'),
                    xanchor='left',
                )

            fig.update_layout(
                xaxis=dict(title="Temperature (°C)",
                           range=[t_min - 1, t_max + (t_max-t_min)*0.45 + 2],
                           showgrid=True, gridcolor="#e8ecef"),
                yaxis=dict(title="Height (m)", range=[-0.1, hr + 0.5],
                           showgrid=False, zeroline=False),
                height=440,
                margin=dict(l=10, r=10, t=10, b=40),
                showlegend=False,
                font=dict(size=12, family="Times New Roman"),
                plot_bgcolor="white",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            st.markdown(f"<b style='color:#003399;'>PMV Index</b>",
                        unsafe_allow_html=True)
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pmv_v,
                domain={'x': [0.02, 0.98], 'y': [0, 1]},
                number={'font': {'size': 36, 'family': 'Times New Roman'},
                        'valueformat': '.2f'},
                gauge={
                    'axis': {'range': [-3, 3], 'tickwidth': 1,
                             'tickvals': [-3,-2,-1,0,1,2,3],
                             'tickcolor': C_NEUTRAL,
                             'tickfont': {'size': 13}},
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
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Times New Roman"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Detailed Results"):
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Supply temperature t<sub>s</sub> | {ts:.2f} °C |
| Supply velocity v<sub>s</sub> | {vs:.2f} m/s |
| Air change rate ACH | {ach_v:.2f} h⁻¹ |
| Exhaust zone temperature t<sub>e</sub> | {res['te']:.2f} °C |
| Floor temperature t<sub>f</sub> | {res['tf']:.2f} °C |
| Ceiling temperature t<sub>c</sub> | {res['tc']:.2f} °C |
| Floor zone temperature t<sub>nf</sub> | {res['tnf']:.2f} °C |
| Mixed zone temperature t<sub>mz</sub> | {res['tmz']:.2f} °C |
| Energy utilisation coefficient E | {e_v:.3f} |
""", unsafe_allow_html=True)

    # ── Batch mode ────────────────────────────────────────────────────────────
    else:
        ts_list = [round(ts_min + i*ts_step, 4)
                   for i in range(int(round((ts_max-ts_min)/ts_step))+1)]
        vs_list = [round(vs_min + i*vs_step, 4)
                   for i in range(int(round((vs_max-vs_min)/vs_step))+1)]
        total = len(ts_list) * len(vs_list)

        st.markdown(f"<b style='color:#003399;'>Batch Results — "
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
                         'tf': round(res['tf'], 3), 'tnf': round(res['tnf'], 3),
                         'toz': round(toz_v, 3), 'tmz': round(res['tmz'], 3),
                         'te': round(res['te'], 3), 'tc': round(res['tc'], 3),
                         'PMV': round(pmv_v, 3),
                         'PPD': round(res['PPD'], 2), 'PD': round(pd_v, 2),
                         'E': round(res['E'], 3),
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
                    line=dict(color='#003399', width=2))
            fig.update_layout(
                title=dict(text=title, font=dict(size=13, color='#003399')),
                xaxis_title="Supply temperature t<sub>s</sub> (°C)",
                yaxis_title="Supply velocity v<sub>s</sub> (m/s)",
                font=dict(family="Times New Roman", size=12),
                paper_bgcolor="rgba(0,0,0,0)",
                height=420)
            return fig

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Occupied Zone Temp", "PMV", "PPD", "Draft PD", "Energy Coeff E", "Data Table"])
        with tab1:
            st.plotly_chart(make_heatmap('toz',
                'Occupied Zone Temperature t<sub>oz</sub> (°C)', 'RdYlGn_r'),
                use_container_width=True)
            st.caption("Dark blue border = all criteria satisfied")
        with tab2:
            st.plotly_chart(make_heatmap('PMV', 'PMV Index',
                'RdBu_r', zmid=0), use_container_width=True)
        with tab3:
            st.plotly_chart(make_heatmap('PPD', 'PPD (%)',
                'YlOrRd'), use_container_width=True)
        with tab4:
            st.plotly_chart(make_heatmap('PD', 'Draft PD (%)',
                'YlOrRd'), use_container_width=True)
        with tab5:
            st.plotly_chart(make_heatmap('E',
                'Energy Utilisation Coefficient E', 'Blues'),
                use_container_width=True)
        with tab6:
            st.dataframe(df, use_container_width=True, height=380)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️  Download CSV", csv,
                               file_name="ijv_results.csv", mime="text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "IJV Thermal Comfort Tool  ·  four-zonal model  ·  "
    "PMV/PPD: ISO 7730 / ASHRAE 55-2023  ·  "
    "Comfort criteria: toz 24–28 °C  |  Draft PD ≤ 20 %  |  PMV ∈ [−0.5, 0.5]"
)
