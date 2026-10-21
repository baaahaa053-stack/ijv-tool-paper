"""
IJV Thermal Comfort Tool
Impinging Jet Ventilation · Thermal Comfort Evaluation
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time   #  界面显示实测计算时间
from thermal_model import (ThermalModel, BatchSolver, calc_PD,
                           check_validity_domain, CALIB_RANGE,   
                           pmv_category)                          

#  图片高分辨率下载：绘图工具栏的“下载图片”导出 3 倍分辨率 PNG
PLOT_CONFIG = {'toImageButtonOptions': {'format': 'png', 'scale': 3}}

st.set_page_config(
    page_title="IJV Thermal Comfort Tool",
    layout="wide",
)

# ── 减少页面顶部留白 ───────────────────────────────────────────────
st.markdown("""
<style>
/* 主容器顶部/底部内边距 */
.block-container {
    padding-top: 0rem !important;
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

# ── Default PMV parameters ────────────────────────────────────────────────────
#  原“固定参数”改为默认值，并在左侧面板中可由用户编辑
#   （M、W、I_cl、RH、U 的实际取值见左侧 “PMV Parameters” 面板）。默认值保持不变。
DEFAULT_M    = 58.15   # Metabolic rate (W/m²) = 1.0 met
DEFAULT_W    = 0.0     # Mechanical work (W/m²)
DEFAULT_I_cl = 0.124   # Clothing insulation (m²K/W) = 0.8 clo
DEFAULT_RH   = 50.0    # Relative humidity (%)
DEFAULT_U    = 0.3     # Air speed in occupied zone (m/s)

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
    #  增加 ISO 7730 Category C（|PMV| ≤ 0.7）这一档
    if abs(v) <= 0.5: return C_OK
    if abs(v) <= 0.7: return "#c8b400"
    if abs(v) <= 1.0: return "#e67e22"
    return C_WARN

def pmv_label(v):
    #  统一一套标签与阈值，同时给出 ASHRAE 55 与 ISO 7730 等级：
    #   |PMV| ≤ 0.5 : Comfortable        （ASHRAE 55 满足；ISO 7730 Cat. A 或 B）
    #   0.5–0.7     : Acceptable (ISO Cat. C only)（ASHRAE 55 不满足）
    #   0.7–1.0     : Slightly uncomfortable
    #   > 1.0       : Uncomfortable
    ashrae_ok, iso = pmv_category(v)
    if abs(v) <= 0.5:
        return f"✔  Comfortable (ASHRAE 55 ✓ · ISO 7730:2025 Category {iso})"
    if abs(v) <= 0.7:
        return "△  Acceptable only under ISO 7730 Category C (ASHRAE 55 ✗)"
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
    f"&nbsp;·&nbsp; ISO 7730:2025 / ASHRAE 55-2023</span>"
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

    #  “Batch Solve/批量优化”改称 “Batch Screening（批量参数筛选）”：
    #   仅对送风温度×风速做笛卡尔积扫描并按判据筛选，并非优化算法
    mode = st.radio("Mode", ["Single Case", "Batch Screening"],
                    horizontal=True, label_visibility="collapsed")
    single = (mode == "Single Case")
    st.markdown("<hr style='margin:4px 0 12px 0; border:none; "
                "border-top:1px solid #e0e0e0;'>", unsafe_allow_html=True)

    # Room geometry
    st.markdown(f"<b style='color:#003399;'>Room Geometry</b>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    #  各输入增加下限（min_value），避免零/负值进入分母和平方根；
    #    通过 help 提示给出符号含义。
    a  = labeled_input(c1, "Length a (m)",  "in_a",  value=5.00,  step=0.5, format="%.2f",
                       min_value=0.5, help="Room length")
    b  = labeled_input(c2, "Width b (m)",   "in_b",  value=5.00,  step=0.5, format="%.2f",
                       min_value=0.5, help="Room width")
    hr = labeled_input(c3, "Height h<sub>r</sub> (m)", "in_hr", value=3.60,  step=0.1, format="%.2f",
                       min_value=2.0, help="Room height")

    # Occupants
    st.markdown(f"<b style='color:#003399;'>Occupants</b>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    
    N  = labeled_input(c1, "N",             "in_N",  value=32,    step=1, min_value=1,
                       help="Number of occupants")
    Ap = labeled_input(c2, "A<sub>p</sub> (m²)",  "in_Ap", value=2.56,  step=0.1,  format="%.2f",
                       min_value=0.1,
                       help="Occupant surface area per person used in the zonal radiant/convective "
                            "exchange and in the MRT estimate (not necessarily the DuBois body area)")
    Pt = labeled_input(c3, "P<sub>t</sub> (W)",   "in_Pt", value=120.0, step=10.0, format="%.0f",
                       min_value=1.0,
                       help="Heat dissipation per occupant in the zonal heat-balance model. "
                            "Differs from the PMV metabolic rate M (W/m²) used in the comfort calculation")
    hp = labeled_input(c4, "h<sub>p</sub> (m)",   "in_hp", value=1.5,   step=0.05, format="%.2f",
                       min_value=0.2, max_value=2.8,
                       help="Occupant (heat-source) height")

    # Supply air
    st.markdown(f"<b style='color:#003399;'>Supply Air</b>",
                unsafe_allow_html=True)
    S = st.number_input("Nozzle area S (m²)", value=0.175, format="%.3f",
                        min_value=0.001, help="Total supply nozzle area")  

    if single:
        c1, c2 = st.columns(2)
        #  在标签中提示已验证范围；超出范围时右侧会给出警告/阻止计算
        ts = labeled_input(c1, "Supply temp t<sub>s</sub> (°C)", "in_ts",
                           value=18.00, step=0.5, format="%.2f", min_value=-10.0, max_value=40.0,
                           help=f"Supply air temperature. Validated range: "
                                f"{CALIB_RANGE['ts'][0]:g}–{CALIB_RANGE['ts'][1]:g} °C")
        vs = labeled_input(c2, "Supply vel v<sub>s</sub> (m/s)", "in_vs",
                           value=2.00, step=0.1, format="%.2f", min_value=0.1,
                           help=f"Supply air velocity. Validated range: "
                                f"{CALIB_RANGE['vs'][0]:g}–{CALIB_RANGE['vs'][1]:g} m/s")
    else:
        st.markdown(f"<b style='color:#003399;'>Supply temperature t<sub>s</sub> (°C)</b>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        #  批量默认扫描范围改为标定范围内
        ts_min  = c1.number_input("Min",  value=15.0, step=0.5, format="%.2f")
        ts_max  = c2.number_input("Max",  value=19.0, step=0.5, format="%.2f")
        ts_step = c3.number_input("Step", value=1.0,  step=0.5, format="%.2f",
                                  min_value=0.01)

        st.markdown(f"<b style='color:#003399;'>Supply velocity v<sub>s</sub> (m/s)</b>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        vs_min  = c1.number_input("Min",  value=1.6, step=0.1, format="%.2f", min_value=0.1)
        vs_max  = c2.number_input("Max",  value=2.8, step=0.1, format="%.2f", min_value=0.1)
        vs_step = c3.number_input("Step", value=0.3, step=0.1, format="%.2f",
                                  min_value=0.01)

    st.divider()
    #  原 “Fixed PMV Parameters” 面板改为可编辑的 “PMV Parameters”，
    #    办公静坐常取 1.2 met = 69.78 W/m²；1 met = 58.15 W/m²。
    with st.expander("PMV Parameters (editable)"):
        M    = st.number_input("Metabolic rate M (W/m²)", value=DEFAULT_M, step=1.0,
                               format="%.2f", min_value=40.0,
                               help="1.0 met = 58.15 W/m²; sedentary office work is typically ~1.2 met = 69.78 W/m²")
        W    = st.number_input("Mechanical work W (W/m²)", value=DEFAULT_W, step=1.0,
                               format="%.2f", min_value=0.0)
        I_cl = st.number_input("Clothing insulation I_cl (m²K/W)", value=DEFAULT_I_cl, step=0.01,
                               format="%.3f", min_value=0.0,
                               help="0.155 m²K/W = 1 clo; default 0.124 m²K/W = 0.8 clo")
        RH   = st.number_input("Relative humidity RH (%)", value=DEFAULT_RH, step=5.0,
                               format="%.1f", min_value=0.0, max_value=100.0)

        U    = st.number_input("Occupied zone air speed U (m/s)", value=DEFAULT_U, step=0.05,
                               format="%.2f", min_value=0.05,
                               help="Used for both PMV and draft PD. Not coupled to the supply velocity in this version")
        st.caption("MRT: area-weighted mean of floor, ceiling and occupant-surface temperatures "
                   "(walls / view factors are not included).")

    # 合格判据改为用户可配置；默认值不变。
    with st.expander("Acceptance criteria (editable)"):
        c1, c2 = st.columns(2)
        TOZ_LO  = c1.number_input("t_oz min (°C)", value=24.0, step=0.5, format="%.1f")
        TOZ_HI  = c2.number_input("t_oz max (°C)", value=28.0, step=0.5, format="%.1f")
        c1, c2 = st.columns(2)
        PD_LIM  = c1.number_input("Draft PD max (%)", value=20.0, step=1.0, format="%.1f", min_value=0.0)
        PMV_LIM = c2.number_input("|PMV| max", value=0.5, step=0.1, format="%.2f", min_value=0.0,
                                  help="0.5: ASHRAE 55 / ISO 7730 Category B; 0.7: ISO 7730 Category C")

    #  是否允许在标定范围之外计算（默认不允许；允许时结果带“外推”标记）
    allow_extrap = st.checkbox("Allow extrapolation outside the validated range", value=False,
                               help="Validated: t_s 15–19 °C, v_s 1.6–2.8 m/s, "
                                    "occupant density 1.28–2.4 persons/m². Results outside this range are NOT validated.")

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
        #   范围检查：超出标定范围时给出明确警告；
        #   未勾选 “Allow extrapolation” 时直接阻止计算
        dom_msgs = check_validity_domain(a, b, N, ts, vs)
        if dom_msgs and not allow_extrap:
            st.error("⛔ Input outside the validated range — calculation blocked. "
                     "Tick **Allow extrapolation** in the left panel to run it anyway.")
            for m_ in dom_msgs:
                st.markdown(f"- {m_}")
            st.stop()
        if dom_msgs:
            st.warning("⚠️ Extrapolation — results below are NOT validated:\n\n"
                       + "\n".join(f"- {m_}" for m_ in dom_msgs))

        model = ThermalModel()
        #  set_case 内含输入校验，非法输入给出明确报错而不是静默计算
        try:
            t0_ = time.perf_counter()               #  计时
            model.set_case(a, b, hr, Ap, Pt, N, ts, vs, S, hp,
                           M=M, W=W, I_cl=I_cl, U=U, RH=RH)
            res = model.solve()
            runtime_s = time.perf_counter() - t0_
        except ValueError as err:
            st.error(f"⛔ Invalid input: {err}")
            st.stop()

        #  收敛判据包含残差范数与物理合理性（见 thermal_model.solve）
        if not res['converged']:
            st.error("⚠️ Solver did not converge (or the solution is physically implausible). "
                     "Please adjust the input parameters.")
            st.stop()

        #  PD 显式传入风速 U（与 PMV 一致，与 run_batch 保持一致）
        toz_v = res['toz'];  pd_v  = calc_PD(toz_v, U)
        pmv_v = res['PMV'];  ppd_v = res['PPD']
        e_v   = res['E'];    ach_v = model.ACH
        #  判据取自左侧 “Acceptance criteria” 面板（默认 24–28 °C / 20 % / 0.5）
        ok = (TOZ_LO <= toz_v <= TOZ_HI and pd_v <= PD_LIM and abs(pmv_v) <= PMV_LIM)
        # Pass / fail banner
        if ok:
            st.success("✔  All comfort criteria satisfied: "
                       f"t_oz {TOZ_LO:g}–{TOZ_HI:g} °C; PD ≤ {PD_LIM:g} %; |PMV| ≤ {PMV_LIM:g}.")
        else:
            fails = []
            if not (TOZ_LO <= toz_v <= TOZ_HI):
                fails.append(f"t_oz = {toz_v:.1f} °C  ∉  [{TOZ_LO:g}, {TOZ_HI:g}]")   # 温度 1 位小数
            if not (pd_v <= PD_LIM):
                fails.append(f"PD = {pd_v:.1f} %  > {PD_LIM:g} %")
            if not (abs(pmv_v) <= PMV_LIM):
                fails.append(f"PMV = {pmv_v:.2f}  ∉  [−{PMV_LIM:g}, {PMV_LIM:g}]")
            st.warning("⚠️  Criteria not met:  " + "   ·   ".join(fails))

        st.markdown(f"**{pmv_label(pmv_v)}**"
                    f"&nbsp;&nbsp;&nbsp;ACH = **{ach_v:.1f} h⁻¹**")
        st.markdown("<hr style='margin:4px 0 12px 0; border:none; "
                    "border-top:1px solid #e0e0e0;'>", unsafe_allow_html=True)

        # KPI cards
        k1, k2, k3, k4, k5 = st.columns(5)
        #  温度统一 1 位小数；E 统一 3 位小数；其余 2 位（由 dec 控制）
        def kpi(col, label, val, unit="", ok_range=None, dec=2):
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
                f"{val:.{dec}f}<span style='font-size:26px;'>{unit}</span></div>"
                f"</div>", unsafe_allow_html=True)

        kpi(k1, "Occ. Zone Temp", toz_v, " °C", (TOZ_LO, TOZ_HI), dec=1)
        kpi(k2, "PMV",            pmv_v, "",     (-PMV_LIM, PMV_LIM))
        kpi(k3, "PPD",            ppd_v, " %",   dec=1)
        kpi(k4, "Draft PD",       pd_v,  " %",   (0, PD_LIM), dec=1)
        kpi(k5, "Energy Utilisation Index E", e_v, "", dec=3)
        st.caption(f"Computed in {runtime_s * 1000:.0f} ms.")
        st.markdown("")

        # Charts
        g1, g2 = st.columns([3, 2])

        with g1:
            # 改称 “Zonal Temperature Profile”，并说明折线仅为节点间的视觉引导
            st.markdown(f"<b style='color:#003399;'>Zonal Temperature Profile</b>",
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

            # ── 节点圆点 + 标注 ──────────────────────────────────────────
            # 标签统一紧贴在各自节点右侧 4px 处，随点的位置（温度值）
            # 自动跟随移动，不再使用每个点各不相同的偏移方向/偏移量。
            NODE_LABELS = {
                'tf':  ("t<sub>f</sub>",  dict(xanchor='left', yanchor='middle', xshift=4, yshift=0)),
                'tnf': ("t<sub>nf</sub>", dict(xanchor='left', yanchor='top',    xshift=4, yshift=-4)),
                'toz': ("t<sub>oz</sub>", dict(xanchor='left', yanchor='middle', xshift=4, yshift=0)),
                'tmz': ("t<sub>mz</sub>", dict(xanchor='left', yanchor='middle', xshift=4, yshift=0)),
                'te':  ("t<sub>e</sub>",  dict(xanchor='left', yanchor='middle', xshift=4, yshift=0)),
                'tc':  ("t<sub>c</sub>",  dict(xanchor='left', yanchor='bottom', xshift=4, yshift=4)),
            }
            for z, y, t, _ in nodes:
                fig.add_trace(go.Scatter(
                    x=[t], y=[y], mode='markers',
                    marker=dict(size=6, color=_t2color(t),
                               line=dict(color='white', width=1.5)),
                    showlegend=False,
                    hovertext=f"{t:.1f} °C @ {y:.2f} m", hoverinfo='text',   #  温度 1 位小数
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
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)   #  高分辨率导出
            st.caption("Markers are discrete zonal node temperatures; the connecting lines are "
                       "visual guides only, not a continuously resolved vertical temperature field.")   

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
                    #  同时标出 ASHRAE 55（±0.5）与 ISO 7730 Category C（±0.7）范围
                    'steps': [
                        {'range': [-3,   -0.7], 'color': '#d6eaf8'},
                        {'range': [-0.7, -0.5], 'color': '#eef5d0'},
                        {'range': [-0.5,  0.5], 'color': '#d5f5e3'},
                        {'range': [0.5,   0.7], 'color': '#eef5d0'},
                        {'range': [0.7,   3  ], 'color': '#fde8e8'},
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
            st.plotly_chart(fig2, use_container_width=True, config=PLOT_CONFIG)
            st.caption("Green: ASHRAE 55 (|PMV| ≤ 0.5, = ISO 7730 Cat. B); "
                       "yellow-green: ISO 7730 Cat. C only (|PMV| ≤ 0.7).")   

        with st.expander("Detailed Results"):
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Supply temperature t<sub>s</sub> | {ts:.1f} °C |
| Supply velocity v<sub>s</sub> | {vs:.2f} m/s |
| Air change rate ACH | {ach_v:.2f} h⁻¹ |
| Exhaust zone temperature t<sub>e</sub> | {res['te']:.1f} °C |
| Floor temperature t<sub>f</sub> | {res['tf']:.1f} °C |
| Ceiling temperature t<sub>c</sub> | {res['tc']:.1f} °C |
| Floor zone temperature t<sub>nf</sub> | {res['tnf']:.1f} °C |
| Mixed zone temperature t<sub>mz</sub> | {res['tmz']:.1f} °C |
| Occupied zone temperature t<sub>oz</sub> | {toz_v:.1f} °C |
| Energy utilisation index E | {e_v:.3f} |
| Solver residual (max norm) | {res['res_norm']:.1e} W |
""", unsafe_allow_html=True)   

    # ── Batch mode ────────────────────────────────────────────────────────────
    else:
        ts_list = [round(ts_min + i*ts_step, 4)
                   for i in range(int(round((ts_max-ts_min)/ts_step))+1)]
        vs_list = [round(vs_min + i*vs_step, 4)
                   for i in range(int(round((vs_max-vs_min)/vs_step))+1)]
        total = len(ts_list) * len(vs_list)
        combos = [(t, v) for t in ts_list for v in vs_list]

        #  标定范围检查：默认跳过外推工况；
        #   勾选 “Allow extrapolation” 后计算，但在热力图/表格中明确标记为外推（未验证）
        ext_flags = {c: len(check_validity_domain(a, b, N, c[0], c[1])) > 0 for c in combos}
        n_ext = sum(ext_flags.values())
        if n_ext and not allow_extrap:
            combos = [c for c in combos if not ext_flags[c]]
            st.warning(f"⚠️ {n_ext} of {total} combinations lie outside the validated range "
                       f"(t_s {CALIB_RANGE['ts'][0]:g}–{CALIB_RANGE['ts'][1]:g} °C, "
                       f"v_s {CALIB_RANGE['vs'][0]:g}–{CALIB_RANGE['vs'][1]:g} m/s, "
                       f"density {CALIB_RANGE['density'][0]:g}–{CALIB_RANGE['density'][1]:g} persons/m²) "
                       f"and were skipped. Tick **Allow extrapolation** to include them (flagged as extrapolated).")
            if not combos:
                st.error("No combination lies inside the validated range.")
                st.stop()
        elif n_ext:
            st.warning(f"⚠️ {n_ext} of {total} cases are extrapolations (grey-shaded cells) — NOT validated.")
        n_run = len(combos)

        #  “批量优化”改称 “批量参数筛选”
        st.markdown(f"<b style='color:#003399;'>Batch Screening Results — "
                    f"{n_run} cases</b>", unsafe_allow_html=True)
        bar = st.progress(0)
        rows = []
        t_batch0 = time.perf_counter()          #  批量总计时
        for idx, (ts_i, vs_i) in enumerate(combos, 1):
            t_case0 = time.perf_counter()
            model = ThermalModel()
            try:   #  非法输入给出明确报错
                model.set_case(a, b, hr, Ap, Pt, N, ts_i, vs_i, S, hp,
                               M=M, W=W, I_cl=I_cl, U=U, RH=RH)
            except ValueError as err:
                st.error(f"⛔ Invalid input: {err}")
                st.stop()
            res   = model.solve()
            toz_v = res['toz']
            pd_v  = calc_PD(toz_v, U) if res['converged'] else float('nan')   
            pmv_v = res['PMV']
            # 判据取自 “Acceptance criteria” 面板
            ok    = (res['converged'] and TOZ_LO <= toz_v <= TOZ_HI
                     and pd_v <= PD_LIM and abs(pmv_v) <= PMV_LIM)
            #  表中保留全精度数值，仅在显示时按位数舍入，
            #   避免“先舍入再舍入”造成单工况与热力图相差 0.01 °C
            rows.append({'ts': ts_i, 'vs': vs_i, 'ACH': model.ACH,
                         'tf': res['tf'], 'tnf': res['tnf'],
                         'toz': toz_v, 'tmz': res['tmz'],
                         'te': res['te'], 'tc': res['tc'],
                         'PMV': pmv_v,
                         'PPD': res['PPD'], 'PD': pd_v,
                         'E': res['E'],
                         'converged': res['converged'], 'qualified': ok,
                         'extrapolated': ext_flags[(ts_i, vs_i)],           
                         'res_norm': res['res_norm'],                        
                         'runtime_s': time.perf_counter() - t_case0})        
            bar.progress(idx / n_run)
        t_batch = time.perf_counter() - t_batch0
        bar.empty()

        df   = pd.DataFrame(rows)
        df_q = df[df['qualified']]
        if len(df_q):
            st.success(f"✔  {len(df_q)} / {n_run} cases satisfy all criteria")
        else:
            st.warning(f"⚠️  No cases satisfy all criteria ({n_run} computed)")
        n_fail = int((~df['converged']).sum())
        st.caption(f"Computed {n_run} cases in {t_batch:.2f} s "
                   f"(mean {1000 * t_batch / n_run:.0f} ms/case); "
                   f"non-converged: {n_fail}; max residual {df['res_norm'].max():.1e} W.")  

        #  合格工况汇总表（含 E 与全部相关指标），并说明 E 最大者是否唯一；
        #   E 是温度利用有效度指标，不是系统总能效，故不据此给出“全局最佳/推荐”工况
        if len(df_q):
            disp = df_q[['ts', 'vs', 'toz', 'PMV', 'PPD', 'PD', 'E', 'extrapolated']]
            st.dataframe(disp.round({'ts': 1, 'vs': 2, 'toz': 1, 'PMV': 2, 'PPD': 1, 'PD': 1,
                                     'E': 3}),
                         use_container_width=True, hide_index=True)
            top = df_q[df_q['E'] >= df_q['E'].max() - 1e-3]
            st.caption(f"Highest E among the acceptable cases: {df_q['E'].max():.3f} "
                       f"({'unique' if len(top) == 1 else f'tied: {len(top)} cases within 0.001'}). ")

        def make_heatmap(col, title, colorscale, zmid=None, dec=2):
            pivot = df.pivot(index='vs', columns='ts', values=col)
            kw = dict(zmid=zmid) if zmid is not None else {}
            #  温度 1 位小数、E 3 位小数，其余 2 位；缺失格（被跳过的外推工况）不显示文字
            txt = [[f"{v:.{dec}f}" if np.isfinite(v) else "" for v in row] for row in pivot.values]
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(),
                y=pivot.index.tolist(), colorscale=colorscale,
                text=txt, texttemplate="%{text}",
                colorbar=dict(title=col), **kw))
            #  外推单元格灰色遮罩标记（未验证）
            for _, row in df[df['extrapolated']].iterrows():
                fig.add_shape(type='rect',
                    x0=row['ts'] - 0.5*ts_step, x1=row['ts'] + 0.5*ts_step,
                    y0=row['vs'] - 0.5*vs_step, y1=row['vs'] + 0.5*vs_step,
                    fillcolor='rgba(110,110,110,0.45)', line=dict(width=0))
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

        tab1, tab2, tab3, tab4, tab6, tab7 = st.tabs([
            "Occupied Zone Temp", "PMV", "PPD", "Draft PD",
            "Energy Utilisation E", "Data Table"])
        with tab1:
            st.plotly_chart(make_heatmap('toz',
                'Occupied Zone Temperature t<sub>oz</sub> (°C)', 'RdYlGn_r', dec=1),
                use_container_width=True, config=PLOT_CONFIG)
            st.caption("Dark blue border = all criteria satisfied; "
                       "grey shading = extrapolated (outside the validated range, not validated).")
        with tab2:
            st.plotly_chart(make_heatmap('PMV', 'PMV Index',
                'RdBu_r', zmid=0), use_container_width=True, config=PLOT_CONFIG)
        with tab3:
            st.plotly_chart(make_heatmap('PPD', 'PPD (%)',
                'YlOrRd', dec=1), use_container_width=True, config=PLOT_CONFIG)
        with tab4:
            st.plotly_chart(make_heatmap('PD', 'Draft PD (%)',
                'YlOrRd', dec=1), use_container_width=True, config=PLOT_CONFIG)
        with tab6:
            #  E 热力图（文中依据 E 比较工况，需给出 E 的结果）
            st.plotly_chart(make_heatmap('E',
                'Energy Utilisation Index E', 'Blues', dec=3),
                use_container_width=True, config=PLOT_CONFIG)
        with tab7:
            st.dataframe(df.round({'ts': 1, 'vs': 2, 'ACH': 2, 'tf': 1, 'tnf': 1, 'toz': 1,
                                   'tmz': 1, 'te': 1, 'tc': 1, 'PMV': 2, 'PPD': 1, 'PD': 1,
                                   'E': 3, 'runtime_s': 4}),
                         use_container_width=True, height=380)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️  Download CSV (full precision)", csv,
                               file_name="ijv_results.csv", mime="text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "IJV Thermal Comfort Tool  ·  four-zonal model  ·  "
    "PMV/PPD: ISO 7730:2025 / ASHRAE 55-2023  ·  "
    "Default comfort criteria (editable): t_oz 24–28 °C  |  Draft PD ≤ 20 %  |  |PMV| ≤ 0.5  ·  "
    "Validated range: t_s 15–19 °C, v_s 1.6–2.8 m/s, 1.28–2.4 persons/m²"   
)
