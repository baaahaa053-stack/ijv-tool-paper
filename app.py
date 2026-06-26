"""
IJV Thermal Comfort Tool
基于撞击射流通风（IJV）的热舒适评价工具
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from thermal_model import ThermalModel, BatchSolver, calc_PD

# ── 页面基础配置 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IJV Thermal Comfort Tool",
    page_icon="🌡️",
    layout="wide",
)

st.markdown("### 🌡️ IJV Thermal Comfort Tool")
st.caption("Impinging Jet Ventilation · Thermal Comfort Evaluation")

# ── 侧边栏：参数输入 ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    mode = st.radio("Mode", ["Single Case", "Batch Solve"], horizontal=True)
    st.divider()

    # ── 房间参数 ──
    st.subheader("Room")
    a  = st.number_input("Length a (m)",  value=5.0, step=0.5)
    b  = st.number_input("Width b (m)",   value=5.0, step=0.5)
    hr = st.number_input("Height hr (m)", value=3.6, step=0.1)

    st.subheader("Occupants")
    N  = st.number_input("Number N",             value=32,   step=1)
    Ap = st.number_input("Projected area Ap (m²)", value=2.56, step=0.1)
    Pt = st.number_input("Heat gain Pt (W)",     value=120.0, step=10.0)
    hp = st.number_input("Body height hp (m)",   value=1.5,  step=0.05)

    st.subheader("Supply Air")
    S  = st.number_input("Nozzle area S (m²)", value=0.170625, format="%.6f")

    if mode == "Single Case":
        ts = st.slider("Supply temp ts (°C)", 14, 24, 18)
        vs = st.slider("Supply vel vs (m/s)", 0.5, 3.0, 1.5, step=0.1)
    else:
        st.markdown("**ts range (°C)**")
        ts_min, ts_max = st.select_slider(
            "ts", options=list(range(14, 25)), value=(15, 22), label_visibility="collapsed")
        ts_step = st.number_input("ts step", value=1, min_value=1)

        st.markdown("**vs range (m/s)**")
        vs_min = st.number_input("vs min", value=1.0, step=0.1)
        vs_max = st.number_input("vs max", value=2.0, step=0.1)
        vs_step = st.number_input("vs step", value=0.1, step=0.1, format="%.1f")

    # ── PMV 固定参数（不在界面显示）──
    M    = 58.15   # 新陈代谢量 W/m²
    W    = 0.0     # 机械做功 W/m²
    I_cl = 0.124   # 服装热阻 m²K/W
    RH   = 50.0    # 相对湿度 %
    U    = 0.3     # 工作区风速 m/s

    st.divider()
    run = st.button("▶ Run", type="primary", use_container_width=True)

# ── 主区域 ────────────────────────────────────────────────────────────────────
ZONE_NAMES = ['tf','tnf','toz','tpo','tp','tmz','tpm','te','tpe','tc']
ZONE_LABELS = {
    'tf':'Floor','tnf':'Near-floor air','toz':'Occupied zone',
    'tpo':'Plume (lower)','tp':'Body surface','tmz':'Mixed zone',
    'tpm':'Plume (upper)','te':'Exhaust','tpe':'Plume (exhaust)','tc':'Ceiling'
}
ZONE_HEIGHT = {
    'tf':0.0,'tnf':0.1,'toz':0.6,'tpo':0.8,'tp':0.9,
    'tmz':1.8,'tpm':2.5,'te':3.5,'tpe':3.55,'tc':3.6
}

def pmv_color(pmv):
    if abs(pmv) <= 0.5:  return "#2ecc71"
    if abs(pmv) <= 1.0:  return "#f39c12"
    return "#e74c3c"

def pmv_label(pmv):
    if abs(pmv) <= 0.5:  return "😊 Comfortable"
    if abs(pmv) <= 1.0:  return "😐 Slightly uncomfortable"
    return "😣 Uncomfortable"

# ── 单工况模式 ────────────────────────────────────────────────────────────────
if mode == "Single Case":
    if not run:
        st.info("Set parameters in the sidebar and click **▶ Run**.")
        st.stop()

    model = ThermalModel()
    model.set_case(a, b, hr, Ap, Pt, N, ts, vs, S, hp,
                   M=M, W=0.0, I_cl=I_cl, U=U, RH=RH)
    res = model.solve()

    if not res['converged']:
        st.error("⚠️ Solver did not converge. Please adjust parameters.")
        st.stop()

    toz_v = res['toz']
    pd_v  = calc_PD(toz_v)
    pmv_v = res['PMV']
    ppd_v = res['PPD']
    e_v   = res['E']
    ach_v = model.ACH

    # ── KPI 卡片 ──
    st.subheader("📊 Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Occupied Zone Temp", f"{toz_v:.2f} °C",
              delta=f"{toz_v-26:.2f} vs 26°C", delta_color="inverse")
    c2.metric("PMV", f"{pmv_v:.2f}", help="ISO 7730: |PMV|≤0.5 = Category B")
    c3.metric("PPD", f"{ppd_v:.1f} %")
    c4.metric("Draft PD", f"{pd_v:.1f} %")
    c5.metric("Energy Util. E", f"{e_v:.3f}")

    st.markdown(f"**Comfort assessment:** {pmv_label(pmv_v)}&nbsp;&nbsp;"
                f"&nbsp;ACH = **{ach_v:.1f} h⁻¹**")

    ok = (24 <= toz_v <= 28 and 0 <= pd_v <= 20 and -0.5 <= pmv_v <= 0.5)
    if ok:
        st.success("✅ All criteria met (toz 24–28°C | PD ≤20% | PMV ±0.5)")
    else:
        fails = []
        if not (24 <= toz_v <= 28): fails.append(f"toz={toz_v:.2f}°C out of [24,28]")
        if not (pd_v <= 20):        fails.append(f"PD={pd_v:.1f}% > 20%")
        if not (abs(pmv_v) <= 0.5): fails.append(f"PMV={pmv_v:.2f} out of [−0.5, 0.5]")
        st.warning("⚠️ Failed: " + " · ".join(fails))

    st.divider()

    # ── 两栏图表 ──
    col_left, col_right = st.columns(2)

    # 左：温度分布图 — 只显示 6 个主要区域
    with col_left:
        st.subheader("Temperature Profile")
        PROFILE_ZONES = [
            ('tf',  0.00, 'Floor',         '#3498db'),
            ('tnf', 0.10, 'Near-floor air','#3498db'),
            ('toz', 0.60, 'Occupied zone', '#2ecc71'),
            ('tmz', 1.80, 'Mixed zone',    '#f39c12'),
            ('te',  3.50, 'Exhaust',       '#f39c12'),
            ('tc',  3.60, 'Ceiling',       '#f39c12'),
        ]
        fig_profile = go.Figure()
        fig_profile.add_trace(go.Bar(
            x=[res[z] for z, *_ in PROFILE_ZONES],
            y=[f"{h:.2f}m  {l}" for _, h, l, _ in PROFILE_ZONES],
            orientation='h',
            marker_color=[c for _, _, _, c in PROFILE_ZONES],
            text=[f"{res[z]:.2f}°C" for z, *_ in PROFILE_ZONES],
            textposition='outside',
        ))
        fig_profile.update_layout(
            xaxis_title="Temperature (°C)",
            yaxis_title="Height",
            height=320,
            margin=dict(l=10, r=70, t=20, b=40),
            showlegend=False,
            font=dict(size=12),
        )
        st.plotly_chart(fig_profile, use_container_width=True)

    # 右：PMV gauge + 各指标汇总
    with col_right:
        st.subheader("PMV Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pmv_v,
            number={'suffix': '', 'font': {'size': 36}},
            gauge={
                'axis': {'range': [-3, 3], 'tickwidth': 1},
                'bar': {'color': pmv_color(pmv_v), 'thickness': 0.3},
                'steps': [
                    {'range': [-3, -0.5], 'color': '#fadbd8'},
                    {'range': [-0.5, 0.5], 'color': '#d5f5e3'},
                    {'range': [0.5, 3],   'color': '#fdebd0'},
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 3},
                    'thickness': 0.8,
                    'value': pmv_v,
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Supply temp ts | {ts} °C |
| Supply vel vs | {vs} m/s |
| ACH | {ach_v:.2f} h⁻¹ |
| Exhaust temp te | {res['te']:.2f} °C |
| Floor temp tf | {res['tf']:.2f} °C |
| Ceiling temp tc | {res['tc']:.2f} °C |
| Body surface tp | {res['tp']:.2f} °C |
| Energy util. E | {e_v:.3f} |
""")

# ── 批量模式 ──────────────────────────────────────────────────────────────────
else:
    if not run:
        st.info("Set parameter ranges in the sidebar and click **▶ Run**.")
        st.stop()

    ts_list = list(range(ts_min, ts_max + 1, ts_step))
    vs_list = [round(vs_min + i * vs_step, 2)
               for i in range(int(round((vs_max - vs_min) / vs_step)) + 1)]

    n_cases = len(ts_list) * len(vs_list)
    st.info(f"Running **{n_cases}** cases …")
    progress = st.progress(0)

    rows = []
    total = len(ts_list) * len(vs_list)
    idx = 0
    for ts_i in ts_list:
        for vs_i in vs_list:
            model = ThermalModel()
            model.set_case(a, b, hr, Ap, Pt, N, ts_i, vs_i, S, hp,
                           M=M, W=0.0, I_cl=I_cl, U=U, RH=RH)
            res = model.solve()
            toz_v = res['toz']
            pd_v  = calc_PD(toz_v) if res['converged'] else float('nan')
            pmv_v = res['PMV']
            ok = (res['converged']
                  and 24 <= toz_v <= 28
                  and 0  <= pd_v  <= 20
                  and -0.5 <= pmv_v <= 0.5)
            rows.append({
                'ts': ts_i, 'vs': vs_i,
                'ACH': round(model.ACH, 2),
                'toz': round(toz_v, 3),
                'PMV': round(pmv_v, 3),
                'PPD': round(res['PPD'], 2),
                'PD':  round(pd_v,  2),
                'E':   round(res['E'], 3),
                'te':  round(res['te'], 3),
                'converged': res['converged'],
                'qualified': ok,
            })
            idx += 1
            progress.progress(idx / total)

    progress.empty()
    df = pd.DataFrame(rows)
    df_q = df[df['qualified']]

    st.subheader(f"📊 Batch Results — {len(df_q)}/{len(df)} cases qualified")

    # ── 热力图：toz ──
    tab1, tab2, tab3, tab4 = st.tabs(["toz heatmap", "PMV heatmap", "E heatmap", "Data table"])

    def make_heatmap(df, col, title, colorscale, zmid=None):
        pivot = df.pivot(index='vs', columns='ts', values=col)
        kw = dict(zmid=zmid) if zmid is not None else {}
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=colorscale,
            text=np.round(pivot.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title=col),
            **kw,
        ))
        # 标记合格区域
        for _, row in df[df['qualified']].iterrows():
            fig.add_shape(type='rect',
                x0=row['ts']-0.45, x1=row['ts']+0.45,
                y0=row['vs']-0.045, y1=row['vs']+0.045,
                line=dict(color='black', width=2))
        fig.update_layout(
            title=title,
            xaxis_title="Supply temperature ts (°C)",
            yaxis_title="Supply velocity vs (m/s)",
            height=420,
        )
        return fig

    with tab1:
        st.plotly_chart(make_heatmap(df, 'toz', 'Occupied Zone Temperature (°C)',
                                     'RdYlGn_r'), use_container_width=True)
        st.caption("Black border = all criteria met")

    with tab2:
        st.plotly_chart(make_heatmap(df, 'PMV', 'PMV',
                                     'RdBu_r', zmid=0), use_container_width=True)

    with tab3:
        st.plotly_chart(make_heatmap(df, 'E', 'Energy Utilization Coefficient E',
                                     'Blues'), use_container_width=True)

    with tab4:
        st.dataframe(
            df.style.applymap(
                lambda v: 'background-color: #d5f5e3' if v else 'background-color: #fadbd8',
                subset=['qualified']
            ),
            use_container_width=True,
            height=400,
        )
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ Download CSV", csv,
                           file_name="ijv_batch_results.csv", mime="text/csv")

# ── 页脚 ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption("IJV Thermal Comfort Tool · Built with Streamlit · "
           "Model: 10-zone nonlinear thermal model | "
           "Comfort: ISO 7730 Fanger PMV/PPD | "
           "Criteria: toz 24–28°C, PD ≤20%, PMV ±0.5")
