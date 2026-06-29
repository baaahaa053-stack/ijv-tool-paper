"""
IJV Thermal Comfort Tool
基于撞击射流通风（IJV）的热舒适评价工具
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

# ── PMV 固定参数 ──────────────────────────────────────────────────────────────
M    = 58.15   # 新陈代谢量 W/m²
W    = 0.0     # 机械做功 W/m²
I_cl = 0.124   # 服装热阻 m²K/W
RH   = 50.0    # 相对湿度 %
U    = 0.3     # 工作区风速 m/s

# ── 辅助函数 ──────────────────────────────────────────────────────────────────
def pmv_color(v):
    if abs(v) <= 0.5: return "#27ae60"
    if abs(v) <= 1.0: return "#e67e22"
    return "#e74c3c"

def pmv_label(v):
    if abs(v) <= 0.5: return "✔ 舒适  Comfortable"
    if abs(v) <= 1.0: return "△ 略不舒适  Slightly uncomfortable"
    return "✘ 不舒适  Uncomfortable"

PROFILE_ZONES = [
    ('tc',  3.60, 'Ceiling',        '#e67e22'),
    ('te',  3.50, 'Exhaust',        '#e67e22'),
    ('tmz', 1.80, 'Mixed zone',     '#e67e22'),
    ('toz', 0.60, 'Occupied zone',  '#27ae60'),
    ('tnf', 0.10, 'Near-floor air', '#2980b9'),
    ('tf',  0.00, 'Floor',          '#2980b9'),
]

# ── 标题 ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='text-align:center; margin-bottom:4px;'>"
    "IJV Thermal Comfort Tool</h2>"
    "<p style='text-align:center; color:gray; margin-top:0;'>"
    "Impinging Jet Ventilation · 10-Zone Nonlinear Thermal Model</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── 双栏布局 ──────────────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 2], gap="large")

# ════════════════════════════════════════
# 左栏：参数输入
# ════════════════════════════════════════
with col_in:
    st.markdown("#### 参数输入 Parameters")

    mode = st.radio("计算模式", ["单工况  Single Case", "批量  Batch Solve"],
                    horizontal=True, label_visibility="collapsed")
    single = mode.startswith("单")
    st.divider()

    # 房间
    st.markdown("**房间 Room**")
    c1, c2, c3 = st.columns(3)
    a  = c1.number_input("a (m)",  value=5.0, step=0.1, format="%.1f")
    b  = c2.number_input("b (m)",  value=5.0, step=0.1, format="%.1f")
    hr = c3.number_input("hr (m)", value=3.6, step=0.1, format="%.1f")

    # 人员
    st.markdown("**人员 Occupants**")
    c1, c2, c3, c4 = st.columns(4)
    N  = c1.number_input("N",       value=32,    step=1)
    Ap = c2.number_input("Ap (m²)", value=2.56,  step=0.1, format="%.2f")
    Pt = c3.number_input("Pt (W)",  value=120.0, step=10.0, format="%.0f")
    hp = c4.number_input("hp (m)",  value=1.5,   step=0.05, format="%.2f")

    # 送风
    st.markdown("**送风 Supply Air**")
    S = st.number_input("送风口面积 S (m²)", value=0.170625, format="%.6f")

    if single:
        c1, c2 = st.columns(2)
        ts = c1.number_input("ts (°C)", value=18, min_value=10, max_value=25, step=1)
        vs = c2.number_input("vs (m/s)", value=1.5, min_value=0.5, max_value=4.0,
                             step=0.1, format="%.1f")
    else:
        st.markdown("**ts 范围 (°C)**")
        c1, c2, c3 = st.columns(3)
        ts_min  = c1.number_input("最小 min", value=15, step=0.1, format="%.1f")
        ts_max  = c2.number_input("最大 max", value=22, step=0.1, format="%.1f")
        ts_step = c3.number_input("步长 step", value=1,  step=0.1, format="%.1f", min_value=1)

        st.markdown("**vs 范围 (m/s)**")
        c1, c2, c3 = st.columns(3)
        vs_min  = c1.number_input("最小 min", value=1.0, step=0.01, format="%.2f")
        vs_max  = c2.number_input("最大 max", value=2.0, step=0.01, format="%.2f")
        vs_step = c3.number_input("步长 step", value=0.1, step=0.01, format="%.2f",
                                  min_value=0.05)

    st.divider()
    # 固定参数展示
    with st.expander("固定参数 Fixed Parameters"):
        st.markdown(f"""
| 参数 | 值 |
|------|----|
| 新陈代谢率 M | {M} W/m² |
| 机械功 W | {W} W/m² |
| 服装热阻 I_cl | {I_cl} m²K/W |
| 相对湿度 RH | {RH} % |
| 工作区风速 U | {U} m/s |
""")

    run = st.button("▶  计算 Run", type="primary", use_container_width=True)

# ════════════════════════════════════════
# 右栏：结果输出
# ════════════════════════════════════════
with col_out:
    if not run:
        st.markdown("#### 计算结果 Results")
        st.info("请在左侧设置参数后点击 **▶ 计算 Run**")
        st.stop()

    # ── 单工况 ────────────────────────────────────────────────────────────────
    if single:
        model = ThermalModel()
        model.set_case(a, b, hr, Ap, Pt, N, ts, vs, S, hp,
                       M=M, W=W, I_cl=I_cl, U=U, RH=RH)
        res = model.solve()

        if not res['converged']:
            st.error("⚠️ 求解未收敛，请调整参数。Solver did not converge.")
            st.stop()

        toz_v = res['toz'];  pd_v = calc_PD(toz_v)
        pmv_v = res['PMV'];  ppd_v = res['PPD']
        e_v   = res['E'];    ach_v = model.ACH
        ok = (24 <= toz_v <= 28 and pd_v <= 20 and abs(pmv_v) <= 0.5)

        # 合格判定横幅
        if ok:
            st.success("✔ 符合全部舒适标准  All criteria met"
                       "（toz 24–28°C · PD ≤20% · PMV ±0.5）")
        else:
            fails = []
            if not (24 <= toz_v <= 28): fails.append(f"toz={toz_v:.2f}°C ∉ [24,28]")
            if not (pd_v <= 20):        fails.append(f"PD={pd_v:.1f}% > 20%")
            if not (abs(pmv_v) <= 0.5): fails.append(f"PMV={pmv_v:.2f} ∉ [−0.5,0.5]")
            st.warning("⚠️ 不合格：" + "  ·  ".join(fails))

        st.markdown(f"**{pmv_label(pmv_v)}**　　ACH = **{ach_v:.1f} h⁻¹**")
        st.divider()

        # KPI 行
        k1, k2, k3, k4, k5 = st.columns(5)
        def kpi(col, label, val, unit="", ok_range=None):
            color = ""
            if ok_range:
                color = "color:#27ae60;" if ok_range[0]<=val<=ok_range[1] else "color:#e74c3c;"
            col.markdown(
                f"<div style='text-align:center'>"
                f"<div style='font-size:12px;color:gray'>{label}</div>"
                f"<div style='font-size:24px;font-weight:bold;{color}'>{val:.2f}{unit}</div>"
                f"</div>", unsafe_allow_html=True)

        kpi(k1, "toz (°C)",    toz_v, "°C", (24, 28))
        kpi(k2, "PMV",         pmv_v, "",   (-0.5, 0.5))
        kpi(k3, "PPD (%)",     ppd_v, "%")
        kpi(k4, "Draft PD (%)",pd_v,  "%",  (0, 20))
        kpi(k5, "Energy E",    e_v,   "")
        st.markdown("")

        # 图表双栏
        g1, g2 = st.columns([3, 2])

        with g1:
            st.markdown("**温度分布 Temperature Profile**")
            fig = go.Figure(go.Bar(
                x=[res[z] for z, *_ in PROFILE_ZONES],
                y=[f"{h:.2f}m  {l}" for _, h, l, _ in PROFILE_ZONES],
                orientation='h',
                marker_color=[c for _, _, _, c in PROFILE_ZONES],
                text=[f"{res[z]:.1f}°C" for z, *_ in PROFILE_ZONES],
                textposition='outside',
                width=0.5,
            ))
            xmin = min(res[z] for z, *_ in PROFILE_ZONES) - 1
            xmax = max(res[z] for z, *_ in PROFILE_ZONES) + 2
            fig.update_layout(
                xaxis=dict(title="Temperature (°C)", range=[xmin, xmax]),
                yaxis_title="Height",
                height=300, margin=dict(l=0, r=60, t=10, b=30),
                showlegend=False, font=dict(size=12),
                plot_bgcolor='white',
            )
            fig.update_xaxes(showgrid=True, gridcolor='#eee')
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            st.markdown("**PMV**")
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pmv_v,
                number={'font': {'size': 32}},
                gauge={
                    'axis': {'range': [-3, 3], 'tickwidth': 1,
                             'tickvals': [-3,-2,-1,0,1,2,3]},
                    'bar': {'color': pmv_color(pmv_v), 'thickness': 0.25},
                    'steps': [
                        {'range': [-3,   -0.5], 'color': '#fde8e8'},
                        {'range': [-0.5,  0.5], 'color': '#e8f8ee'},
                        {'range': [0.5,   3  ], 'color': '#fef3e2'},
                    ],
                }
            ))
            fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        # 详细数据表
        with st.expander("详细数据 Detail"):
            st.markdown(f"""
| 参数 | 数值 |
|------|------|
| 送风温度 ts | {ts} °C |
| 送风速度 vs | {vs} m/s |
| 换气次数 ACH | {ach_v:.2f} h⁻¹ |
| 排风温度 te | {res['te']:.2f} °C |
| 地板温度 tf | {res['tf']:.2f} °C |
| 天花板温度 tc | {res['tc']:.2f} °C |
| 近地面气温 tnf | {res['tnf']:.2f} °C |
| 混合区温度 tmz | {res['tmz']:.2f} °C |
| 能量利用系数 E | {e_v:.3f} |
""")

    # ── 批量模式 ──────────────────────────────────────────────────────────────
    else:
        ts_list = list(range(int(ts_min), int(ts_max)+1, int(ts_step)))
        vs_list = [round(vs_min + i*vs_step, 3)
                   for i in range(int(round((vs_max-vs_min)/vs_step))+1)]
        total = len(ts_list) * len(vs_list)

        st.markdown(f"#### 批量计算  共 {total} 个工况")
        bar = st.progress(0)
        rows = []
        for idx, (ts_i, vs_i) in enumerate(
                [(t,v) for t in ts_list for v in vs_list], 1):
            model = ThermalModel()
            model.set_case(a,b,hr,Ap,Pt,N,ts_i,vs_i,S,hp,
                           M=M,W=W,I_cl=I_cl,U=U,RH=RH)
            res   = model.solve()
            toz_v = res['toz']
            pd_v  = calc_PD(toz_v) if res['converged'] else float('nan')
            pmv_v = res['PMV']
            ok    = (res['converged'] and 24<=toz_v<=28
                     and pd_v<=20 and abs(pmv_v)<=0.5)
            rows.append({'ts':ts_i,'vs':vs_i,'ACH':round(model.ACH,2),
                         'toz':round(toz_v,3),'PMV':round(pmv_v,3),
                         'PPD':round(res['PPD'],2),'PD':round(pd_v,2),
                         'E':round(res['E'],3),'te':round(res['te'],3),
                         'converged':res['converged'],'qualified':ok})
            bar.progress(idx/total)
        bar.empty()

        df   = pd.DataFrame(rows)
        df_q = df[df['qualified']]
        st.success(f"✔ 合格工况 {len(df_q)} / {total}") if len(df_q) \
            else st.warning(f"⚠️ 无合格工况（共 {total} 个）")

        def make_heatmap(col, title, colorscale, zmid=None):
            pivot = df.pivot(index='vs', columns='ts', values=col)
            kw = dict(zmid=zmid) if zmid is not None else {}
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(),
                y=pivot.index.tolist(), colorscale=colorscale,
                text=np.round(pivot.values,2), texttemplate="%{text}",
                colorbar=dict(title=col), **kw))
            for _, row in df_q.iterrows():
                fig.add_shape(type='rect',
                    x0=row['ts']-.45, x1=row['ts']+.45,
                    y0=row['vs']-.45*vs_step, y1=row['vs']+.45*vs_step,
                    line=dict(color='black', width=2))
            fig.update_layout(
                title=title,
                xaxis_title="送风温度 ts (°C)",
                yaxis_title="送风速度 vs (m/s)",
                height=400)
            return fig

        tab1,tab2,tab3,tab4 = st.tabs(
            ["toz 热力图","PMV 热力图","E 热力图","数据表"])
        with tab1:
            st.plotly_chart(make_heatmap('toz','工作区温度 toz (°C)',
                'RdYlGn_r'), use_container_width=True)
            st.caption("黑色边框 = 全部合格")
        with tab2:
            st.plotly_chart(make_heatmap('PMV','PMV',
                'RdBu_r', zmid=0), use_container_width=True)
        with tab3:
            st.plotly_chart(make_heatmap('E','能量利用系数 E',
                'Blues'), use_container_width=True)
        with tab4:
            st.dataframe(df, use_container_width=True, height=380)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ 下载 CSV", csv,
                               file_name="ijv_results.csv", mime="text/csv")

# ── 页脚 ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption("IJV Thermal Comfort Tool · Streamlit · "
           "10-zone nonlinear model · ISO 7730 PMV/PPD · "
           "Criteria: toz 24–28°C | PD ≤20% | PMV ±0.5")
