import math
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from itertools import product
import time          # 实测计算时间
import platform      # 记录运行环境
import sys


def tandg(deg):
    return np.tan(np.deg2rad(deg))



CALIB_RANGE = {
    'ts':      (15.0, 19.0),    # 送风温度 (°C)
    'vs':      (1.6, 2.8),      # 送风速度 (m/s)
    'density': (1.28, 2.4),     # 人员密度 N/(a·b) (人/m²)
}
_RANGE_TOL = 1e-9               # 边界浮点容差


def check_validity_domain(a, b, N, ts, vs):
    """
    返回“超出标定范围”的提示列表；空列表表示全部落在已验证范围内。
    """
    msgs = []
    dens = N / (a * b)
    for name, val, unit in [('ts', ts, '°C'), ('vs', vs, 'm/s'), ('density', dens, 'persons/m²')]:
        lo, hi = CALIB_RANGE[name]
        if val < lo - _RANGE_TOL or val > hi + _RANGE_TOL:
            msgs.append(f"{name} = {val:.2f} {unit} is outside the calibrated range "
                        f"[{lo}, {hi}] {unit} (extrapolation, not validated)")
    return msgs


def validate_inputs(a, b, hr, Ap, Pt, N, ts, vs, S, hp, De=0.4, ho=2.0, hnf=0.1):
    """
    输入合法性检查（数学定义域 / 物理合理性）：
      - N、a、b、hr、Ap、Pt、S、vs 必须为正（N 进入分母和平方根）
      - 送风温度必须在物理合理范围内
      - 人员高度 hp 必须使 (ho - hp + r) 与 (hm - hp + r) 为正（含 5/3 次幂）
    """
    bad = []
    for name, val in [('a', a), ('b', b), ('hr', hr), ('Ap', Ap), ('Pt', Pt),
                      ('N', N), ('S', S), ('vs', vs)]:
        if not (np.isfinite(val) and val > 0):
            bad.append(f"{name} must be positive (got {val})")
    if not (np.isfinite(ts) and 15 <= ts <= 24):
        bad.append(f"ts must be within [15, 24] °C (got {ts})")
    if not bad:
        r_ = De / (2 * tandg(12.5))
        hm = hr - 0.1
        if not (hnf < hp < hm and (ho - hp + r_) > 0 and (hm - hp + r_) > 0):
            bad.append(f"hp = {hp} m is outside the admissible range "
                       f"({hnf} m, {min(hm, ho + r_):.2f} m) for the plume model")
    if bad:
        raise ValueError("; ".join(bad))

def calc_PD(toz, v=0.3, Tu=10.0):
    """
    吹风感不满意率 DR (%)  —— Fanger et al. 1988 / ISO 7730
        DR = (34 - ta) * (v - 0.05)^0.62 * (0.37 * v * Tu + 3.14)

    固定值说明：
      ta = toz   局部空气温度 (°C)，取模型算出的工作区温度
      v  = U     局部平均风速 (m/s)，与 PMV 共用 U（run_batch 中的 U_list，默认 0.3）
      Tu = 10    湍流强度 (%)，假设值（常用范围 10~60），需要时在此处修改默认值
      v < 0.05 m/s 时按 0.05 计；DR 限制在 [0, 100] %
    """
    v = max(v, 0.05)
    DR = (34.0 - toz) * (v - 0.05) ** 0.62 * (0.37 * v * Tu + 3.14)
    return min(max(DR, 0.0), 100.0)


def calc_E(te, toz, ts):
    """能量利用系数 E = (te - ts) / (toz - ts) 

    E 是“空气温度/能量利用有效度”类指标，不是 HVAC 总能效
                 （不含风机功率、盘管负荷、冷机性能等），解读时请勿等同于节能量。
                 分母为零（toz == ts）时返回 nan，避免除零。
    """
    denom = toz - ts
    if abs(denom) < 1e-9:
        return float('nan')
    return (te - ts) / denom


# ── PMV / PPD  (ISO 7730 / ASHRAE 55) ──────────────────────

def calc_pa(ta, RH):
    """水蒸气分压 (Pa)，Magnus公式"""
    return RH / 100.0 * 610.78 * math.exp(17.27 * ta / (ta + 237.3))


def calc_PMV(toz, tf, tc, tp, a, b, N, Ap, M=58.15, W=0.0, I_cl=0.124, U=0.3, RH=50.0):
    """Fanger PMV/PPD，tr 由地板/天花板/人体面积加权估算
        平均辐射温度 tr 的说明：
      tr = (tf·A_floor + tc·A_ceiling + tp·N·Ap) / (A_floor + A_ceiling + N·Ap)
      即仅考虑地板、天花板、人体表面三类表面的面积加权平均；
      四周墙面温度与角系数在本模型中未单独求解，因此未计入。
    M、W、I_cl、U、RH 均为函数参数，app.py 中已改为用户可编辑，
      默认值保持不变（M=58.15 W/m²=1.0 met）。
    """
    ta  = toz
    MW  = M - W
    # tr 面积加权 (Eq. tr estimate)
    Ab  = a * b
    tr  = (tf * Ab + tc * Ab + tp * N * Ap) / (2 * Ab + N * Ap)
    pa  = calc_pa(ta, RH)
    #   f_cl  (I_cl 以 m²K/W 为单位，临界值 0.078 m²K/W ≈ 0.5 clo)
    f_cl = (1.00 + 1.290 * I_cl) if I_cl <= 0.078 else (1.05 + 0.645 * I_cl)
    t_cl = 34.0
    converged = False
    for _ in range(150):
        h_c = max(2.38 * abs(t_cl - ta) ** 0.25, 12.1 * math.sqrt(U))
        t_new = (35.7 - 0.028 * MW
                 - I_cl * (3.96e-8 * f_cl * ((t_cl + 273) ** 4 - (tr + 273) ** 4)
                           + f_cl * h_c * (t_cl - ta)))
        if abs(t_new - t_cl) < 1e-6:
            t_cl = t_new
            converged = True
            break
        t_cl = 0.5 * (t_cl + t_new)   # 欠松弛，抑制震荡发散
    else:
        import warnings
        warnings.warn(
            f"t_cl iteration did not converge (last t_cl={t_cl:.3f}, "
            f"ta={ta:.2f}, I_cl={I_cl}, U={U})"
        )

    h_c = max(2.38 * abs(t_cl - ta) ** 0.25, 12.1 * math.sqrt(U))
    #  PMV
    PMV = (0.303 * math.exp(-0.036 * M) + 0.028) * (
        MW
        - 3.05e-3 * (5733 - 6.99 * MW - pa)
        - 0.42  * (MW - 58.15)
        - 1.7e-5 * M * (5867 - pa)
        - 0.0014 * M * (34 - ta)
        - 3.96e-8 * f_cl * ((t_cl + 273) ** 4 - (tr + 273) ** 4)
        - f_cl * h_c * (t_cl - ta)
    )
    # PMV 超出 Fanger 模型有效范围 [-3, 3] 时截断取边界值
    # （超出该范围 PPD 曲线已无实际意义，ISO 7730 / ASHRAE 55 均以此为界）
    PMV = max(-3.0, min(3.0, PMV))
    PPD = 100 - 95 * math.exp(-0.03353 * PMV**4 - 0.2179 * PMV**2)
    return PMV, PPD


#  统一的 PMV 等级与阈值（供界面和批量共用）
#   ASHRAE 55-2023 ：|PMV| ≤ 0.5
#   ISO 7730:2025  ：Category A |PMV| ≤ 0.2 ；B ≤ 0.5 ；C ≤ 0.7
def pmv_category(pmv):
    """返回 (ASHRAE 55 是否满足, ISO 7730 等级 'A'/'B'/'C'/'outside')"""
    p = abs(pmv)
    ashrae_ok = p <= 0.5
    if p <= 0.2:   iso = 'A'
    elif p <= 0.5: iso = 'B'
    elif p <= 0.7: iso = 'C'
    else:          iso = 'outside'
    return ashrae_ok, iso

# ─────────────────────────────────────────────────────────────────────────────


class ThermalModel:
    # ── 固定参数 ──
    φl = 0.8;  Ql = 250.0;  ho = 2.0
    Fp_f = 0.5;  Fp_c = 0.5
    hrp = hrf = hrc = 4.7
    cp = 1002;  De = 0.4;  hnf = 0.1;  ρ = 1.2
    k_tpm = k_tpe = 1.0;  g = 9.81

    def set_case(self, a, b, hr, Ap, Pt, N, ts, vs, S, hp,
                 M=58.15, W=0.0, I_cl=0.124, U=0.3, RH=50.0):
        # 输入校验：非法输入直接报错，不再静默计算
        validate_inputs(a, b, hr, Ap, Pt, N, ts, vs, S, hp,
                        De=self.De, ho=self.ho, hnf=self.hnf)
        self.a, self.b, self.hr = a, b, hr
        self.Ap, self.Pt, self.N = Ap, Pt, N
        self.ts, self.vs, self.S, self.hp = ts, vs, S, hp
        self.M, self.W, self.I_cl, self.U, self.RH = M, W, I_cl, U, RH

        self.Af   = a * b
        self.hm   = hr - 0.1
        #  此处 4.2、0.4 为标定房间的固定几何量
        self.De_A = (4.2**2 -(0.4 + (4.2/(N+4)**0.5 - 1)*2 - 0.4)**2) / 4.2
        self.Qs   = S * vs
        self.ACH  = self.Qs * 3600 / (a * b * hr)
        self.ms   = self.Qs * 1.2
        self.Ff_c = a * b / (a * b + Ap * N)
        self.Ff_p = 1.0 - self.Ff_c
        self.Fc_f = self.Ff_c
        self.Fc_p = 1.0 - self.Fc_f

    def equations_system(self, T):
        tf, tnf, toz, tpo, tp, tmz, tpm, te, tpe, tc = T
        residuals = np.zeros(10)
        eps = 1e-10

        Ar   = self.g * self.hr * (te - self.ts) / ((toz + 273.15) * (self.Qs / self.Af)**2)
        # 25.0 为标定房间地面面积
        Zr   = (25.0 / self.N)**0.5 / 1.71
        #  ln(Ar) 要求 Ar > 0。求解迭代中若 Ar ≤ 0（te ≤ ts 等非物理状态）
        #   用极小正数截断以避免 nan/-inf 破坏迭代；Ar > 0 时方程与原来完全相同。
        #   最终解是否落在 Ar > 0 的物理区域，由 solve() 中的合理性检查负责判定。
        Ar   = max(Ar, 1e-12)
        lnAr = np.log(Ar)

        k_tnfo = -14.08 + 5.68 * lnAr - 0.37 * lnAr**2 + 0.15 * Zr
        k_tom  = -12.84 + 3.66 * lnAr - 0.19 * lnAr**2 - 4.09 * Zr
        k_tpo  =   8.82 + 0.62 * lnAr - 0.04 * lnAr**2 - 0.2  * Zr
        k_tme  =  14.86 - 0.15 * lnAr + 0.01 * lnAr**2 - 5.94 * Zr

        d_f   = max(abs(tf  - tnf),     eps)
        d_ts  = max(abs(tf  - self.ts), eps)
        d_c   = max(abs(te  - tc),      eps)
        d_po  = max(abs(tp  - tpo),     eps)
        d_pnf = max(abs(tp  - tnf),     eps)
        d_oz  = max(abs(tp  - toz),     eps)

        hconv_f = ((0.704 / (2*self.a*self.b/(self.a+self.b))**0.601 * d_f**0.133)**6
                   + ((d_ts/d_f) * 0.48 * self.ACH**0.8)**6) ** (1/6)

        _base = 0.0055 * (0.5*self.N*self.Pt)**(1/3) * self.ρ
        _r    = self.De / (2*tandg(12.5))
        mp_oz = _base * (self.ho - self.hp + _r)**(5/3)
        mp_hm = _base * (self.hm - self.hp + _r)**(5/3)

        Ab = self.a * self.b
        residuals[0] = (hconv_f*Ab*(tnf-tf)
                        + self.Ff_p*self.hrf*Ab*(tp-tf)
                        + 0.5*self.φl*self.Ql
                        + self.Ff_c*self.hrf*Ab*(tc-tf))

        residuals[1] = ((0.704/(2*Ab/(self.a+self.b))**0.601*d_c**0.133)*Ab*(te-tc)
                        + self.Fc_p*self.hrc*Ab*(tp-tc)
                        + 0.5*self.φl*self.Ql
                        + self.Fc_f*self.hrc*Ab*(tf-tc))

        residuals[2] = (self.N*self.Pt
                        - self.Fp_f*self.hrp*self.N*self.Ap*(tp-tf)
                        - self.Fp_c*self.hrp*self.N*self.Ap*(tp-tc)
                        - 0.3*1.183*d_po**0.347*self.N*(_r+0.4)*1.6*(tp-tpo)
                        - 5.541*1.183*d_oz**0.347*self.N*(self.hp-_r-self.hnf)*1.6*(tp-toz)
                        - 1.183*d_pnf**0.347*self.N*self.hnf*1.6*(tp-tnf))

        residuals[3] = (self.cp*self.ms*(self.ts-tnf)
                        + k_tnfo*Ab*(toz-tnf)
                        - hconv_f*Ab*(tnf-tf)
                        + 1.183*d_pnf**0.347*self.N*self.hnf*1.6*(tp-tnf))

        residuals[4] = (self.cp*self.ms*(tnf-toz)
                        + k_tom*Ab*(tmz-toz)
                        - k_tnfo*Ab*(toz-tnf)
                        - k_tpo*np.pi*self.De_A/self.De
                          * ((self.ho+self.hnf)/2+_r-self.hp)*2*tandg(12.5)*(self.ho-self.hnf)*(toz-tpo)
                        + 5.541*1.183*d_oz**0.347*self.N*(self.hp-_r-self.hnf)*1.6*(tp-toz))

        residuals[5] = (self.cp*mp_oz*(toz-tpo)
                        + k_tpo*np.pi*self.De_A/self.De
                          * ((self.ho+self.hnf)/2+_r-self.hp)*2*tandg(12.5)*(self.ho-self.hnf)*(toz-tpo)
                        + 0.3*1.183*d_po**0.347*self.N*(_r+0.4)*1.6*(tp-tpo))

        residuals[6] = (self.cp*(self.ms-mp_oz)*(toz-tmz)
                        - k_tme*Ab*(tmz-te)
                        - k_tom*Ab*(tmz-toz)
                        - self.k_tpm*np.pi*self.De_A/self.De
                          * ((self.hm+self.ho)/2+_r-self.hp)*2*tandg(12.5)*(self.hm-self.ho)*(tmz-tpm))

        residuals[7] = (self.cp*mp_oz*(tpo-tpm)
                        + self.cp*(mp_hm-mp_oz)*(tmz-tpm)
                        + self.k_tpm*np.pi*self.De_A/self.De
                          * ((self.hm+self.ho)/2+_r-self.hp)*2*tandg(12.5)*(self.hm-self.ho)*(tmz-tpm))

        residuals[8] = self.cp*self.ms*(te-self.ts) - (self.N*self.Pt + self.Ql)

        residuals[9] = (self.cp*mp_hm*(tpm-tpe)
                        + self.k_tpe*np.pi*self.De_A/self.De
                          * ((self.hr+self.hm)/2+_r-self.hp)*2*tandg(12.5)*(self.hr-self.hm)*(te-tpe))

        return residuals

    #  收敛判据与多初值
    RES_TOL = 1e-3          # 残差(∞范数)收敛容差 [W]
    NAMES = ['tf','tnf','toz','tpo','tp','tmz','tpm','te','tpe','tc']
    T0_DEFAULT = np.array([20, 20, 22, 25, 35, 25, 28, 30, 30, 25], dtype=float)
    # 备用初值：默认初值整体平移（仅在默认初值失败时依次尝试）
    T0_SHIFTS = (0.0, -3.0, 3.0, 6.0, 10.0)

    def _is_plausible(self, T):
        """
       物理合理性检查（不仅依赖 fsolve 的返回标志）：
          - 所有温度有限，且位于 [ts-1, 50] °C 内；
          - 排风区温度 te > ts（Ar > 0，ln(Ar) 有定义）。
        """
        T = np.asarray(T, dtype=float)
        if not np.all(np.isfinite(T)):
            return False
        te = T[self.NAMES.index('te')]
        if te <= self.ts:
            return False
        return bool(np.all(T >= self.ts - 1.0) and np.all(T <= 50.0))

    def solve(self, T0=None, multi_start=True):
        """
        收敛 = fsolve 返回标志 ier==1  AND  残差∞范数 < RES_TOL  AND  物理合理。
        默认初值失败时，自动尝试 T0_SHIFTS 中的备用初值（multi_start=True）。
        结果中新增：res_norm（残差∞范数）、n_start（使用的初值序号，1=默认）。
        """
        if T0 is None:
            starts = [self.T0_DEFAULT + s for s in (self.T0_SHIFTS if multi_start else (0.0,))]
        else:
            starts = [np.asarray(T0, dtype=float)]

        sol, ok, res_norm, n_used = None, False, float('nan'), 0
        for k, T_init in enumerate(starts, 1):
            sol_k, _, ier, _ = fsolve(self.equations_system, T_init, full_output=True)
            rn = float(np.max(np.abs(self.equations_system(sol_k))))
            ok_k = (ier == 1) and (rn < self.RES_TOL) and self._is_plausible(sol_k)
            if sol is None or ok_k:
                sol, res_norm, n_used = sol_k, rn, k
            if ok_k:
                ok = True
                break

        res = {**dict(zip(self.NAMES, sol)), 'converged': ok,
               'res_norm': res_norm, 'n_start': n_used}
        # ── PMV/PPD 附加到结果 ──
        if res['converged']:
            res['PMV'], res['PPD'] = calc_PMV(
                res['toz'], res['tf'], res['tc'], res['tp'],
                self.a, self.b, self.N, self.Ap,
                M=self.M, W=self.W, I_cl=self.I_cl, U=self.U, RH=self.RH)
            res['E'] = calc_E(res['te'], res['toz'], self.ts)
        else:
            res['PMV'] = res['PPD'] = res['E'] = float('nan')
        return res

    def sensitivity_to_initial_guess(self, shifts=(-6.0, -3.0, 0.0, 3.0, 6.0, 10.0)):
        """
          初值敏感性：用不同初值求解，返回各次收敛情况和 toz，
        用于检查是否存在多重根（各 toz 的极差应接近 0）。
        """
        out = []
        for s in shifts:
            r = self.solve(T0=self.T0_DEFAULT + s, multi_start=False)
            out.append((s, r['converged'], r['toz'], r['res_norm']))
        return out


class BatchSolver:
    TOZ_MIN, TOZ_MAX = 24.0, 28.0   # 工作区温度合格范围 (°C)
    PD_MIN,  PD_MAX  =  0.0, 20.0   # 草稿热不舒适度合格范围 (%)
    PMV_MIN, PMV_MAX = -0.5,  0.5   # PMV 合格范围 (ISO 7730 Category B)
    # 以上判据为默认值；app.py 中已改为用户可配置。

    def __init__(self):
        self.df = pd.DataFrame()

    def run_batch(self, a_list, b_list, hr_list, Ap_list, Pt_list,
                  N_list, ts_list, vs_list, S_list, hp_list,
                  # ── PMV 批量参数 ──
                  M_list=[58.15], W_list=[0.0], I_cl_list=[0.124],
                  U_list=[0.3], RH_list=[50.0],
                  verbose=True):

        all_lists  = [a_list, b_list, hr_list, Ap_list, Pt_list,
                      N_list, ts_list, vs_list, S_list, hp_list,
                      M_list, W_list, I_cl_list, U_list, RH_list]
        col_names  = ['a','b','hr','Ap','Pt','N','ts','vs','S','hp',
                      'M','W','I_cl','U','RH']
        col_units  = ['m','m','m','m²','W','','°C','m/s','m²','m',
                      'W/m²','W/m²','m²K/W','m/s','%']
        cases = list(product(*all_lists))

        print(f"总工况数: {len(cases)}")
        for n, lst, u in zip(col_names, all_lists, col_units):
            print(f"  {n:5s}: {min(lst)} ~ {max(lst)} {u}")
        print(f"合格判定: toz {self.TOZ_MIN}~{self.TOZ_MAX}°C | "
              f"PD {self.PD_MIN}~{self.PD_MAX}% | PMV {self.PMV_MIN}~{self.PMV_MAX}")
        print("=" * 70)

        rows = []
        temp_cols = ['tf','tnf','toz','tpo','tp','tmz','tpm','te','tpe','tc']
        t_batch0 = time.perf_counter()          # 批量总计时

        for idx, vals in enumerate(cases, 1):
            a,b,hr,Ap,Pt,N,ts,vs,S,hp,M,W,I_cl,U,RH = vals
            if verbose:
                print(f"[{idx}/{len(cases)}] N={N}, ts={ts}°C, vs={vs}m/s ... ", end='', flush=True)

            model = ThermalModel()
            try:
                t_case0 = time.perf_counter()   #  单工况计时
                #  set_case 内含输入校验，移入 try，非法工况只记录错误不中断批量
                model.set_case(a,b,hr,Ap,Pt,N,ts,vs,S,hp, M=M,W=W,I_cl=I_cl,U=U,RH=RH)
                res   = model.solve()
                runtime = time.perf_counter() - t_case0
                toz_v = res['toz']
                pd_v  = calc_PD(toz_v, U) if res['converged'] else float('nan')   # 风速取 U，与 PMV 一致
                pmv_v = res['PMV']

                ok = (res['converged']
                      and self.TOZ_MIN <= toz_v <= self.TOZ_MAX
                      and self.PD_MIN  <= pd_v  <= self.PD_MAX
                      and self.PMV_MIN <= pmv_v <= self.PMV_MAX)

                #  标记是否外推（超出标定范围）
                extrap = len(check_validity_domain(a, b, N, ts, vs)) > 0

                row = dict(zip(col_names, vals))
                row.update(Qs=model.Qs, ACH=model.ACH,
                           converged=res['converged'], toz_qualified=ok,
                           extrapolated=extrap,
                           PD=pd_v, PMV=pmv_v, PPD=res['PPD'], E=res['E'],
                           res_norm=res['res_norm'], n_start=res['n_start'],   
                           runtime_s=runtime)                          
                row.update({k: res[k] for k in temp_cols})
                rows.append(row)

                if verbose:
                    if res['converged']:
                        # 温度统一保留 1 位小数
                        print(f"{'✓' if ok else '△'} toz={toz_v:.1f}°C  PD={pd_v:.2f}%  "
                              f"PMV={pmv_v:.2f}  E={res['E']:.3f}  [{'合格' if ok else '不合格'}]"
                              f"{'  [外推]' if extrap else ''}")
                    else:
                        print("✗ 未收敛")
            except Exception as e:
                print(f"✗ 错误: {e}")

        self.total_runtime = time.perf_counter() - t_batch0
        self.df = pd.DataFrame(rows)
        n_ok = self.df['toz_qualified'].sum()
        print("=" * 70)
        print(f"完成! 收敛: {self.df['converged'].sum()}/{len(cases)}  合格: {n_ok}/{len(cases)}")
        return self.df

    def get_qualified(self):
        return self.df[self.df['toz_qualified']].reset_index(drop=True)

    def save_csv(self, filename, qualified_only=False):
        df = self.get_qualified() if qualified_only else self.df
        df.to_csv(filename, index=False, float_format='%.4f')
        print(f"{'合格' if qualified_only else '全部'}工况 → {filename}  ({len(df)} 条)")
        return df

    def best_by_E(self, tol=1e-3):
        """
        合格工况中 E 最大者；E 相差不超过 tol 视为并列。
        返回 (DataFrame, 是否唯一)。E 仅为温度利用有效度指标，不是总能效，
        是否据此“推荐”工况需谨慎（见 calc_E 注释）。
        """
        df_q = self.get_qualified()
        if not len(df_q):
            return df_q, False
        top = df_q[df_q['E'] >= df_q['E'].max() - tol]
        return top, len(top) == 1

    def summary_stats(self):
        df_q = self.get_qualified()
        dc   = self.df[self.df['converged']]
        print(f"\n汇总统计:")
        print(f"  收敛: {len(dc)}/{len(self.df)}  合格: {len(df_q)}/{len(self.df)}")
        # 外推工况统计
        n_ext = int(self.df['extrapolated'].sum())
        n_q_in = int((self.df['toz_qualified'] & ~self.df['extrapolated']).sum())
        print(f"  外推工况（超出标定范围）: {n_ext}/{len(self.df)}  "
              f"范围内合格: {n_q_in}  外推合格: {len(df_q) - n_q_in}")
        #  收敛质量
        print(f"  最大残差(∞范数): {self.df['res_norm'].max():.2e}  "
              f"使用备用初值的工况: {int((self.df['n_start'] > 1).sum())}")
        #  计时
        print(f"  计算时间: 总计 {self.total_runtime:.2f} s，单工况平均 "
              f"{self.df['runtime_s'].mean() * 1000:.1f} ms")
        print(f"  toz [全部收敛]: {dc['toz'].min():.1f}~{dc['toz'].max():.1f} °C")   # 温度 1 位小数
        if len(df_q):
            for col, unit in [('toz','°C'),('te','°C'),('ACH','h⁻¹'),('PD','%'),('PMV','-'),('PPD','%'),('E','-')]:
                dec = 1 if unit == '°C' else (3 if col == 'E' else 2)
                print(f"  {col:3s} [合格]: {df_q[col].min():.{dec}f}~{df_q[col].max():.{dec}f} {unit}")
            #  合格工况中 E 最大者及是否并列
            top, unique = self.best_by_E()
            print(f"  E 最大的合格工况: {len(top)} 个（{'唯一' if unique else '并列'}）")


def print_environment():
    """  记录运行环境，便于论文报告基准测试条件"""
    import scipy
    print(f"Python {sys.version.split()[0]} | NumPy {np.__version__} | "
          f"SciPy {scipy.__version__} | pandas {pd.__version__} | {platform.platform()}")


# ── 主程序 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 原有热力学参数
    a_list  = [5.0];    b_list  = [5.0];    hr_list = [3.6]
    Ap_list = [2.56];   Pt_list = [120.0];  N_list  = [32]
    #  演示扫描范围限定在标定范围内
    ts_list = [15, 16, 17, 18, 19]
    vs_list = [1.6, 1.9, 2.2, 2.5, 2.8]
    S_list  = [0.170625];  hp_list = [1.5]

    # PMV 固定参数（坐姿 / 0.8clo / RH50% / U0.3m/s）
    # M=58.15 W/m²  W=0.0  I_cl=0.124 m²K/W  U=0.3 m/s  RH=50%
    # 吹风感 DR 固定参数：v=U=0.3 m/s（与 PMV 共用），Tu=10 %（在 calc_PD 中设定）

    print_environment()          
    solver = BatchSolver()
    solver.run_batch(a_list, b_list, hr_list, Ap_list, Pt_list,
                     N_list, ts_list, vs_list, S_list, hp_list)
    solver.summary_stats()
    solver.save_csv('thermal_model_results_all.csv')
    solver.save_csv('thermal_model_results_qualified.csv', qualified_only=True)

    df_q = solver.get_qualified()
    if len(df_q):
        print("\n符合要求的工况:")
        #  同时列出 E 以及是否外推
        print(df_q[['N','ts','vs','ACH','toz','PD','PMV','PPD','E',
                    'te','tf','tc','extrapolated']].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("\n无符合要求的工况。")
