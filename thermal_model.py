import math
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from itertools import product


def tandg(deg):
    return np.tan(np.deg2rad(deg))


def calc_PD(toz):
    return 1.38e4 * (((0.26 / (toz - 13.7)) + 0.0239) ** 2 - 8.57e-4)


def calc_E(te, toz, ts):
    """能量利用系数 E = (te - ts) / (toz - ts)  (Eq. 11)"""
    return (te - ts) / (toz - ts)


# ── PMV / PPD  (ISO 7730 / ASHRAE 55, Fanger Eqs. 4-7) ──────────────────────

def calc_pa(ta, RH):
    """水蒸气分压 (Pa)，Magnus公式"""
    return RH / 100.0 * 610.78 * math.exp(17.27 * ta / (ta + 237.3))


def calc_PMV(toz, tf, tc, tp, a, b, N, Ap, M=58.15, W=0.0, I_cl=0.124, U=0.3, RH=50.0):
    """Fanger PMV/PPD，tr 由地板/天花板/人体面积加权估算"""
    ta  = toz
    MW  = M - W
    # tr 面积加权 (Eq. tr estimate)
    Ab  = a * b
    tr  = (tf * Ab + tc * Ab + tp * N * Ap) / (2 * Ab + N * Ap)
    pa  = calc_pa(ta, RH)
    # Eq. 7  f_cl
    f_cl = (1.00 + 1.290 * I_cl) if I_cl > 0.078 else (1.05 + 0.645 * I_cl)
    # Eqs. 5-6  迭代求 t_cl, h_c
    t_cl = 34.0
    for _ in range(100):
        h_c = max(2.38 * abs(t_cl - ta) ** 0.25, 1.21 * math.sqrt(U))
        t_new = (35.7 - 0.028 * MW
                 - 0.155 * I_cl * (3.96e-8 * f_cl * ((t_cl+273)**4 - (tr+273)**4)
                                   + f_cl * h_c * (t_cl - ta)))
        if abs(t_new - t_cl) < 1e-6:
            break
        t_cl = t_new
    h_c = max(2.38 * abs(t_cl - ta) ** 0.25, 1.21 * math.sqrt(U))
    # Eq. 4  PMV
    PMV = (0.303 * math.exp(-0.036 * M) + 0.028) * (
        MW
        - 3.05e-3 * (5733 - 6.99 * MW - pa)
        - 0.42  * (MW - 58.15)
        - 1.7e-5 * M * (5867 - pa)
        - 0.0014 * M * (34 - ta)
        - 3.96e-8 * f_cl * ((t_cl+273)**4 - (tr+273)**4)
        - f_cl * h_c * (t_cl - ta)
    )
    # PMV 超出 Fanger 模型有效范围 [-3, 3] 时截断取边界值
    # （超出该范围 PPD 曲线已无实际意义，ISO 7730 / ASHRAE 55 均以此为界）
    PMV = max(-3.0, min(3.0, PMV))
    PPD = 100 - 95 * math.exp(-0.03353 * PMV**4 - 0.2179 * PMV**2)
    return PMV, PPD

# ─────────────────────────────────────────────────────────────────────────────


class ThermalModel:
    # ── 固定参数 ──
    φl = 0.8;  Ql = 250.0;  ho = 2.0
    Fp_f = 0.5;  Fp_c = 0.5
    hrp = hrf = hrc = 4.7
    cp = 1002;  De = 0.4;  hnf = 0.1;  ρ = 1.2
    k_tpm = k_tpe = 1.0;  g = 9.81

    def set_case(self, a, b, hr, Ap, Pt, N, ts, vs, S, hp,
                 # ── PMV 附加参数 (默认: 上海夏季 / 站立轻活动 / 0.5clo) ──
                 M=58.15, W=0.0, I_cl=0.124, U=0.3, RH=50.0):
        self.a, self.b, self.hr = a, b, hr
        self.Ap, self.Pt, self.N = Ap, Pt, N
        self.ts, self.vs, self.S, self.hp = ts, vs, S, hp
        self.M, self.W, self.I_cl, self.U, self.RH = M, W, I_cl, U, RH

        self.Af   = a * b
        self.hm   = hr - 0.1
        self.De_A = (4.2**2 - (0.4 + (4.2/(N+4)**0.5 - 1)*2 - 0.4)**2) / 4.2
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
        Zr   = (25.0 / self.N)**0.5 / 1.71
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

    def solve(self, T0=None):
        if T0 is None:
            T0 = np.array([20, 20, 22, 25, 35, 25, 28, 30, 30, 25], dtype=float)
        sol, _, ier, _ = fsolve(self.equations_system, T0, full_output=True)
        names = ['tf','tnf','toz','tpo','tp','tmz','tpm','te','tpe','tc']
        res = {**dict(zip(names, sol)), 'converged': ier == 1}
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


class BatchSolver:
    TOZ_MIN, TOZ_MAX = 24.0, 28.0   # 工作区温度合格范围 (°C)
    PD_MIN,  PD_MAX  =  0.0, 20.0   # 草稿热不舒适度合格范围 (%)
    PMV_MIN, PMV_MAX = -0.5,  0.5   # PMV 合格范围 (ISO 7730 Category B)

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
        col_units  = ['m','m','m','m²','W','','℃','m/s','m²','m',
                      'W/m²','W/m²','m²K/W','m/s','%']
        cases = list(product(*all_lists))

        print(f"总工况数: {len(cases)}")
        for n, lst, u in zip(col_names, all_lists, col_units):
            print(f"  {n:5s}: {min(lst)} ~ {max(lst)} {u}")
        print(f"合格判定: toz {self.TOZ_MIN}~{self.TOZ_MAX}℃ | "
              f"PD {self.PD_MIN}~{self.PD_MAX}% | PMV {self.PMV_MIN}~{self.PMV_MAX}")
        print("=" * 70)

        rows = []
        temp_cols = ['tf','tnf','toz','tpo','tp','tmz','tpm','te','tpe','tc']

        for idx, vals in enumerate(cases, 1):
            a,b,hr,Ap,Pt,N,ts,vs,S,hp,M,W,I_cl,U,RH = vals
            if verbose:
                print(f"[{idx}/{len(cases)}] N={N}, ts={ts}℃, vs={vs}m/s ... ", end='', flush=True)

            model = ThermalModel()
            model.set_case(a,b,hr,Ap,Pt,N,ts,vs,S,hp, M=M,W=W,I_cl=I_cl,U=U,RH=RH)
            try:
                res   = model.solve()
                toz_v = res['toz']
                pd_v  = calc_PD(toz_v) if res['converged'] else float('nan')
                pmv_v = res['PMV']

                ok = (res['converged']
                      and self.TOZ_MIN <= toz_v <= self.TOZ_MAX
                      and self.PD_MIN  <= pd_v  <= self.PD_MAX
                      and self.PMV_MIN <= pmv_v <= self.PMV_MAX)

                row = dict(zip(col_names, vals))
                row.update(Qs=model.Qs, ACH=model.ACH,
                           converged=res['converged'], toz_qualified=ok,
                           PD=pd_v, PMV=pmv_v, PPD=res['PPD'], E=res['E'])
                row.update({k: res[k] for k in temp_cols})
                rows.append(row)

                if verbose:
                    if res['converged']:
                        print(f"{'✓' if ok else '△'} toz={toz_v:.2f}℃  PD={pd_v:.2f}%  "
                              f"PMV={pmv_v:.2f}  E={res['E']:.3f}  [{'合格' if ok else '不合格'}]")
                    else:
                        print("✗ 未收敛")
            except Exception as e:
                print(f"✗ 错误: {e}")

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

    def summary_stats(self):
        df_q = self.get_qualified()
        dc   = self.df[self.df['converged']]
        print(f"\n汇总统计:")
        print(f"  收敛: {len(dc)}/{len(self.df)}  合格: {len(df_q)}/{len(self.df)}")
        print(f"  toz [全部收敛]: {dc['toz'].min():.2f}~{dc['toz'].max():.2f} ℃")
        if len(df_q):
            for col, unit in [('toz','℃'),('te','℃'),('ACH','h⁻¹'),('PD','%'),('PMV','-'),('PPD','%'),('E','-')]:
                print(f"  {col:3s} [合格]: {df_q[col].min():.2f}~{df_q[col].max():.2f} {unit}")


# ── 主程序 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 原有热力学参数
    a_list  = [5.0];    b_list  = [5.0];    hr_list = [3.6]
    Ap_list = [2.56];   Pt_list = [120.0];  N_list  = [32]
    ts_list = [15, 16, 17, 18, 19, 20]
    vs_list = [1.6, 1.9, 2.2, 2.5, 2.8]
    S_list  = [0.170625];  hp_list = [1.5]

    # PMV 固定参数（坐姿 / 0.8clo / RH50% / U0.3m/s）
    # M=58.15 W/m²  W=0.0  I_cl=0.124 m²K/W  U=0.3 m/s  RH=50%

    solver = BatchSolver()
    solver.run_batch(a_list, b_list, hr_list, Ap_list, Pt_list,
                     N_list, ts_list, vs_list, S_list, hp_list)
    solver.summary_stats()
    solver.save_csv('thermal_model_results_all.csv')
    solver.save_csv('thermal_model_results_qualified.csv', qualified_only=True)

    df_q = solver.get_qualified()
    if len(df_q):
        print("\n符合要求的工况:")
        print(df_q[['N','ts','vs','ACH','toz','PD','PMV','PPD','E','te','tf','tc']].to_string(index=False))
    else:
        print("\n无符合要求的工况。")
