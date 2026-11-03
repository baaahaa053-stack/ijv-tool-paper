# IJV Thermal Comfort Tool

**English summary**: An open-source Streamlit web app for rapid indoor thermal environment prediction and thermal comfort assessment in impinging jet ventilation (IJV) systems. It solves a four-zone, ten-node heat balance model and couples PMV/PPD, draught rate (PD), and energy utilization coefficient (E) calculations in a single workflow. Try it online at https://ijvtool.com/. To run locally: `pip install -r requirements.txt` then `streamlit run app.py`. Tested with Python 3.10 and 3.11; dependency versions are pinned in `requirements.txt`.

---

碰撞射流通风（Impinging Jet Ventilation, IJV）室内热环境快速预测与热舒适一体化评估工具。基于四区十节点热平衡模型求解室内竖向温度分层，并在同一流程中耦合完成 Fanger PMV/PPD、局部吹风感不满意率（PD）及送风能量利用系数（E）的计算。

在线体验：https://ijvtool.com/

## 功能

- 输入房间几何、人员参数、送风参数，秒级求解室内十个特征节点温度
- 自动计算 PMV、PPD、局部吹风感不满意率 PD、送风能量利用系数 E
- 按预设舒适性判据（24 °C ≤ toz ≤ 28 °C、PD ≤ 20%、|PMV| ≤ 0.5）自动判定工况是否合格
- 批量筛选模式：设定送风温度、送风速度取值区间与步长，自动生成全部参数组合并批量求解，以热力图标注合格域
- 计算结果支持表格查看与 CSV 导出

## 本地运行

```bash
git clone https://github.com/baaahaa053-stack/ijv-tool-paper.git
cd ijv-tool-paper
pip install -r requirements.txt
streamlit run app.py
```

运行后浏览器会自动打开本地地址（默认 http://localhost:8501），界面与在线版本一致。

已在 Python 3.10 和 3.11 下测试。`requirements.txt` 中的依赖版本固定为经测试的确切版本，与论文第 2.1 节所述的部署环境一致。

## 输入参数

| 参数 | 说明 | 单位 |
|---|---|---|
| 房间长、宽、高 | 房间几何尺寸 | m |
| 人数 | 房间内人员数量 | — |
| 单人表面积 | 单个人体表面积 | m² |
| 单人散热功率 | 单个热源散热量 | W |
| 人员高度 | 人体特征高度 | m |
| 送风口面积 | 送风口截面积 | m² |
| 送风温度 | 单工况模式为定值，批量模式为取值区间与步长 | °C |
| 送风速度 | 单工况模式为定值，批量模式为取值区间与步长 | m/s |

人体热舒适固定参数（新陈代谢率、机械功、服装热阻、工作区风速、相对湿度）依据 ISO 7730:2025 与 ASHRAE 55-2023 标准设定。

## 有效范围与外推

经验关联式依据 75 个 CFD 工况标定，有效范围为：送风温度 15–19 °C、送风速度 1.6–2.8 m/s、人员密度 1.28–2.4 persons/m²（5 m × 5 m × 3.6 m 房间）。输入超出该范围时，软件默认阻止计算并给出警告；勾选 "Allow extrapolation" 后才会计算，结果标注为未验证。非正的几何尺寸、人数、送风速度和送风口面积会在求解前被拒绝。

求解结果仅在以下条件同时满足时被接受：fsolve 返回成功、十个方程的最大残差小于 10⁻³ W、各节点温度物理合理（t_e > t_s，且在 t_s − 1 °C 至 50 °C 之间）。默认初值失败时，求解器会依次改用备用初值。

## 输出结果

- 十个特征节点温度：地板、天花板、人体表面及各区空气、热羽流温度
- 送风量、换气次数
- PMV（预测平均热感觉指数）、PPD（预测不满意百分率）
- PD（局部吹风感不满意率）
- E（送风能量利用系数）
- 合格性判定结果（单工况模式），或合格域标注（批量模式）

## 项目结构

```
app.py                 # Streamlit 前端界面与主入口
thermal_model.py       # 四区十节点热平衡模型与热舒适指标计算
robustness_check.py    # 求解器稳健性检验脚本
tests/                 # 自动化回归测试（pytest）
requirements.txt       # Python 依赖（固定版本）
requirements-dev.txt   # 测试依赖
```

## 测试

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v
```

每次 push 和 pull request 时,GitHub Actions 会在 Python 3.10 和 3.11 下自动运行测试。测试涵盖热模型(以论文第 3.1 节单工况算例为参考值)、PMV/PPD、PD 和 E。其中 PMV/PPD 与 pythermalcomfort(ISO 7730)的对照测试需要 pythermalcomfort 4.8.0,该版本要求 Python ≥ 3.11,因此仅在 Python 3.11 下运行,在 3.10 下自动跳过。

## 作为 Python 模块调用（无界面）

计算核心 `thermal_model.py` 独立于 Streamlit 界面，可直接调用：

```python
from thermal_model import ThermalModel

m = ThermalModel()
m.set_case(a=5.0, b=5.0, hr=3.6, Ap=2.56, Pt=120.0, N=32,
           ts=18.0, vs=2.0, S=0.175, hp=1.5)
res = m.solve()
print(res["converged"], res["toz"], res["PMV"], res["PPD"], res["E"])
```

## 求解器稳健性检验

在仓库根目录运行 `python robustness_check.py`，可复现收敛性统计结果
（已验证范围内的 468 个工况，以及 700 组随机初值）。

## 引用

如果本工具对你的研究或工程设计有帮助，欢迎引用【投稿录用后补充正式引用格式，例如作者、期刊、DOI】。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 联系方式

如有问题或建议，欢迎通过 jhu@shou.edu.cn 联系，或在本仓库提交 Issue。
