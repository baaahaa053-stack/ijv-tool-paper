# IJV Thermal Comfort Tool

**English summary**: An open-source Streamlit web app for rapid indoor thermal environment prediction and thermal comfort assessment in impinging jet ventilation (IJV) systems. It solves a four-zone, ten-node heat balance model and couples PMV/PPD, draught rate (PD), and energy utilization coefficient (E) calculations in a single workflow. Try it online at https://ijvtool.com/. To run locally: `pip install -r requirements.txt` then `streamlit run app.py`.

---

碰撞射流通风（Impinging Jet Ventilation, IJV）室内热环境快速预测与热舒适一体化评估工具。基于四区十节点热平衡模型求解室内竖向温度分层，并在同一流程中耦合完成 Fanger PMV/PPD、局部吹风感不满意率（PD）及送风能量利用系数（E）的计算。

在线体验：https://ijvtool.com/

## 功能

- 输入房间几何、人员参数、送风参数，秒级求解室内十个特征节点温度
- 自动计算 PMV、PPD、局部吹风感不满意率 PD、送风能量利用系数 E
- 按预设舒适性判据（24 °C ≤ toz ≤ 28 °C、PD ≤ 20%、|PMV| ≤ 0.5）自动判定工况是否合格
- 批量寻优模式：设定送风温度、送风速度取值区间与步长，自动生成全部参数组合并批量求解，以热力图标注合格域
- 计算结果支持表格查看与 CSV 导出

## 本地运行

```bash
git clone https://github.com/baaahaa053-stack/ijv-tool-paper.git
cd ijv-tool-paper
pip install -r requirements.txt
streamlit run app.py
```

运行后浏览器会自动打开本地地址（默认 http://localhost:8501），界面与在线版本一致。

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

## 输出结果

- 十个特征节点温度：地板、天花板、人体表面及各区空气、热羽流温度
- 送风量、换气次数
- PMV（预测平均热感觉指数）、PPD（预测不满意百分率）
- PD（局部吹风感不满意率）
- E（送风能量利用系数）
- 合格性判定结果（单工况模式），或合格域标注（批量模式）

## 项目结构

```
app.py              # Streamlit 前端界面与主入口
thermal_model.py     # 四区十节点热平衡模型与热舒适指标计算
requirements.txt     # Python 依赖
```

## 引用

如果本工具对你的研究或工程设计有帮助，欢迎引用【投稿录用后补充正式引用格式，例如作者、期刊、DOI】。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 联系方式

如有问题或建议，欢迎通过 m2394953640@163.com 联系，或在本仓库提交 Issue。
