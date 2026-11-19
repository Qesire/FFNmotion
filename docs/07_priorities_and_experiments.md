# 07 · 大规模企业技术调查后的研究路线修订

> 2026-11-19，FFNmotion 独立解释性项目。**本文件仅修改研究优先级和实验定义；不改变已有推导的数学有效范围、不把 FFNDS 的加速任务并入本项目。**

## 一、从企业报告得到的三个真正的研究约束

1. **Renderer、simulator、planner 是不同任务对象。** [World Labs 2026 taxonomy](https://worldlabs.ai/blog/taxonomy-of-world-models) 把分别输出观测、状态、动作的模型区分开；[Cosmos 3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/) 和 [Praxis-1](https://runway.com/research/praxis-1-video-pretraining-to-robot-control) 展示它们可能在同一系统结合。**要对 Wan2.1 的物理能力作解释，先说明目标是文本运动遵循、视频中的物理合理性，还是对外部物理条件的反事实响应。**
2. **语义条件链是必需对照。** [Kling 2026 understanding stack](https://arxiv.org/abs/2611.04916)、[Seedance 1.0](https://seed.bytedance.com/en/public_papers/seedance-1-0-exploring-the-boundaries-of-video-generation-models) 说明数据 caption、文本编码器/指令增强、reward 与 generator 均可能影响运动。**Head → FFN 的解释不能忽略源条件是否表达了目标行为。**
3. **物理外推必须是独立测试。** [ByteDance PhyWorld](https://seed.bytedance.com/public_papers/how-far-is-video-generation-from-world-model-a-physical-law-perspective) 与最新 [PhysResponse](https://arxiv.org/abs/2611.24419) 提供 ID/OOD、外观-动力学冲突、物理参数响应、事件边界测试。**找到运动方向与证明物理定律泛化之间存在明确的证据断层。**

## 二、改进研究问题树

### R1. 哪里首先出现具有因果影响的动作信息？

- **输入层**：文本/参考条件中有没有可区分的动作语义？在文本 encoder 输出和 cross-attention head write 中，能否区分 move-left/move-right、rise/fall 与 no-motion？
- **中间层**：同一条件下 \(U_h\) 被 LN/AdaLN、FFN 和下游 attention 转换时，产生什么方向、写入量与消融效应？
- **输出层**：这些因果路径是否改变正确对象的轨迹及指令遵循，而非仅改变背景/镜头或物体身份？

对照：同义 prompt、反义动作、相同运动不同颜色/大小、相同视觉元素不同动作。

### R2. 运动功能通过原始通道、分布式方向还是跨层协同表达？

- 使用 Wan 原始 FFN 的精确有限差分、写入抵消和因果路径 patching；
- 首先比较直接 head write、完整 FFN write、Top-K 通道集合和下游 attention；
- 原始通道不稳定时再做输出子空间与 orthogonal/locally nonlinear probes；仅在真实必要时训练 transcoder；
- 对与运动任务无关的图像/外观评分设置匹配对照，避免只追到强响应的视觉特征。

### R3. 模型是否把运动表面形态和物理状态变量区分开？

- 首先测基本速度、方向、事件与相机独立性；其次测可控/有效的参数变化。
- 在合成受控环境中明确 \(g,\mu,v_0\) 与事件真值，使用真实数据标签训练/检查 probe；在 Wan 自由生成中报告与该分布的 gap。
- 再做**同一运动规律不同外观**和**同一外观不同运动规律**的反事实配对。
- 如果没有可靠的数值条件接口，严禁把 prompt 里的“twice the gravity”当作模型真的接受了数值 \(2g\)；先做动作/趋势级能力测试。

### R4. 动态信息能否跨去噪步、长帧距和交互控制复用？

- Wan2.1: \(s\)=去噪步、\(f\)=生成视频帧、\(\ell\)=网络层；同一步轨迹因果和跨步 sampler state 必须分开。
- 通过 [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay)、[Cosmos Simulation](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/) 与 [Genie](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) 解释**history/context/state/action** 在交互式模型中承担的不同信息角色。
- 当基础 Wan 机制已经清楚时，才选一个开放的 action-conditioned 模型进行外推，不把所有闭源产品都作为可运行基础。

## 三、建议的实际执行顺序（经济且有区分力）

| 步骤 | 冻结对象 | 核心干预和对照 | 验收/停止条件 |
|---|---|---|---|
| 0 · 源头预检 | Wan checkpoint、代码、采样器、CFG、head 名单及其来源 | 未干预 patch≈官方 baseline；复核作者头组与单头差异 | 数值身份/真实调用位置通过，否则不进行机制结论 |
| 1 · 语义反事实 | 相同场景、seed、物体/镜头 | 左↔右、落下↔上升、运动↔静止；同义句与颜色/大小变化 | 能在输入/输出端定义独立的“动作改变 vs 表象改变” |
| 2 · Head/FFN 精确差分 | 固定 timestep 与 branch 的 latent | \(\Delta B=U_h+\sum_jD_{hj}\)，归一化/W1/GELU/W2 分量 | 数值恒等式通过；确认每个读取张量的 axis/dtype |
| 3 · 路径因果 | 冻结源头策略与候选集合 | 直接 head、FFN 全量、FFN Top-K、联合、随机/能量匹配 | 下游 denoiser 和完整视频行为形成可区分效应 |
| 4 · 物理/条件检验 | 冻结识别的路径，留出场景/seed | 颜色↔动力学冲突、受控运动模式、接触阈值 | 已确认路径是否对动作和物理响应各自稳健 |
| 5 · 深化机制 | 阶段 3–4 支持的具体路径 | 跨层 attention、子空间、latent action 或 time-conditioned 特征 | 有独立证据支持才继续扩展 |

### 建议的最小成对数据

**探索组**：选至少四类运动：水平平移、自由落体、一次弹跳、接触/阻挡，每类先选两个随机种子。每类设同义改写或反方向/静止对照，尽量保持物体、视角、颜色固定。仅依据有限初始样本选择 head/方向，但这些样本不用于确认性统计。

**确认组**：冻结候选路径和主要任务指标后，重新生成未用 seed 与新场景，至少一个外观变化、一个物理模式变化。事先登记异常输出的失败类型，不以“跟踪失败”的名义剔除物体消失/严重变形视频。

**重要条件**：不要用同一 prompt 的多个相邻帧当作独立样本。以 scenario/prompt × seed 的视频作为配对实验单位；对帧内观测做嵌套/层级汇总。

## 四、从企业报告引入的两种关键新实验

### A. 真未来（oracle）与模型生成未来的功能读出对照

借鉴 [Runway Praxis-1](https://runway.com/research/praxis-1-video-pretraining-to-robot-control)：

1. 用合成视频或明确轨迹标签建立读取器 \(R_\ell(z)\) 预测方向、速度/事件。
2. 在同一噪声阶段比较真实视频编码/加噪的状态与自由生成状态，明确这是两种不同数据分布。
3. 对有/无候选 head、FFN 路径的输出分别施加读出器，并与**最终生成视频的客观轨迹**比较。

**能区分**：“内部表征已有目标运动信息”与“生成后半段/采样器未将信息转换成正确视频”。不能将 oracle 和 generated 差异全部归因于一个模块；仍需要路径干预。

### B. 物理条件敏感性与表观因素不变性

借鉴 [ByteDance PhyWorld](https://seed.bytedance.com/public_papers/how-far-is-video-generation-from-world-model-a-physical-law-perspective) + [PhysResponse](https://arxiv.org/abs/2611.24419)：

固定样本/seed，构造：
- \(\Delta_{\rm appearance}\)：仅改变颜色/背景，运动控制保持；
- \(\Delta_{\rm motion}\)：仅改变动作/初速，外观保持；
- \(\Delta_{\rm event}\)：构造碰撞/未碰撞等临界事件变化。

比较三个差分在 Head write、FFN write、任务 probe、最终视频轨迹中的效应大小与方向。若表观变化造成与运动变化相似的内部“运动通道”响应，则该通道可能是视觉/语义混合特征；继续通过更独立的表示与路径恢复查明原因。

## 五、评价指标和工具

- **Object-level motion**：同一个对象的轨迹、方向、速度变化；相机运动单独估计，并保留身份漂移、遮挡与跟踪失败事件。
- **Event-level physics**：自由飞行、接触、反弹、停止、穿模、形变。不同物理现象使用不同判据。
- **Physical plausibility**：独立人类/规则审查与 [PhyProbe](https://research.nvidia.com/labs/cosmos-lab/phyprobe/) 的连续分数共同报告，而不把模型评估器输出当作地面真值。
- **Out-of-distribution**：在 appearance/object/initial speed/material/camera 上系统性留出因子，避免单纯跨 seed 被误认成 OOD。
- **Compute and fidelity**：干预精确重建误差、动态算子 JVP 近似误差、checkpoint/branch/precision；资源仅用于解释、不是被优化的任务指标。

## 六、新增科学假说

| 标号 | 假说 | 可检验预测 |
|---|---|---|
| H6 | 不同来源的动作信息具有不同功能路径 | 明确文本动作与图像运动条件触发不同 head/FFN write；输出行为可有相同运动 |
| H7 | 运动“表象编码”可与动力学编码分离 | 同色/同类对象但不同速度的条件，会激活具有稳定物理敏感性的方向；若只颜色变化就改变方向读出则反对该分离 |
| H8 | 视频内部正确运动表示与最终错误运动可以共存 | Probe 对真实运动正确、生成视频错误；通过路径恢复/下游模块干预能定位第一次显著偏离 |
| H9 | 分布式运动方向可能是跨层稳定的因果对象 | 同一任务在不同层投影到不同坐标，但其有限幅度干预的下游响应方向保持一致 |
| H10 | 自回归交互的长程错误可能来自记忆/状态而非瞬时 FFN | 固定当前动作、更换历史/记忆上下文可显著改变运动与几何一致性；在 Wan 单步问题中不应混为一谈 |

H6–H10 均标记 HYPOTHESIS。可与已有 H1–H5 组合检验，不要求全做。

## 七、工程交接与证据收束

研究推进只需要三类明确资产：
1. **算子资产**：checkpoint commit、实际张量 axis/shapes/precision、head 与 FFN 差分、数值 oracle；
2. **干预资产**：视频配对 seed、sampler states、branch/time/layer/head、被恢复的具体路径和自动/人工评测；
3. **证据资产**：source URI、版本与技术披露等级、训练/确认样本划分、原始结果和失败类型。

新报告中的“机制关联”必须经过 SOURCE_VERIFIED（真实原文）→ MODULE_TESTED（实际算子）→ GENERATION_TESTED（配对生成）→ REPLICATED（独立样本）逐级提升。未通过某个阶段的效果只标记相应边界，不向下游扩大结论。

本方案按 [05 公司的技术调查](05_industry_technical_reports_2026-11.md) 与 [06 机制知识综合](06_mechanistic_synthesis.md) 修订，不取代 [03 原始实验方案](03_experimental_plan.md) 的局部数值操作契约。**推荐 05 → 06 → 07 → 02 → 03 的新阅读顺序**，从科学问题和外部证据进入，再由具体算子定义约束实施。
