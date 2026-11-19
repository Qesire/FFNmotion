# FFNmotion

**Tracing motion-relevant computations in video diffusion transformers**  
视频生成模型中的运动表征、算子转换与因果回路研究。

## 研究问题

文本中指定的运动，如何通过条件注意力写入、FFN 非线性转换、后续自注意力与去噪采样，最终成为视频中的运动行为？

项目关注 **信息存在 → 算子响应 → 因果传递 → 可观测运动** 四个层次。主要目标是解释，而不是加速视频生成、优化稀疏率或开发 FFN 近似算子。

初始平台为 **Wan2.1-T2V-1.3B（原始公开权重）**。FFNDS 的 Q1r2 仅提供“沿输入增量分析非线性算子”的数学灵感；不继承 FFNDS 的数据、实验结论、压缩指标或优化目标。

## 阅读顺序

1. [研究认识与问题树](docs/01_research_questions.md)：三篇论文提供的真实证据、竞争机制和独立研究定位。
2. [算子定义与因果对象](docs/02_operator_formulation.md)：Wan 实际计算图，head 写入→FFN 的精确差分、路径干预、跨层与跨去噪时间。
3. [分阶段研究与实验规范](docs/03_experimental_plan.md)：数据、对照、候选筛选、干预、指标、验收和资源边界。
4. [原始文献与源码证据](docs/04_sources_and_evidence.md)：原论文/实现链接、可信度边界、待复核事项。
5. **[国际与中国企业技术报告调查（2026-11）](docs/05_industry_technical_reports_2026-11.md)**：跨 14 个前沿企业/团队的技术报告、工程博客、源码及物理评测，注明资料类型与证据边界。
6. **[从世界模型到运动算子：机制知识综合](docs/06_mechanistic_synthesis.md)**：renderer/simulator/planner、运动表征、case-based vs physical law、条件语义链、oracle future、参数反事实。
7. **[产业调研后的研究优先级与实验修订](docs/07_priorities_and_experiments.md)**：新增语义-动力学解耦、物理参数响应、真实/生成未来读出和跨条件因果测试。

## 当前状态（2026-11-19）

- **已完成**：研究命题、文献与源码核对、算子推导、实验方案文档；追加 2024–2026 国际及中国企业一手资料广覆盖调研、机制综合与实验路线修订。
- **未完成**：Wan 权重上的实际算子验收；head 单头消融复现；视频生成的因果干预；probe/子空间实验。
- 文档中的公式为数学定义或推导；实验矩阵中的预测与阈值是**待检验计划**，不是已测结果。

### 新调研的关键阅读提示

- 企业报告的披露层级不同：详实技术报告、开源源码、研究博客和产品公告分别处理；不能将演示视频直接解释为真实内部计算回路。
- 世界模型研究需区分**视频渲染、动作条件状态预测、规划控制**。FFNmotion 以解释为主，不预设 FFN 负责物理规律。
- [ByteDance PhyWorld](https://seed.bytedance.com/public_papers/how-far-is-video-generation-from-world-model-a-physical-law-perspective)、[Runway Praxis-1](https://runway.com/research/praxis-1-video-pretraining-to-robot-control)、[NVIDIA PhyProbe](https://research.nvidia.com/labs/cosmos-lab/phyprobe/) 与 [Kling 2026 研究报告](https://arxiv.org/abs/2611.04916) 是新阶段最直接的四类方法参照。

## 核心判据

- **表示**：某个注意力图/神经元/probe 能表达运动信息，不足以证明其决定生成结果。
- **算子**：局部 head→FFN 差分可精确分解，要求在真实 Wan 前向中检验数值恒等式。
- **因果**：对特定路径进行移除/恢复，并在冻结其他状态时比较下游输出；必要时继续完成整段视频。
- **泛化**：在未用于筛选路径的 prompt、seed、运动机制上进行独立验证。

> 项目从机制问题出发，允许结果支持 FFN 路径、自注意力路径、分布式方向或其他竞争解释；不预设“运动神经元”的存在。

## 主要外部依赖

- [Wan2.1 官方模型](https://github.com/Wan-Video/Wan2.1)
- [Why Do Video Diffusion Models Violate Physics?](https://arxiv.org/abs/2609.23658) / [作者实现](https://github.com/Siriuslala/physics)
- [DifFRACT](https://arxiv.org/abs/2606.15796) / [作者实现](https://github.com/Artalmaz31/DifFRACT)
- [Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050) / [ICML 2026 论文](https://proceedings.mlr.press/v306/joseph26a.html)

以上链接用于溯源；本项目与原论文作者没有隶属关系。