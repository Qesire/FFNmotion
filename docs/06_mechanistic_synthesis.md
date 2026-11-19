# 06 · 跨企业知识的机制综合：从生成图像到可干预运动计算

> 2026-11-19，研究综合，不是实验报告。原始来源清单在 [05](05_industry_technical_reports_2026-11.md)。**以下每个论点均明确区分：公开已测、合理推断、待测假说。**

## 1. 重新确定“世界模型”的研究对象

### 1.1 三种能力对应三个不同的函数

综合 [World Labs 的功能分类](https://worldlabs.ai/blog/taxonomy-of-world-models)、[Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/)、[Cosmos 3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/) 与 [Praxis-1](https://runway.com/research/praxis-1-video-pretraining-to-robot-control)，可以用输出语义区分：

- **Renderer**：给定描述/参考，生成观测 \(o\sim p_\theta(o\mid c)\)；评价可见一致性、指令遵循、视觉质量。
- **Simulator**：给定状态及行动预测状态转移 \(s_{t+1}\sim T_\theta(s_t,a_t)\)；评价几何、动力学、事件与 OOD 物理条件响应。也可能仅输出观察 \(o_{t+1}\) 而隐式实现转移，此时必须通过跨条件干预验证等价性。
- **Planner**：给定状态/目标，输出策略 \(a_t\sim\pi_\theta(a\mid s,g)\)，以环境反馈评价闭环任务完成。

这是**解释性分析分层**，不是将现有产品强行归入互斥类别。Cosmos 3、Runway Praxis、Genie 等系统可以同时包含多个功能。

### 1.2 本项目观测层级

设外部因素 \(\omega=(c_{\rm action},c_{\rm appearance},c_{\rm scene},c_{\rm camera},c_{\rm physics},\epsilon)\)：
- \(c_{\rm action}\)：文本或动作指令；
- \(c_{\rm appearance}\)：物体形状/颜色/材质；
- \(c_{\rm scene}\)：接触面、布局、背景；
- \(c_{\rm camera}\)：视角与相机运动；
- \(c_{\rm physics}\)：重力、摩擦、初速等可控的物理条件（对纯 T2V，它们可能仅在文本中描述，不一定是真正可用的独立控制接口）；
- \(\epsilon\)：随机噪声/采样器状态。

原视频生成模型输出 \(X=G_\theta(\omega)\)。我们研究：
1. 哪些因素在输入条件中被编码；
2. 信息在注意力、FFN、残差和跨去噪步骤之间如何改变；
3. 哪些内部干预改变下游视频运动量 \(m(X)\)；
4. 这种改变是否匹配对 \(c_{\rm physics}\) 或 \(c_{\rm action}\) 的真实反事实预期。

一个模型即使能对所有文字动作生成漂亮视频，仍可能主要实现 case-based matching，而不是可泛化的动力学规律。

## 2. “世界知识”可以存在于表征中，但动力学功能必须另外验证

[Meta V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) 证明视频预训练的 latent 可以支持 motion understanding、prediction 和 action-conditioned 规划任务；[Veo 3 的零样本研究](https://deepmind.google/research/publications/203190/) 表明生成模型可被用于诸多物理/视觉推断任务。

**已成立**：训练视频的骨干可以产生可供不同任务使用的丰富表征。  
**未随之成立**：所有可读物理变量都参与生成，或内部实际采用显式的牛顿状态更新。

因此应定义三类不同的效果：

\[
\mathrm{Decodability}(z) \quad(\text{从内部状态读出}),
\]

\[
\mathrm{LocalSensitivity}(z;u)=D_u Q(z) \quad(\text{局部扰动响应}),
\]

\[
\mathrm{BehaviorEffect}(u)=m(G(z+u))-m(G(z)) \quad(\text{完整生成反事实}).
\]

不能由其中任一类直接推出另外两类。对运动特征，最佳结构可能是**低内禀维物理状态 → 高维分布式神经编码 → 多层函数转换**，而不是固定 FFN 通道。

## 3. 物理规则泛化与案例匹配必须用因子干预分开

[ByteDance: How Far is Video Generation from World Model](https://seed.bytedance.com/public_papers/how-far-is-video-generation-from-world-model-a-physical-law-perspective) 的受控 2D 运动/碰撞实验，观察到训练分布内较强拟合、分布外泛化失败，以及颜色/尺寸等表面属性驱动相似样本匹配的倾向。这是视频生成模型在**指定合成设置中**的真实负证据，不是关于所有大模型的普遍定理。

与其只做“篮球向上/向下”的相关性，不如构造冲突实验：

| 改变因素 | 保持因素 | 如果是动作语义控制 | 如果是物理条件泛化 | 如果是案例匹配 |
|---|---|---|---|---|
| 改变动作词 | 对象/颜色/镜头不变 | 视频运动方向应跟随动作词 | 还应满足动力学事件 | 可能受训练动作习惯影响 |
| 改变颜色/材质 | 同样运动指令、初态、场景 | 运动读出应尽可能不变 | 只在材料真实影响摩擦等时改变 | 可能因表观相似性而明显改变 |
| 改变初速/重力参数 | 物体身份/视角不变 | 文本可能表达该参数但不一定可执行 | 预测加速度、落地时刻等定量变化 | 可能不遵守数值比例 |
| 跨碰撞阈值改变输入 | 非关键外观不变 | 应满足具体任务动作 | 接触与非接触两个结局应发生合理转换 | 可能只生成单一常见结局 |

这可直接连接 [PhysResponse](https://arxiv.org/abs/2611.24419) 的参数响应与转变事件评测。后者为 2026-11-18 新预印本，其结果应记为作者报告，待第三方复核。对 Wan2.1 纯 T2V，首轮可以测试定性 action/初速度条件，只有验证模型确实接收并响应连续数值条件后，才继续研究定量参数的单调性/比例关系。

## 4. 条件语义与生成计算共同决定运动，不能把所有差异归给单个 Head

[Seedance 1.0](https://seed.bytedance.com/en/public_papers/seedance-1-0-exploring-the-boundaries-of-video-generation-models) 将精细 caption、数据覆盖、位置嵌入、SFT 与视频 RLHF 一起作为 motion/prompt following 的核心工程手段；[Kling 2026-11 的 understanding-stack 技术报告](https://arxiv.org/abs/2611.04916) 进一步将数据、reference-grounded caption、prompt enhancement、perceptual text encoder、reward model 和 evaluation 视为整条语义契约链。

**因此，模型出现某种运动失败时，至少有三种竞争来源：**

1. 输入条件没有精确区分目标动作/物理概念；
2. cross-attention 等模块把语义编码到了残差流，但后续模型没有形成相应动作；
3. 运动被生成了，但物体跟踪、相机运动、遮挡/接触规则或解码影响最终观测。

设计对应的测试：冻结生成权重，比较文本近义改写、明确反义动作、物体/背景交换；在同一步检查 head write 是否可被区分，继而检查 FFN 和最终视频变化。**条件编码差异**与**模型使用已编码信息的效果**需要两个单独的 estimand。

## 5. 原生算子响应是连接内部表征与行为的第一层中介

Wan2.1 的单 block 模型与精确 head→FFN 形式见 [02](02_operator_formulation.md)：

\[
B(c^+)-B(c^-)=U_h+\sum_j D_{hj},
\]

其中 \(U_h\) 是输出投影后的单 head 写入，\(D_{hj}\) 是 head 引起的第 \(j\) 个 FFN 隐藏通道**增量写入**。

这是一个**局部模块恒等式**，不是物理运动机制的证明。

结合企业报告，可以把一个 head 的影响分成：

- **运动条件来源**：文本/多模态条件输入是否含运动信息（Kling/Seedance 的语义链）；
- **局部转换**：LayerNorm、AdaLN、W1、GELU、W2 处理该 head 的真实有限扰动；
- **跨层传播**：在后续 self-attention、跨帧 Value/readout 等模块中再混合；
- **跨步骤传播**：通过 sampler 和历史状态传至最终视频（Genie/WorldPlay/Cosmos/Praxis 的长时模型强调的这一维）；
- **行为读出**：动作跟随、轨迹事件、状态变迁、物理一致性指标。

其中 A/B 层的响应矩阵可精确计算，但行为层的独立因果验证不可省略。

## 6. Runway Praxis-1 提供了一个更强的“预测/读出”错误分解

[Praxis-1 技术报告](https://runway.com/research/praxis-1-video-pretraining-to-robot-control) 2026-11-11 公开了可复用的方法：

- 视频骨干先预测未来的潜在视觉状态；
- 独立 action expert 跨层读取骨干的未来/上下文特征，预测与视频对应的机器人动作；
- 在推断中比较 **model-imagined future** 与 **oracle ground-truth future** 两个条件，区分“想象错误未来”与“从正确未来读出错误动作”。

作者报告当提供真实未来时，末端运动误差约缩小到原来的五分之一，说明在该检查点/任务中视频预测仍是主要误差来源。该证据是 *Praxis-1 世界动作模型* 的任务结果，不应直接移植为 Wan2.1 的效应量。

### 对 FFNmotion 的新工具性启发

可以训练一个轻量 **motion readout**，输入 Wan 某层状态，输出方向/速度/物体事件；同时比较：
- 对照 A：读取 baseline 自然生成 latent；
- 对照 B：读取匹配的真实/合成视频经正确噪声日程得到的 latent；
- 对照 C：在 baseline 前向中注入目标 head/FFN 路径，观察 motion readout 与最终视频的变化。

由此分离：
**状态是否编码了正确运动**、**读出器是否能够可靠读取**、**生成后半段是否将这条信息转化为运动**。

必须谨慎：真实视频 latent 是不同条件分布，B 不是 A 的自然可得 oracle；读出器应在独立样本上验证，不要直接把 oracle 性能提升当作内生机制的证明。

## 7. 物理一致性应当分层、参数化、对照评测

### 7.1 四个互补估计目标

- **Prompt alignment / motion instruction**：目标动作是否出现、有向位移是否正确。
- **Kinematics**：轨迹连续性、速度、加速度、时序接触；检查失踪/形变/镜头掩盖。
- **Physical violation severity**：连续违规分数，可参考 [PhyProbe](https://research.nvidia.com/labs/cosmos-lab/phyprobe/)；它是学习得来的代理，有 domain shift，不能独自决定真假。
- **Counterfactual parameter responsiveness**：改变 \(g,\mu,v_0\) 或动作控制后，生成的动力学指标是否发生预期变化，可参考 [PhysResponse](https://arxiv.org/abs/2611.24419) 与 [ByteDance PhyWorld](https://seed.bytedance.com/public_papers/how-far-is-video-generation-from-world-model-a-physical-law-perspective)。

已有 [VideoPhy-2](https://arxiv.org/abs/2503.06800)、[VBench-2.0](https://arxiv.org/abs/2503.21755)、[NVIDIA PBench](https://research.nvidia.com/labs/dir/pbench/) 和 [PhyWorldBench](https://research.nvidia.com/labs/cosmos-lab/phyworldbench/) 对 motion、commonsense、physics 的覆盖不同，跨基准模型得分不可直接放在一个大小比较表中。

### 7.2 最值得做的“物理反事实”

对一类明确的物理量 \(q\)，设 \(\widehat m(q)\) 为生成视频中由追踪或物理参数拟合器得到的估计：

\[
R(q_1,q_2)=
\widehat m\!\left(G(c,q_2,\epsilon)\right)
-
\widehat m\!\left(G(c,q_1,\epsilon)\right).
\]

还应记录机制目标预测的响应方向 \(R^*(q_1,q_2)\)，以及 nuisance factor（颜色、相机、外观）保持不变时的重复性。对纯文本条件，\(q\) 通常只是文字描述的控制代理，而不是保证可执行的数值输入；不应将无法受控的任务当作物理规律的严格检验。

对撞击/弹跳等离散事件，使用事件触发阈值的两侧输入，更容易区分连续相似生成与正确的动力学状态转换。

## 8. 跨帧记忆和跨去噪步属于不同的时间轴

在 [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) 中，Reconstituted Context Memory 与 Context Forcing 解决的是**自回归流式模型的长时上下文**；[Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) 强调重访场景需要读取久远历史；[Cosmos Simulation](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/) 明确以流式 chunk 和物理状态控制交互世界。

Wan2.1 则在每个去噪调用中利用整个 spatiotemporal latent 的双向 self-attention；其“不同视频帧 token”与“不同采样迭代”的信息流不同于 action-conditioned autoregressive world model。

所以本项目追踪三种变化必须分开：
- 网络深度 \(\ell\)；
- 同一次生成的视频时空 token \((f,p_x,p_y)\)；
- 去噪积分步 \(s\) / 自回归 chunk \(k\)（不同模型切勿混用）。

尤其在同一迭代内的 FFN 只做 token-wise 非线性作用，后续跨 token/跨帧传播需由 self-attention 等算子实现。**“上一去噪步同层功能复用”与“过去视频帧的记忆”不是同一个因果关系**。

## 9. 成立的六项认识与新的可区分预测

| 可保留认识 | 直接来源 | 新的可区分预测 |
|---|---|---|
| 外观逼真与物理守恒可分离 | OpenAI Sora 自述局限；NVIDIA PhyProbe、PhyWorldBench | 仅依赖美学/平滑度排序无法稳定预测物理事件正确性 |
| 可控动作不是简单文本匹配 | Genie、Wayve GAIA、Praxis、WorldPlay | 在控制显式 action 时，关键 motion readout/path 对反事实动作变化应更稳定 |
| 生成知识可以迁移到动作预测 | Meta V-JEPA2、Runway Praxis | 在生成骨干某些层存在可用 motion direction/readout，但不保证与 FFN 映射一一对应 |
| 物理外推是模型的薄弱点 | ByteDance PhyWorld、PhysResponse | 原始神经元的“运动响应”可能绑定训练中颜色/对象等表象，而非动力学变量 |
| 输入语义链影响模型行为 | Kling understanding stack、Seedance | 同义改写/精细 caption 控制后，可减少把语义偏差误认成 FFN 动力学失败 |
| 运动路径可能跨空间与时间传递 | Wan self-attention、Genie/WorldPlay/Cosmos 的流式设计 | 局部 FFN 通道响应未必等于最终运动作用，需要路径恢复和跨时间因果试验 |

表中第三列是**待验证研究预测**，不能因为前两列有来源就当作已经成立。

## 10. 建议的研究对象重定义

以前的中心疑问：**哪些 FFN 通道连接某个 motion cross-attention head？**

新的更广义问题：

\[
\boxed{\text{给定可控运动干预，哪些内部计算路径使生成视频产生符合目标的运动响应？}}
\]

它允许在不同模型、任务和证据条件下选择：
- head/通道级（可直接跑 Wan2.1）；
- residual subspace 级（如果通道解释分布式）；
- motion state/readout 级（如果内部状态可读但自然路径未知）；
- streaming action/state 级（面向 Genie/Praxis/WorldPlay 外推）。

研究目标仍是**解释**：不是事先决定模型必须具有单一运动回路或一个可压缩的状态表征。

下一步：[07 · 更新后的研究路线](07_priorities_and_experiments.md)。
