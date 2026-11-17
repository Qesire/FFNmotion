# 04 · 文献溯源、有效认识与源码证据

> 更新：2026-11-17。此页记录**已核查的原始来源及其能够支持的范围**，避免从论文标题、可视化和作者的机制叙述直接推导超出证据的结论。文献页、代码 main 分支未来可能更新；真实复现实验需冻结源码 commit 与 checkpoint hash。

## 1. 直接相关的三篇论文

### 1.1 Yueyan Li et al. · Why Do Video Diffusion Models Violate Physics? Unveiling the Flaws in Attention Mechanisms

- [arXiv:2609.23658](https://arxiv.org/abs/2609.23658)；2026-09-20 提交，已核对版本为 v1。
- [作者项目页](https://siriuslala.github.io/physics/)。
- [作者 GitHub](https://github.com/Siriuslala/physics)。
- [head 消融具体实现](https://github.com/Siriuslala/physics/blob/e20e09832b3b1321266810ac0fbbc597803e3e11/wan21_t2v_experiments/cross_attn_head_ablation.py)。
- [作者实验脚本及预设 head 分组](https://github.com/Siriuslala/physics/blob/e20e09832b3b1321266810ac0fbbc597803e3e11/scripts/cross_attn_head_ablation_wan2_1.sh)。

**可直接借鉴的证据**：
- 对 Wan2.1-T2V-1.3B 进行 cross-attention 时空轨迹可视化与 head/头组消融；
- 轨迹显著性与 head 输出干预的最终效果不完全一致；
- 早期空间注意力修改能够在报告的设置下改善部分 VideoPhy 结果。

**解释边界**：
- 注意力轨迹只是关联图；实际 head write 还取决于 Values 与输出投影；
- 头组消融证明的是组合作用，不自动给出单头贡献；
- 时间调制 RoPE 具有干预效果，不唯一证明 RoPE 是所有物理错误的根因；单频 RoPE logit 可含余弦振荡而非严格单调的距离衰减；
- “训练 FFN LoRA 无额外益处”与“原 FFN 不承载运动功能”不是相同命题；
- 原 head 消融函数在合并每个 head 的输出前置零，然后执行 `cross_attn.o`，但补丁没有默认显式处理 cond/uncond 分支，需要补实验。

**对本项目的用途**：源候选 head、作者设置复核、行为实验起点。先复核真实单头效应，再展开 FFN 路径分析。不要直接给 L15H2/L9H7 贴因果强弱标签：公开脚本里的分组与早期讨论所用的标签不一致。

### 1.2 Artyom Mazur, Nina Konovalova, Aibek Alanov · DifFRACT: Diffusion Feature Reconstruction and Attribution for Circuit Tracing

- [arXiv:2606.15796](https://arxiv.org/abs/2606.15796)；v1 2026-06-14，v2 2026-10-26；NeurIPS 2026。
- [官方代码仓库](https://github.com/Artalmaz31/DifFRACT)；[项目主页](https://artalmaz31.github.io/DifFRACT/)。
- [README 与训练/干预说明](https://github.com/Artalmaz31/DifFRACT/blob/main/README.md)。

**可直接借鉴的证据**：
- 在 FLUX.1[schnell] 等图像 MM-DiT 中，时间条件 transcoder 逼近 MLP 的输入→输出，并将稀疏特征形成局部归因回路；
- graph-guided 的特征选择、抑制路径与跨模态语义作用有作者在原模型中的干预证据；
- v2 额外报告盲标注和其他图像生成 backbone 的验证。

**解释边界**：
- “精确的 feature-to-feature attribution”相对于作者冻结注意力等操作后构造的 Local Replacement Model；受 transcoder 重建与缓存残差的适用范围限制；
- 输入在真实模型中发生有限反事实变化时，softmax 注意力和 LayerNorm 可改变，局部归因不应自动推广到整个模型；
- FFN 特征字典不是原始模型唯一的、天然存在的语义坐标系；
- 这些验证是图像生成而非 Wan T2V 运动生成。

**对本项目的用途**：提供“计算回路而非激活热力图”的方法思路；原始 FFN 的精确 finite-difference 可以先于 transcoder。只有后续证据表明单通道/输出方向不足以解释跨样本的运动计算时，才引入跨时间稀疏特征替代模型。

### 1.3 Sonia Joseph et al. · Interpreting Physics in Video World Models

- [arXiv:2602.07050](https://arxiv.org/abs/2602.07050)。
- [ICML 2026 / PMLR 306, pp. 54687–54724](https://proceedings.mlr.press/v306/joseph26a.html)。
- [Meta 研究出版页](https://ai.meta.com/research/publications/interpreting-physics-in-video-world-models/)。

**可直接借鉴的证据**：
- 在 V-JEPA 2、VideoMAE-v2 等视频**编码器**中，对速度、方向、物理合理性进行层级 probe；
- 运动方向的可解码性具有中层增强结构，作者称为 Physics Emergence Zone；速度大小等量的层级性质有所不同；
- 局部时空注意力连接的干预会影响运动相关 probe/物理基准；
- 方向包含分布式群体编码性质，联合多方向 activation steering 能改变独立 probe 的内部方向读出。

**解释边界**：
- 成功 probe 说明信息可被读取，而非模型显式使用牛顿物理方程；
- 多个正交 probe 方向不等于高内禀维度，更不排除简洁的非线性状态流形；
- 内部 steering 改变 probe 方向，不等于生成模型最终物体轨迹改变；
- 视频编码器结论不可不经检验迁移到去噪生成式 Wan。

**对本项目的用途**：指导速度/方向 probe、运动与外观解耦、协同方向及子空间/流形的对照设计。

## 2. 模型和运行代码证据

### 2.1 Wan2.1 官方模型

- [T2V 1.3B 配置](https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/wan_t2v_1_3B.py)：`dim=1536, ffn_dim=8960, num_heads=12, num_layers=30, patch_size=(1,2,2), qk_norm=True, cross_attn_norm=True`。
- [Wan 模型实现](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py)：
  - `WanAttentionBlock.forward`：self-attention 残差 → cross-attention 残差 → norm2/AdaLN → FFN → 门控残差；
  - `WanT2VCrossAttention.forward`：Q/K/V、flash_attention、flatten heads、`self.o`；
  - `ffn = nn.Sequential(nn.Linear(dim,ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim,dim))`。
- 注意源码 `norm3` 位于 cross-attention 之前，`norm2` 位于 FFN 之前；其他论文附录的归一化编号可能不同。

这些是源码结构，不证明任何一个通道已经被实验识别为运动通道。必须读取真实模型权重的 forward 数据才能做算子验收。

### 2.2 原论文实验复用

- 作者脚本固定的第一组 T2V 1.3B 配置：frame_num=81、size=832×480、sampling_steps=50、shift=5.0、guidance_scale=5.0。
- 代码中按 `LxHy`（0-based）指定 head，既支持全部去噪步也支持特定步。
- 基础补丁对每个选定 head 的 flash-attention 输出置零；其默认对象是所选整个模型的交叉注意力调用，而非分别筛选 CFG 分支。
- 复现中不可把作者注释性的头组分级看作经过本项目复核的单头行为标签。

建议启动前冻结第三方仓库当前 commit，并记录任何运行时 monkey patch 的差异。只要 branch 逻辑未经过具体代码绑定，记录为 OPEN_BINDING。

## 3. 数学主张与实验证据的对应关系

| 对象 | 属于什么性质 | 本项目如何核对 |
|---|---|---|
| \(B^+-B^-=U_h+\sum_jD_{hj}\) | 给定 Wan 模块结构下的精确有限差分恒等式 | 运行真实权重并检查数值误差 |
| \(\Delta F=W_2\operatorname{Diag}(\bar\beta)W_1\Delta u\) | 给定输入对与可微 GELU 下的积分恒等式 | 与 FFN 直接差分比较 |
| LayerNorm 扰动可能被削弱 | 来自 Jacobian 的局部几何推论 | 测量实际 \(J_{\rm LN}U_h\) 与有限差分 |
| Head 显示轨迹但因果影响弱 | 原论文所报告的特定观察 | 重新进行单头/头组与 CFG 对照 |
| FFN 中存在关键运动转换路径 | **尚未证实的假说** | 直接/FFN 路径恢复，最终视频对照 |
| 原始单通道是稳定的“运动神经元” | **尚未证实的假说** | 跨样本 Top-K 恢复与独立验证 |
| 运动具有低内禀维流形 | **候选解释** | 比较线性张成/局部切空间/干预响应 |
| 物理合理性可由内部功能方向操控 | **研究目标** | 生成视频中的独立物理事件与轨迹验证 |

## 4. 用于下阶段查重的具体检索范围

研究问题尚未经过系统性最新文献查重，不应宣称“首次视频运动回路追踪”。下一阶段至少覆盖：

- diffusion/video transformer 的 activation patching、path patching、causal tracing；
- generating motion 的 steering vectors、motion editing、direction/velocity probes；
- sparse autoencoder 与 transcoder 的 diffusion/DiT 应用和 faithful intervention；
- video world model 中物理变量、群体编码、内禀维度、时空注意力的可解释性；
- flow-matching/CFG 对条件路径以及跨采样步的因果敏感性。

查重时比较实际算子、干预位置、可用信息时点、是否进行完整生成行为验证，避免仅按题目相近与否判断覆盖范围。

## 5. 资料维护约定

每个未来实验结论应有：source/version、model/checkpoint、prompt/seed、CFG branch、time/layer/token 坐标、intervention、raw metric、paired control、视频或日志。按以下证据等级记录：

**DERIVED（公式推导） / SOURCE_VERIFIED（源码或论文原始记录） / MODULE_TESTED（算子数值通过） / GENERATION_TESTED（实际视频效果） / REPLICATED（独立复现） / HYPOTHESIS（待测）**。

一个实验通过“模块测试”不能写成“真实生成因果验证通过”；论文作者报告的结果应注明“reported by authors”，不当作本项目复现实测。

本项目文档入口：[README](../README.md) → [01 研究问题](01_research_questions.md) → [02 算子](02_operator_formulation.md) → [03 实验](03_experimental_plan.md)。
