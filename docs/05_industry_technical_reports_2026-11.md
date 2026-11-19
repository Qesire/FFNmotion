# 05 · 国际与中国前沿企业技术报告调查（2026-11-19）

> 主题：视频运动生成、世界模型、物理规律、时空表征、跨模态条件、可控性、内部运动读出、因果干预和评测。
> 状态：**一手资料检索与机制级筛查完成；不代表原始模型已独立复现。** 本库区分工程技术报告、研究论文、官方技术博客、产品公告与独立评测，禁止用营销演示证明模型内部机理。

## 0. 调查范围、证据等级与阅读方法

- 截止时间：2026-11-19；优先覆盖 2024–2026 的模型家族与 2026 年新技术发布。
- 主体：国际 **OpenAI、Google DeepMind、Meta、NVIDIA、Runway、Wayve、World Labs**；中国 **字节跳动 Seed、阿里巴巴 Wan、腾讯 Hunyuan、快手 Kling、智谱 CogVideoX、生数 Vidu、MiniMax Hailuo**。
- 主要检索入口：各公司官网研究栏目、官方技术报告/论文摘要、项目源码和模型卡，必要时使用第三方独立基准确定评价性质。
- 证据分级：**A** 可核对技术报告/原论文（含模型/实验方法）；**B** 官方研究博客/工程说明；**C** 产品发布与演示/公司自评；**E** 第三方评测。A/B/C/E 不是性能评分。是否公开权重/训练代码另外记录。
- 研究任务不是给视频模型排名，而是获得对 *表示、计算、可干预性、跨步动态、物理状态与泛化* 的有效认识。
- 由于非开放模型缺乏中间激活/源代码，**产品演示无法回答 FFN 通道的作用机制**。应把它们当作机制假说与行为验收条件来源。

## 1. 国际团队：原始技术来源

| 主体 / 时间 | 公开资料、链接 | 等级/公开性 | 真正提供的技术知识与 FFNmotion 对应点 |
|---|---|---|---|
| OpenAI · 2024 | [Video generation models as world simulators / Sora](https://openai.com/index/video-generation-models-as-world-simulators/) | B，技术研究说明，未公开模型细节和权重 | 时空 latent patches、diffusion transformer、可变时长/分辨率联合训练；展示三维一致性、遮挡后物体持续性与部分交互，同时承认物理错误。*可见能力不等于显式世界状态* |
| OpenAI · 2025 | [Sora 2 system card](https://openai.com/index/sora-2-system-card/) | B/C，系统卡 | 报告更强物理表现和控制能力，主要为产品行为与安全资料，不能从中推出内部动力学算子 |
| Google DeepMind · 2024 | [Genie: Generative Interactive Environments](https://deepmind.google/research/publications/60474/) | A，研究报告 | 无真实 action 标签的视频中学习 latent action，与 video tokenizer、autoregressive dynamics 联合；**动作条件**是不同于静态文本条件的对象 |
| Google DeepMind · 2024 | [Genie 2 world model](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) | B，架构说明 | 自回归 latent diffusion + causal mask + 逐帧动作条件 + CFG，真实可交互模型与一次性整段生成不同；内存与长时闭环是因果传播问题 |
| Google DeepMind · 2025 | [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) | B，研究博客 | 720p / 24fps、分钟级交互一致性的作者报告；承认动作空间、多个独立 agent、长时持续等限制。**长时运动一致性**不能只看单段视频 |
| Google DeepMind · 2025 | [Video models are zero-shot learners and reasoners / Veo 3](https://deepmind.google/research/publications/203190/) | A，研究报告 | 生成模型可通过指定任务的输入输出设置表现出物理属性、affordance、工具使用等零样本理解，提供“生成骨干有可读功能”方向，但不是内部回路定位 |
| Google DeepMind · 2025–2026 | [Veo 模型与控制功能](https://deepmind.google/models/veo/) | C，产品/模型页 | 相机运动、物体路径、首尾帧、参考素材等条件接口构成**可控行为类型库**；无法核验内部注意力/FFN |
| Meta · 2025 | [V-JEPA 2 technical report](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) | A，研究报告、公开模型 | 从互联网视频无动作训练 + 少量机器人数据 post-train action-conditioned latent world model；证明部分物理功能可在非渲染 latent 表征上读出，并能用于规划 |
| Meta · 2025 | [V-JEPA 2 physical reasoning benchmarks](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) | B，官方基准博客 | understanding/prediction/planning 三种任务应分离；可作为运动 probe 的外部表示对照 |
| NVIDIA · 2025 | [Cosmos World Foundation Model Platform](https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai) | A，技术报告、开源平台 | video tokenizer、world model、定制后训练、视频数据清洗，对视觉生成与物理 AI 分开设计任务 |
| NVIDIA · 2025 | [Cosmos-Predict2](https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2/) | B，研究页，开源模型 | image/video-to-world 任务化预测，将生成视频置于物理 AI 环境而非纯影视评价 |
| NVIDIA · 2025 | [Cosmos-Reason1](https://research.nvidia.com/publication/2025-03_cosmos-reason-1-physical-ai-common-sense-embodied-decisions) | A，技术报告，代码权重公开 | 物理常识层级 ontology、embodied reasoning SFT/RL；**会对物理任务作答**与**视频动力学服从规律**分开 |
| NVIDIA · 2025 | [PBench](https://research.nvidia.com/labs/dir/pbench/) | B，公开基准与模型结果 | AV、robot、industry、physics、human、commonsense 分域的 image+text→video；1044 样本、5636 个 QA；将 domain score 与视频质量分开 |
| NVIDIA · 2026 | [Cosmos 3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/) | A，公开技术报告/项目 | language/image/video/audio/action 的理解+生成统一，Mixture-of-Transformers/自回归扩散；功能模型可直接输出 forward/inverse dynamics，但**任务输出共享**不等于已知单一物理子空间 |
| NVIDIA · 2026-11 | [Interactive World Simulation with Cosmos](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/) | A，技术报告、代码/模型 | 状态化流式生成与新动作条件，包含相机/机器人/多视角驾驶；明确区分模型预测动作后果与外部 simulator 维护显式状态两种机制 |
| NVIDIA · 2026 | [PhyProbe](https://research.nvidia.com/labs/cosmos-lab/phyprobe/) | A，NeurIPS 2026 | 冻结时空表征+小评分头，通过 pairwise ranking/scalar/anchor 获取连续物理违规分数，验证“观感很好但物理错误”的情况 |
| NVIDIA · 2026 | [PhyWorldBench](https://research.nvidia.com/labs/cosmos-lab/phyworldbench/) | A，ICLR 2026 Oral | 从基础动力学到复杂交互，包含 anti-physics 条件；不应把违背常识的文本执行能力与物理正确性混合成一个总分 |
| Wayve · 2025 | [GAIA-2 research brief](https://wayve.ai/thinking/gaia-2/) + [technical report](https://wayve.ai/wp-content/uploads/2025/03/GAIA_2_Technical_Report.pdf) | A/B，报告与官方总结 | 通过自车动力学、其他 agent 配置、天气和道路语义进行可控、多视角潜在扩散世界生成；**结构化条件与观察相机**须作为独立变量 |
| Wayve · 2025–2026 | [GAIA-3](https://wayve.ai/thinking/gaia-3/) | B，研究说明 | 更大数据/模型和跨国场景，以安全评估/长轨迹控制为目标；能力数字以官方报告为准，未核验中间算子 |
| Runway · 2025 | [GWM-1](https://runway.com/research/introducing-runway-gwm-1) | B，研究说明 | 基于视频生成基础模型的交互式自回归世界模型，分 Worlds/Avatars/Robotics；**控制动作和闭环**成为接口 |
| Runway · 2026 | [GWM Worlds 2](https://runway.com/research/introducing-gwm-worlds-2) | B，研究预览 | 文本 action、相机连续控制、生成音画；属于可观测 action-following，无法从 demo 确证内部动力学 |
| Runway · 2026-11-11 | [Praxis-1 technical report](https://runway.com/research/praxis-1-video-pretraining-to-robot-control) | A，技术报告、开源权重 | 27B 级视频骨干，action expert 跨层读取视频模型的隐藏特征并与视频预测联合训练。**oracle 真未来视频 vs 模型想象未来**可区分未来预测错误与 action readout 错误；这是非常接近本项目的验证方法 |
| World Labs · 2025 | [Marble](https://worldlabs.ai/blog/marble-world-model) | B，技术/产品说明 | 视频/图像/文本/粗 3D 进入可探索场景，Gaussian splats、mesh 等输出；显式空间结构是另一种对象，不能与视频 latent 概念混淆 |
| World Labs · 2026-06 | [A Functional Taxonomy of World Models](https://worldlabs.ai/blog/taxonomy-of-world-models) | B，方法论/概念文章 | 把 renderer（观测）、simulator（状态）、planner（动作）按输出对象分开，提出统一闭环；**这是概念框架，不是已由干预证明的定理** |
| World Labs · 2026 | [Atlas](https://worldlabs.ai/blog/atlas) | B，研究博客 | 文本/图像/视频/3D 的自回归扩散空间上下文与显式点云、depth、3DGS，强调视角变化中的几何一致性 |
| World Labs · 2026-11 | [Frameboy](https://worldlabs.ai/blog/atlas-realtime) | B，研究预览 | Atlas 高效实时视角/相机控制，图像生成帧与结构化空间世界的一致性关系 |

**国际部分核心区别**：Sora/Veo 主要是高质量 *observational renderers*；Genie/Runway/Cosmos 把 *actions* 纳入环境转移；Wayve 还要求对多智能体与多视角交通状态进行结构条件控制；Meta/Runway Praxis 证明视频表示可用于独立的规划/动作读出任务。以上是已公开任务能力与系统架构事实，**不是各自的 FFN 已承载显式动力学方程的证据**。

## 2. 中国团队：模型技术报告与工程披露

| 主体 / 时间 | 公开资料、链接 | 等级/公开性 | 技术内容与机制相关性 |
|---|---|---|---|
| 阿里巴巴 · 2025 | [Wan 论文/技术报告](https://arxiv.org/abs/2503.20314) + [Wan2.1 官方源码](https://github.com/Wan-Video/Wan2.1) | A，报告/权重/推理代码 | Flow Matching 视频 DiT、T5 cross-attention、3D causal VAE、时间调制；**FFNmotion 的第一性原理算子绑定对象** |
| 阿里巴巴 · 2025 | [Wan2.2 官方实现与说明](https://github.com/Wan-Video/Wan2.2) | A/B，源码/模型 | 跨去噪时间划分高/低噪声专家的生成骨干与运动数据扩充；在本研究只是**跨权重/时间专家外推样本**，不可把结构优化目标混入研究 |
| 阿里巴巴 · 2025 | [Wan2.6 发布](https://www.alibabacloud.com/en/press-room/alibaba-unveils-wan2-6-series-enabling-everyone) | C，产品官方说明 | 参考视频、人物/声音持续性、多镜头叙事和更长时段，作为**长时身份/动作一致性**行为问题来源；不可推出内部 head 设计 |
| 字节跳动 Seed · 2025 | [Seedance 1.0 技术报告](https://seed.bytedance.com/en/public_papers/seedance-1-0-exploring-the-boundaries-of-video-generation-models) | A，团队完整技术报告 | 复杂动作来自多源训练数据、精细视频字幕、跨模态位置设计、视频 SFT/RLHF 与多阶段蒸馏；**不能把动作生成改善全归因到 attention 几何** |
| 字节跳动 Seed · 2025 | [Seedance 1.5 pro 技术报告](https://seed.bytedance.com/en/public_papers/seedance-1-5-pro-a-native-audio-visual-joint-generation-foundation-model) | A，研究报告 | 双分支音视频 DiT、联合模块、音画同步、动作/声画语义一致性；提供跨模态动作事件同步的观测条件 |
| 字节跳动 Seed · 2026-04 | [Seedance 2.0: Advancing Video Generation for World Complexity](https://arxiv.org/abs/2604.14148) | A，正式技术报告 | 统一 text/image/audio/video 参考条件、复杂交互、跨镜头、动作遵循；公开报告偏模型/评价与任务，不能当作完整可复现实验源码 |
| 字节跳动 Seed · 2026 | [Seedance 2.5 产品线](https://seed.bytedance.com/en/models) | C，官方模型页 | 30 秒叙事及参考控制的产品能力信息，内部机制和定量物理因果仍需原始报告验证 |
| 字节跳动研究 · 2024/ICML 2025 | [How Far is Video Generation from World Model: A Physical Law Perspective](https://seed.bytedance.com/public_papers/how-far-is-video-generation-from-world-model-a-physical-law-perspective) | A，原始研究/仿真实验 | **最重要的反例**：受控 2D 运动/碰撞模型 ID 较好、OOD 失效，趋向 case-based generalization；颜色等表象特征可能比物理速度更强地控制匹配 |
| 腾讯 · 2024 | [HunyuanVideo technical report](https://arxiv.org/abs/2412.03603) + [源代码](https://github.com/Tencent-Hunyuan/HunyuanVideo) | A，报告/代码权重 | 大型视频 DiT、视频数据与统一生成方案；适合作跨架构可复用对照，不借鉴无验证的营销质量结论 |
| 腾讯 · 2025 | [HunyuanVideo 1.5 technical report](https://arxiv.org/abs/2511.18870) + [源码](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | A，报告/代码权重 | Selective & Sliding Tile Attention (SSTA)、时空 KV block 选择、progressive training、超分；提示*跨帧消息传播可能受注意力稀疏结构影响* |
| 腾讯 · 2025–2026 | [HY-WorldPlay / HY-World 1.5](https://github.com/Tencent-Hunyuan/HY-WorldPlay) | A，技术报告/公开代码权重 | 实时可控视频世界模型：Dual Action Representation、Reconstituted Context Memory、WorldCompass RL、Context Forcing 蒸馏；重点是**长程记忆、结构和动作条件**，区别于单步 Head→FFN |
| 快手 · 2024 | [Kling 官方架构说明](https://ir.kuaishou.com/news-releases/news-release-details/kuaishou-unveils-proprietary-video-generation-model-kling/) | B/C，官方披露 | diffusion Transformer、3D VAE、跨时空 full attention；不包含足以验证单头功能的模型实现 |
| 快手 · 2026-11 | [Behind Video Generation: Observations and Insights on Multimodal Understanding from Kling](https://arxiv.org/abs/2611.04916) | A，技术报告 | **很重要**：将质量沿数据→caption→prompt enhancement→text encoder→reward→evaluation 全链追溯；视频失败未必起因于生成 backbone 的某个算子 |
| 快手 · 2026-10 | [Kling 4.0 官方公告](https://ir.kuaishou.com/news-releases/news-release-details/kling-ai-officially-launches-next-generation-kling-40-raising) | C，产品发布 | 多参考、多关键帧、相机运动、长视频运动控制；可提炼对照任务，不将其设为机制真实性证据 |
| 智谱/清华 · 2024 | [CogVideoX: Expert Transformer](https://arxiv.org/abs/2408.06072) + [源码](https://github.com/zai-org/CogVideo) | A，报告/模型公开 | 3D VAE、video text expert AdaLN、渐进时空训练；架构对照：**文本注入并非只能使用 Wan 的独立 cross-attention** |
| 生数/清华 · 2024 | [Vidu technical report](https://arxiv.org/abs/2405.04233) | A，技术报告 | U-ViT 视频模型、时间一致性、可控生成，显示另一种 backbone 设计，适合作行为级而非直接 FFN 索引级比较 |
| 生数 · 2026 | [Vidu 开放平台模型说明](https://platform.vidu.com/docs/overview/introduction) | C，API/产品技术说明 | multi-subject reference、首尾帧等可控动作条件；当前不具备公开内部因果 tracing 的直接资料 |
| MiniMax · 2025 | [Hailuo 02 官方发布](https://www.minimax.io/news/minimax-hailuo-02) | C，官方模型演示 | 复杂体操/动态/物理表现的官方主张，可选择为困难动作 benchmark，不提供可验证 FFN 算子 |
| MiniMax · 2025 | [Hailuo 2.3 官方说明](https://www.minimax.io/news/minimax-hailuo-23) | C，官方发布 | 复杂人体动作与微表情控制的报告，技术披露不足以确定运动机制 |

中国团队资料中的**主要新知识**：动作质量可由数据语义密度、文本编码器、奖励监督、位置与时空注意力、条件参考、跨帧上下文和蒸馏等多层系统共同影响。FFNmotion 应当把“文本条件导致真实 motion 差异”分解到数据/条件语义与生成内部路径两个层次，而不是仅根据 Wan 上某个 head 的异常解释全部物理问题。

## 3. 与企业工作直接相连的独立评测/理论资料

| 来源 | 重要问题 | 与本项目的用途 |
|---|---|---|
| [VideoPhy-2](https://arxiv.org/abs/2503.06800) | 动作语义遵循 + 物理常识的联合评测；保留失败原因 | 终端行为分离；避免“没动就没有物理错误” |
| [VBench-2.0](https://arxiv.org/abs/2503.21755) | human/creativity/controllability/physics/commonsense 五维 intrinsic faithfulness | 不用传统美学与运动平滑度替代物理 |
| [PBench](https://research.nvidia.com/labs/dir/pbench/) | 对 AV/机器人/物理任务的条件视频生成和 QA | 多域视频任务用于测试外部有效性，非单一通道标签 |
| [PhyWorldBench](https://research.nvidia.com/labs/cosmos-lab/phyworldbench/) | 物理规律与 anti-physics 条件 | 验证“遵循反物理指令”和“默认物理合理”是不同能力 |
| [PhyProbe](https://research.nvidia.com/labs/cosmos-lab/phyprobe/) | 连续物理违规度量 | 与显式轨迹/事件标签、人类配对审查作三角验证 |
| [PhysResponse](https://arxiv.org/abs/2611.24419) · 2026-11-18 | gravity/friction/initial speed 的参数响应与关键事件跨界转变 | 最新的**反事实参数敏感性**框架；不能只看生成图像相似度 |
| [Simulating the Visual World with Artificial Intelligence: A Roadmap](https://arxiv.org/abs/2511.08585) | 隐式世界状态与 video renderer 的分工 | 理论/术语地图，非单独实验事实 |

## 4. 这批资料对 FFNmotion 的直接价值排序

**P0 · 优先精读并决定实验结构**
1. [ByteDance PhyWorld](https://seed.bytedance.com/public_papers/how-far-is-video-generation-from-world-model-a-physical-law-perspective)：排除“画面正确即学到物理规律”，设计 ID/OOD 与表象-动力学冲突对照。
2. [Runway Praxis-1](https://runway.com/research/praxis-1-video-pretraining-to-robot-control)：跨层 backbone→action expert，oracle future 与 imagined future 的误差分解。
3. [NVIDIA PhyProbe](https://research.nvidia.com/labs/cosmos-lab/phyprobe/) + [PhysResponse](https://arxiv.org/abs/2611.24419)：从观感指标走向连续物理违规和参数反事实响应。
4. [Kling understanding stack](https://arxiv.org/abs/2611.04916)：对由条件编码/训练对齐造成的系统失败建立上游控制。
5. [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) / [Cosmos Simulation](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/)：跨时间记忆与闭环状态；不同于单个 DiT block 的解释。

**P1 · 算子/架构参照**
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) / [Wan2.2](https://github.com/Wan-Video/Wan2.2)、[HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)、[CogVideoX](https://github.com/zai-org/CogVideo)、[Seedance 1.0](https://seed.bytedance.com/en/public_papers/seedance-1-0-exploring-the-boundaries-of-video-generation-models)：区别跨模态条件位置、FFN/attention 结构与时空训练。

**P2 · 功能分类与未来泛化场景**
- [World Labs taxonomy](https://worldlabs.ai/blog/taxonomy-of-world-models)、[Meta V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)、[Google Genie](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) / [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)、[Wayve GAIA-2](https://wayve.ai/thinking/gaia-2/)、[Cosmos 3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)：把最终研究从被动视频生成拓展至 action-conditioned dynamics 的候选方向。

## 5. 调查的局限与后续可检索项

- 本轮是覆盖主要研究主体的**广覆盖一手资料筛查**，而非每份技术报告全部附录、每个 checkpoint 的系统性代码复现。个别最新材料只有摘要/研究博客；其超出披露的内部机制记为 UNVERIFIED。
- **百度、华为、蚂蚁等团队**尚未找到同时满足“直接关联生成运动/世界动力学、可定位官方一手技术资料”的高可信材料，不为凑名单编造报告或内部方法。可继续在其研究栏目按 world model、physical dynamics、video diffusion、action-conditioned prediction 逐项补齐。
- 对闭源模型（Sora、Veo、Kling、Seedance、Hailuo、Atlas 等）只可讨论公开任务设计、产品行为和作者报告；不可声明实际发现了某个 FFN/attention 机制。
- 对企业自评的“物理准确度提升”“SOTA”均归类为 *作者报告*，在 FFNmotion 只用作可测候选，不作为独立结论。
- 2026-11 的 [Kling understanding stack](https://arxiv.org/abs/2611.04916)、[Runway Praxis-1](https://runway.com/research/praxis-1-video-pretraining-to-robot-control)、[NVIDIA Cosmos Simulation](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/) 和 [PhysResponse](https://arxiv.org/abs/2611.24419) 应优先跟踪修订、开源代码与新实验。

下一步请读 [06 · 机制知识综合](06_mechanistic_synthesis.md)，它将上述资料压缩成可区分的科学命题与算子对象。
