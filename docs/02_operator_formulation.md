# 02 · 算子定义：Head 写入、FFN 有限响应与下游因果传播

> 状态：公式与源码结构已核对；具体 Wan 权重上的数值验收尚未执行。所有“精确”均指给定前向状态、实数算子定义下的恒等式，实际 BF16/混合精度需设置数值容差。

## 0. 研究坐标与张量契约

**模型**：官方 Wan2.1-T2V-1.3B，30 blocks，\(d=1536\)，\(m=8960\)，12 heads/block，head_dim=128；FFN 为 Linear → GELU(approximate='tanh') → Linear。T2V 的 cross-attention 为文本 Q/K/V 与输出投影，空间/时间 RoPE 位于 self-attention 的位置路径，不应将两者混淆。

| 索引 | 含义 | 与其他时间变量的区别 |
|---|---|---|
| \(s\) | 去噪/采样步骤 | 采样器积分时间 |
| \(\ell\) | Transformer block | 网络计算深度 |
| \(p\) | 视频 latent 时空 token | 与解码视频帧 \(f\) 不同 |
| \(f\) | 输出视频帧 | 物理时间/观察序列 |
| \(h\) | cross-attention head | \(0 \ldots 11\) |
| \(j\) | FFN 隐藏通道 | \(0 \ldots 8959\) |
| \(b\) | CFG branch: conditional / unconditional | 与 batch 维分开记录 |

对固定的 \((s,\ell,b)\)，令 \(r_p\in\mathbb R^d\) 是执行完该层 self-attention 残差后、cross-attention 前的状态。令 \(c_p\) 是 cross-attention 更新后、FFN 前的状态。

源码：[WanAttentionBlock](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py)；[T2V 1.3B config](https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/wan_t2v_1_3B.py)。

## 1. 恢复官方 block 的真正计算流

Wan2.1 的该 block 先执行 self-attention 残差：

\[
r=x+\operatorname{SA}\big(\operatorname{norm1}(x)\odot(1+e_1)+e_0\big)\odot e_2.
\]

然后 cross-attention 与 FFN：

\[
c=r+\operatorname{CA}(\operatorname{norm3}(r),\,E_{\mathrm{text}}),
\]

\[
u=T_s(c)=\operatorname{norm2}(c)\odot(1+e_4)+e_3,
\]

\[
F(u)=W_2\phi(W_1u+b_1)+b_2,\qquad
B(c)=c+e_5\odot F(T_s(c)).
\]

这里 \(\phi\) 是 Wan 的 tanh-approximate GELU；调制向量 \(e_0,\ldots,e_5\) 由当前 timestep 的嵌入与层参数确定。输出投影与 FFN 内 bias 均按原源码保留。

在这组分析中，**只干预某个 head，固定此时 \(r\)、文本 embedding、timestep、其他 head、参数与 CFG branch**。FFN 的输入在干预后需要重新经过 norm2/AdaLN，而不能把 \(U_h\) 直接加到 FFN 激活上。

## 2. 精确恢复一个 head 的残差写入

第 \(h\) 个 head 的 attention 输出：

\[
z_{hpa}=\sum_q A_{hpq}V_{hqa},\qquad z_{hp}\in\mathbb R^{128}.
\]

输出投影 \(W_O\in\mathbb R^{1536\times1536}\) 可按 head 的 128 维列块分组。定义：

\[
\boxed{(U_h)_{pi}=\sum_a (W_O^{(h)})_{ia}z_{hpa}}
\]

它是该 head **经过输出投影后写入残差流的向量**，而不是 attention heatmap，也不是输出投影前的 128 维 head 激活。

在不修改其他 head 的局部反事实中：

\[
c^+=c^-+U_h.
\]

输出投影的共享 bias 在差分中抵消。实现时参考作者 [head ablation](https://github.com/Siriuslala/physics/blob/main/wan21_t2v_experiments/cross_attn_head_ablation.py)：在展平 heads、调用 \`cross_attn.o\` 之前置零指定 head。对 T2V 不应误用包含图像 context 的 I2V cross-attention 路径。

## 3. Head → FFN 隐藏通道的精确有限差分

固定 \(u^\pm=T_s(c^\pm)\)，记第 \(j\) 个 FFN preactivation、activation：

\[
a_j^\pm=w_j^\top u^\pm+b_{1,j},\qquad
m_j^\pm=\phi(a_j^\pm).
\]

head 引起的通道变化：

\[
\Delta m_j=m_j^+-m_j^-.
\]

第 \(j\) 个通道额外产生的残差写入：

\[
\boxed{D_{hj}=e_5\odot v_j\,\Delta m_j}
\]

其中 \(v_j=W_{2,:,j}\)，\(\odot\) 是逐残差维门控。

因此：

\[
\boxed{B(c^+)-B(c^-)=U_h+\sum_{j=1}^{m}D_{hj}}
\tag{A}
\]

**推导**：\(B(c)=c+e_5\odot(\sum_jv_j\phi(w_j^\top T_s(c)+b_{1,j})+b_2)\)，两状态相减后，\(c\) 的差是 \(U_h\)，共同的 \(b_2\) 抵消；其余项逐通道分解即得。式 (A) 是固定输入对上的等式，不依赖梯度线性化或训练 transcoder。

### 割线/积分形式：Q1r2 启发的解释层

定义 \(\Delta u=u^+-u^-\)，沿 FFN 输入直线：

\[
u_\tau=u^-+\tau\Delta u,\quad 0\leq\tau\leq1.
\]

令：

\[
\bar\beta_{hj}=
\int_0^1\phi'\!\left(w_j^\top u_\tau+b_{1,j}\right)\,d\tau.
\]

由微积分基本定理：

\[
\Delta m_j=\bar\beta_{hj}\; w_j^\top\Delta u,
\]

\[
\boxed{
F(u^+)-F(u^-)=
W_2\operatorname{Diag}(\bar\beta_h)W_1\Delta u.
}
\tag{B}
\]

当 \(\phi\) 为 GELU 时，\(\bar\beta\) 不必都非负，须使用实际的导数。由于 norm2 是非线性的，**\(\Delta u\neq\operatorname{Diag}(1+e_4)U_h\)**，正确式子是：

\[
\Delta u=\left[\operatorname{norm2}(c^+)-\operatorname{norm2}(c^-)\right]\odot(1+e_4).
\]

式 (B) 与 Q1r2 使用的增量算子具有相同的数学骨架，但本项目保留输入对所决定的真实 \(\bar\beta_h\)，不把它替换为校准的固定 \(\alpha\)。

实现时的数值 oracle 应优先使用 \(\phi(a^+)-\phi(a^-)\)；\(\bar\beta\) 可以用数值积分或 \(\Delta m_j/\Delta a_j\) 估计，并专门处理 \(\Delta a_j\approx 0\) 的病态区间。

## 4. 什么决定 Head 的写入能否被 FFN 使用？

沿 \(c_\tau=c^-+\tau U_h\)，对一个 FFN 通道有：

\[
\Delta m_j=
\int_0^1
\phi'(a_j(T_s(c_\tau)))\,
w_j^\top J_{T_s}(c_\tau)U_h\,d\tau.
\tag{C}
\]

所以影响由 **head 写入方向 → norm/AdaLN 的切向变换 → W1 读取 → GELU 局部斜率 → W2 写入与门控** 决定。对已有 Wan 权重，这些项可分别记录。

若忽略归一化的可学习仿射参数，带 epsilon 的 LayerNorm Jacobian：

\[
J_{\rm LN}(c)=\frac1{\sigma_\epsilon}
\left[
P-\frac{\tilde c\tilde c^\top}{d\sigma_\epsilon^2}
\right],
\quad
P=I-\frac{\mathbf1\mathbf1^\top}{d},
\]

其中 \(\tilde c=Pc\)，\(\sigma_\epsilon^2=\|\tilde c\|_2^2/d+\epsilon\)。实际 Wan norm2 的完整 Jacobian 还应包含其 affine 参数。这个表达解释了为什么较大的 \(U_h\) 可能主要落在归一化不敏感方向，或被后续 W1/W2 重新定向。**这是候选机制，不是当前已测 Wan 结果。**

### 抵消与协同

定义：

\[
\kappa_h=
\frac{\left\|\sum_jD_{hj}\right\|_2}
{\sum_j\|D_{hj}\|_2+\epsilon}.
\]

较小 \(\kappa\) 表示原始通道写入间存在强抵消；必须同时报告分子和分母规模，否则小数值也可能只是所有写入都接近零。注意这种分解依赖原始 FFN 隐藏基底，跨模型或重参数化比较应转向输出子空间等坐标不敏感对象。

## 5. 控制性路径干预（不是自然中介效应）

令 \(B^-=B(c^-)\)。对候选 FFN 通道集合 \(S\)，构造：

\[
\boxed{B(d,S)=B^-+dU_h+\sum_{j\in S}D_{hj}},
\quad d\in\{0,1\}.
\tag{D}
\]

关键四组：

| 干预 | 直接 head 写入 | FFN 路径写入 |
|---|---|---|
| \(B(0,\varnothing)\) | 关闭 | 关闭 |
| \(B(1,\varnothing)\) | 开启 | 关闭 |
| \(B(0,S)\) | 关闭 | 恢复 \(S\) |
| \(B(1,S)\) | 开启 | 恢复 \(S\) |

完整集合 \(S_{\rm all}\) 有 \(B(1,S_{\rm all})=B^+\)，这是模块级恢复 oracle。只恢复 FFN 写入的状态 \(B(0,S)\) 可能偏离自然计算分布；其效应属于**可控路径干预**，不能自动解读成自然间接效应。

令 \(Q\) 是后续网络中的明确代理评分或最终视频的运动观测，则可以比较：

\[
E_{\rm FFN}(S)=Q(B(0,S))-Q(B(0,\varnothing)),
\]

\[
E_{\rm direct}=Q(B(1,\varnothing))-Q(B(0,\varnothing)),
\]

\[
I(S)=Q(B(1,S))-Q(B(1,\varnothing))
-Q(B(0,S))+Q(B(0,\varnothing)).
\]

后续网络和采样器非线性使 \(I(S)\) 可以不为零，即使式 (A) 的 block 差分是精确加法式。

补充测试：在保留原 head 的正常前向中抑制目标通道的 *head-induced increment*，以比较必要性与恢复性；不要把所有通道的正常激活清零当作同一干预。

## 6. 跨层传播与“功能可观测性”

令 \(\mathcal B_\ell\) 为完整 block。对残差扰动的一阶传播：

\[
\Delta x_k\approx
J_{\mathcal B_k}\cdots J_{\mathcal B_{\ell+1}}\Delta x_\ell.
\]

这个 Jacobian 作用于**全部时空 tokens**，self-attention 引入跨 token/跨帧耦合，不能使用逐 token 的独立 Jacobian 代替。实际计算使用 JVP/VJP，不显式构建高阶稠密矩阵；每条重要推断仍需有限幅度 activation patching 验证。

如果 \(Q\) 是可微的运动代理读出，可用一阶筛选分数：

\[
S_{hj}=\sum_{p\in\mathcal R}
\langle\nabla_{B_p}Q,D_{hjp}\rangle.
\]

\(S_{hj}\) 表示固定状态附近对特定代理目标的局部敏感性，**不是**通道对最终物理运动的真实因果效应，且正负贡献可能抵消。

跨层观察要区分：
- 下游 FFN 对上游干预的*响应*；
- 通过中间路径恢复而引发的*功能效果*；
- 改变最终视频轨迹的*生成行为*。

## 7. 去噪步骤与 CFG 分支必须进入因果图

对标准 classifier-free guidance：

\[
v_{\rm CFG}=v_u+w(v_c-v_u).
\]

当 head 干预改变两条分支：

\[
\Delta v_{\rm CFG}=w\Delta v_c+(1-w)\Delta v_u.
\]

必须测试 conditional-only、unconditional-only、both，并锁定每次调用的 branch 身份；不能仅根据调度步序号猜测 branch。作者原始 head 消融补丁并未默认区分这两条路径。

完整生成映射还包含采样器隐状态 \(H_s\)：

\[
(z_{s+1},H_{s+1})=
\Phi_s(z_s,H_s,v_{{\rm CFG},s}).
\]

若使用 UniPC 等多步积分器，配对干预必须从同一 \((z_s,H_s)\) 快照继续，而非只重用 latent；并保留随机状态、noise schedule、模型模式与完整生成配置。

最终效应：

\[
\Delta \mathcal M=
\mathcal M\!\left(\operatorname{Decode}(\operatorname{Continue}(B^+))\right)
-
\mathcal M\!\left(\operatorname{Decode}(\operatorname{Continue}(B^-))\right).
\]

检测到物体消失/严重变形时，应记录为另一个结果类型，不把它算成“运动更小/更稳定”。

## 8. 表征几何：从原始通道走向坐标无关对象

若原始单通道效应不稳定，收集不同条件下的写入 \(\Delta F_h^{(n)}\) 组成矩阵，分析 SVD 的输出方向与稳定性。SVD 的大奇异值只意味着*高写入方差*，不意味着物理相关。

再用与运动标签分离的验证条件下训练的 probe / CCA / task-weighted subspace 定义候选方向，并进行方向/正交补的因果干预。

物理方向 \(\theta\) 可以通过 \(r(\theta)=[\cos\theta,\sin\theta,\ldots,\cos K\theta,\sin K\theta]\) 表示：**内禀自由度可为 1，线性张成维度可为 \(2K\)**。因此“需要许多正交 probe”并不证明物理状态的内禀维度很高。

只有原始 FFN 通道或线性方向不足以支撑可复核回路时，才考虑训练 timestep-conditioned transcoder。此时可考虑同时约束正常输出与 head-induced 差分：

\[
\mathcal L=
\lambda_{\rm recon}\|F(u)-\widehat F(u)\|^2
+
\lambda_\Delta
\|\Delta F_h-\Delta\widehat F_h\|^2
+
\lambda_{\rm sparse}\|z\|_1.
\]

其归因是否精确，取决于是否构造冻结的局部替代模型及其误差；不能混同于原始 Wan 的精确式 (A)。

## 9. 实现绑定与验收标准

**实际源码绑定**：

- Wan block: \`wan/modules/model.py::WanAttentionBlock.forward\`
- Head: \`WanT2VCrossAttention.forward\` 内 flash_attention 之后、\`self.o\` 之前的逐头激活
- FFN 前状态: \`cross_attn_ffn\` 中 cross-attention 残差加法之后、\`norm2\` 之前
- FFN 隐藏: \`ffn[0]\`、\`ffn[1]\`；写回: \`ffn[2]\` 与门控 \`e[5]\`

采集时必须保证 \(z_h\)、\(c^+\)、\(e_3/e_4/e_5\) 来自同一模型调用、同一 layer/step/branch、同一 token 布局；完整记录 dtype、shape、模型权重身份、seed。跨 batch CFG、梯度 checkpoint、model offload 的具体生命周期需要运行时预检。

**模块验收**：
1. 未干预 monkey patch 与原模型输出在相同数值容差内一致。
2. 重建的 \(\sum_h U_h+\text{output-bias}\) 等于 \`cross_attn.o\` 的实际输出。
3. 重新计算的 \(F(T_s(c^+))\) 与捕获的 FFN 输出一致。
4. 式 (A) 与 \(B(1,S_{\rm all})=B^+\) 成立，报告相对误差与最大绝对误差。
5. Top-K 只改变目标 FFN hidden increment，其他原始状态与路径保持固定。
6. 独立运行的完整生成视频证明至少一种行为指标发生改变，才可声称找到具有生成因果作用的路径。

**当前执行状态**：以上是分析/验收规范；尚未运行真实 Wan 权重。

下一步：[03_experimental_plan.md](03_experimental_plan.md)。
