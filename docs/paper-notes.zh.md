# Neptune: Advanced ML Operator Fusion for Locality and Parallelism on GPUs

- **arXiv**: 2510.08726v2 (cs.PL)，发表于 PLDI 2026 (Proc. ACM Program. Lang. 10, PLDI, Article 220)
- **作者**: Yifan Zhao, Egan Johnson, Prasanth Chatarasi (IBM Research), Vikram S. Adve, Sasa Misailovic — UIUC
- **代码**: https://github.com/uiuc-arc/neptune

---

## 一句话概括

Neptune 是一个张量编译器：它把「**故意打破归约循环带来的数据依赖，再用代数修正项（repair term）把结果补回来**」变成编译器里可自动推导、可证明正确的调度原语，从而让编译器第一次能自动把朴素的 attention 代码生成出等价于 FlashAttention / FlashDecoding 的融合 kernel。

## 1. 要解决的问题

Attention 由 matmul → 逐元素 score → softmax（max/sum 两个归约）→ matmul 组成，融合收益极大（避免 L×L 中间张量的显存往返）。但现有编译器都做不了：

- **Tile 类编译器**（Triton、Pallas、TileLang）：抽象层次太低，融合要程序员手写在输入程序里。
- **Schedule 类编译器**（Halide、TVM）：抽象层次合适，但标准 loop fusion 的合法性检查会因为「归约循环之间的 loop-carried dependency」而**拒绝**这种融合。TVM 的 fusion legality check 直接禁止 reduction fusion，结果是多个 kernel + 额外的全局内存传输。
- 结果：人们只能手写 FlashAttention / FlashDecoding 这类专用 kernel，通用性和可组合性都很差。

论文指出两个缺口：(a) 通用化的算子融合是必要且可能的，只是现有编译器做不到；(b) tile 优化与 schedule 优化长期割裂——tile 优化器不管高层融合，schedule 优化器又会产生让人和 autotuner 都崩溃的细粒度搜索空间。

## 2. 核心思想：Break and Repair（打破—修复）

以 softmax 为例（论文 Figure 1）：

```
s_max: xmax[i] = max(xmax[i], inp[i,j])          # 归约 1
s_exp: xexp[i,j] = exp(inp[i,j] - xmax[i])        # 逐元素，依赖归约 1 的最终值
s_sum: xsum[i] += xexp[i,j]                       # 归约 2，依赖逐元素结果
```

把 `s_exp`/`s_sum` **朴素融合（naive fusion）** 进 `s_max` 的 `j` 循环里，会让 `xexp` 提前读到还没算完的 `xmax`（读到第 j 次迭代的中间值而非最终值），结果**错误**，所以传统编译器会拒绝这个融合。

Neptune 的洞见是：**接受这个错误，然后修正它**。让 `s_sum` 每次迭代除了累加当前项，还要把上一轮的结果「修好」：

```
xsum[i] = exp(xmax_prev[i] - xmax_cur[i]) * xsum[i]   # 修正项 h（repair term）
        + exp(inp[i,j] - xmax_cur[i])                 # 本轮新增项
xmax_prev[i] = xmax_cur[i]
```

修正项的直觉：**把上一轮结果中所有对旧值 `r` 的引用，整体替换为新值 `r'`**（tag update）。数值上，`exp(m_prev - m_cur) * Σexp(x - m_prev) = Σexp(x - m_cur)`，旧的 `m` 被干净地消掉。虽然中间每一轮 `s⟨j⟩` 都「不对」，但**最终 `s⟨3⟩` 与原始程序完全相等**——这正是 FlashAttention 在线 softmax 的同一件事，Neptune 把它一般化了。

### 2.1 形式化：代数修正分析（Algebraic Correction Analysis）

把消费端归约（fusee `L_t`）刻画成 reducer `f`（可结合）加项生成函数 `g`：

- `f: X×X→X`（如 `+`、`max`），`g(r, c)`：`r` 是生产端归约的结果，`c` 是不来自归约的输入。

修复函数 `h(t, r, r')` 的反例必须满足两条：

1. **替换性**：`h(g(r,c), r, r') = g(r', c)` —— 用 `r'` 替掉 `r`；
2. **与 reducer 可分配**：`h(x f y, r, r') = h(x,r,r') f h(y,r,r')` —— 这样 `h` 能穿透整个累加树。

关键定理（Eq. 5）：**只要 `g` 对 `c` 可逆，就能机械地造出 `h`**：

```
h(t, r, r') = g(r', g⁻¹_c(r, t))
```

softmax 实例化：`f(x,y)=x+y`，`g(m,x)=exp(x−m)`，`g⁻¹(m,t)=m+ln t`，于是 `h(t,m,m') = exp(m−m')·t`。

还处理了 `g` 不可逆的情形：做变量替换 `C := g_c(c)`，若 `g'(r,C)` 对 `C` 可逆仍可求解（例：`exp(r − relu²(c))`，令 `C = relu²(c)`）。

正确性有定理与证明（Theorem 4.1 / 4.4、Lemma 4.3，见附录 D）：把循环转成归约的 explicit form 后证明修复后的循环 nest 与原程序产出相同值；依赖 `f` 的结合律。浮点不严格满足结合律，所以论文用实验校验数值精度（附录 F）。

### 2.2 两个实例

| | **Rolling Update**（滚动更新） | **Split-K Update**（切 K 更新） |
|---|---|---|
| 做法 | 沿 rolling loop 融合，用 `h` 做**逐迭代** tag-update | 先做 naive fusion + reduction privatization，拆成 local/global 两层归约 |
| 修复位置 | 施加在 reducer 的**左操作数**（累加值上） | 施加在 reducer 的**右操作数**（partial 结果上） |
| 并行性 | rolling loop 因跨迭代数据流变难并行 | rolling loop 高度可并行（local 归约并行跑，global 归约合并+修复） |
| 代价 | 多一点计算 + 需要缓存前一轮归约结果（寄存器/SMEM，量很小） | 需要额外空间存 partial 结果（由 tile size k 控制） |
| Attention 场景 | 生成 **FlashAttention** 等价物，适合 prefill | 生成 **FlashDecoding** 等价物，适合 decode |
| 实测使用 | 全部 PF benchmark 都用它 | 全部 DC benchmark 都用它 |

Rolling update 算法四步（Algorithm 1）：① 数据流重组（沿依赖路径 inline 中间循环，找出生产端归约集合 `L_r`）；② 朴素循环融合（明知结果错）；③ 用符号求解器（SymPy）从 `g` 解出 `h` 并验证可分配性；④ 改写 `L_t` 的 compute expression 施加 `h`，并把每个 `L_r` 改造成同时保留 prev/curr 两轮输出。任一步失败（前提检查、模式匹配、求解）就整体放弃、返回原程序——**安全失败**。

### 2.3 适用面（不止 attention）

论文明确给出的例子：
- **Attention**：两次 reduction fusion（softmax 内的两个归约 + 第二个 matmul 融入 softmax），得到 `g=exp(c−r)`、`h(t,r,r')=t·exp(r'−r)`。
- **Performer**：大量小步骤，融合成 2 个循环 nest，用了 4 次 reduction fusion。
- **数值稳定的 L2 norm**：`||x||₂ = m·√(Σ(xᵢ/m)²)`，`m = max|xᵢ|`，把求和融合进最大值归约，单趟完成；`h(t,r,r') = t·(r²/r'²)`。
- **RMSNorm + 动态缩放量化**：融合 `mean(x²)` 与逐行 max 两个归约；`h` 随 max 可分配（因系数恒非负且单调）。

**已知失败案例**：Welford 算法（在线 mean/variance）。符号求解器不会利用「前/当前均值、当前方差、新样本互相关联」这一点来化简表达式，因此找不到 `h`。作者承认这属于「融合本质上困难」的情形。

## 3. 系统架构：打通 schedule 与 tile 两级优化

```
Tensor Expression + High-level Schedule Template
        │
        ▼
Template-Guided Optimizer      (§4) 高层：融合、block-tile、数据放置/布局……
   ├─ 标准 loop/layout/placement 变换
   └─ rolling_update / split_k_update（新原语，可与其它变换组合）
        │  loop-scalar IR（loop nest + scalar expr）
        ▼
Loop-to-Tile Translator        (§5) 自动、覆盖 loop-scalar IR 大部分程序
   ├─ Tileable Nest Detection（向上走到第一个动态大小循环或非循环语句）
   ├─ Access Region Division（标量访问 → tile 切片访问）
   └─ Dimension Order Reconciliation（新贡献，见下）
        │  HTile IR（tile 表达式 + reduce/permute/dot）
        ▼
Tile Optimizer（线程 tile、TensorCore、swizzle、数据搬运）→ Kernel
        │
        ▼
Autotuner（复用 TVM MetaSchedule，变异 schedule 模板与参数）
```

设计要点：**职责分离**——高层「表达式的代数重写」放在 loop-scalar IR 上做（适合 rolling update 这类操作），低层硬件相关优化交给现成的 tile 优化器；schedule 因此保持简短，搜索空间也不会爆炸。

**Dimension Order Reconciliation**（§5.1）是翻译部分的新点：光有 tile 切片区域会丢信息。例如 `y3[vi,vj] = x1d[i0,i1] * xtrans[vj,vi]` 若只切区域会得到错误结果，因为丢失了循环顺序编码的广播/转置语义。Neptune 借鉴 Einstein notation / einops，给每个 tensor access 提取维度序 `d`（如 `(j1,i1)`），再对齐到循环序 `d_L`，通过三步改写：
- **dimension squeezing**（去掉 `None`，切片退化为标量索引）
- **permutation**（arg-sort 得到 `permute` 顺序 `σ`）
- **dimension unsqueezing**（补 `NewAxis` 广播）

这使翻译比依赖模式匹配（例如专门识别 TensorCore matmul 的转置输入）的传统 tensorization 更通用。代价：拒绝同一个内层循环变量被多次或非线性使用（`tensor[i1,i1]`、`tensor[i1*i2]`），但这在 ML tensor 程序中罕见。

## 4. 实现与评估

- **实现**：基于 TVM（loop-scalar IR / HTile IR 扩展 TensorIR，schedule 语言扩展）+ Triton（tile 优化器与代码生成），SymPy 求解 `h`，MetaSchedule 做 autotuning。共约 10K 行 C++/Python（其中融合变换 3K 行，loop-to-tile 翻译 1.3K 行）。
- **硬件**：Nvidia A100、RTX A5000、RTX 6000 Ada、AMD MI300（4 种架构，含 2 个数据中心卡）。
- **负载**：10 个来自真实 LLM/VLM 的 attention 变体（Global / Causal / ALiBi / GQA / SoftCap / Window，PF+DC），序列长 128 ~ 32768；另有 L2 norm、RMSNorm+量化、Performer 三个非 attention 算子。
- **结果**：
  - vs 编译器基线（Triton×3、FlexAttention、TVM、Mirage）：320 个 setup 中 284 个取得最低延迟；总体 geomean **1.35×** 加速，Nvidia 上最高 2.65×，AMD 上最高 3.32×；分架构平均 1.15×(6000Ada) / 1.14×(A5000) / 1.38×(A100) / 1.85×(MI300)。
  - vs Mirage（多层级超优化器）：geomean **2.21×** 更低延迟，且 Neptune 提供确定性正确性保证，而 Mirage 用整数输入做概率性正确性检验（对浮点数值误差无感知，attention 上偶有数值不稳定）；Mirage 还会在某些形状下找不到合法 kernel（表中 ✗）。
  - vs 手工库基线（PyTorch+inductor、cuDNN、Tri-Dao CUTLASS、FlashInfer）：256 个 setup 中 101 个不劣于所有库；总体 geomean 1.07×。流行算子/GPU 上手工库仍有优势（Window (PF) 是 Neptune 明显落后的一项，因为其基于 TVM 的循环分析无法识别正确的条件来切分循环）。
  - **可编程性**：Tri-Dao 的 attention Triton kernel 约 650 行；Neptune 输入 = 通用 attention 定义 38 行 + schedule 28 行（附录 A）。
  - 消融（Fig. 10）：融合是收益最大的单项优化，其次是 autotuning（对 Global 显著，对 Causal 较小）。
  - 资源与可扩展性：平均只用寄存器最多基线的 0.68×、SMEM 最多基线的 47%；batch size 加压到显存上限，吞吐接近最好基线（略逊 cuDNN）。
  - **数值精度**（附录 F）：对 FP64 未融合 attention 求 RMS/90%/99% 误差，所有 prefill 算子上 Neptune 误差低于 Tri-Dao 与 FlexAttention，decode 上优于 Tri-Dao、与 FlexAttention 相当。

## 5. 与相关工作的定位

- vs **FlashAttention / FlashDecoding**：数学洞察一致，但它们是「手写的、attention 专用的、被证明过的」变换；Neptune 提供**通用机制**——求解器自动推导修复项，因此能覆盖 attention 变体、Performer、L2 norm、量化统计等，并能与编译器其它优化组合。
- vs **Tile 编译器**：Neptune 把 tile 层当作后端重用，把高层融合交给 schedule 层。
- vs **图级 / 超优化**：TASO/PET 等在算子图上做代数重写，无法表达「loop fusion 与代数重写必须协同发生」的变换；Mirage 是分层超优化但依赖概率性正确性与代价高昂的搜索。
- vs **Hogwild / Deiana 等打破依赖的工作**：那些要么不修复（产生近似结果、非确定性），要么在运行时修复（适合 CPU，不适合确定性 tensor 负载）；Neptune 是**编译期静态变换，在实数域上恢复等价性**。

## 6. 局限与后续

- 求解器覆盖有限（Welford 失败）；浮点结合律不成立，理论等价性靠实测精度背书。
- 翻译层不支持内层循环变量的重复/非线性使用。
- 个别算子落后于手工库（Window PF、部分短序列 cuDNN 场景）。
- 后续工作 **Nautilus**（arXiv:2604.14825）把变换推广到 Hopper/Blackwell 新架构并自动化调度。

## 7. 为什么值得关注（我的解读）

1. **范式层面**：把「融合合法性检查说不」从**终止条件**变成**待修复的中间状态**。这与 FlashAttention 的核心 trick 同源，但被抽象为「构造 h 的两个代数条件 + 可逆性求解」的通用编译期分析，这是一个可复用的编译器原语（论文把它实现成 `sch.rolling_update(block, loop)` 这样一行调度调用）。
2. **工程层面**：不重造轮子——schedule 层用 TVM、tile 层用 Triton，新贡献集中在 loop-scalar IR 上的代数重写 + 一个 loop→tile 的通用翻译（含维度序调和）。这条「两级 IR 桥接」的路线对任何想同时要「高层可编程性」和「低层性能」的编译器都有参考价值。
3. **正确性层面**：与 Mirage 的概率性验证相比，Neptune 走的是符号求解 + 定理证明 + 精度实测的路线，对算子库级别的部署更可接受。
