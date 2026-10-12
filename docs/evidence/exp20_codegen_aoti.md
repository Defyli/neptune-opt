# Milestone 12 / 13 — 真·代码生成，以及把生成的内核装进 AOTI 产物

测试环境：单卡 GPU，torch 2.11.0+cu128，triton 3.6.0，RTX 5090 D v2。
模块：`neptune_codegen/`（本地与远端 md5 一致：`__init__.py` = `c394ba98…`）。

---

## 0. 这一轮修掉的误解

前一轮的 `neptune_codegen` 虽然已经能生成内核，但骨架里写死了太多东西，且只有
`logsumexp` / `stable_l2` 两个算子。这一轮把它改成：

```
算子 = 一个 sympy 表达式 g(s, c) + finalize + producer 语义
repair R(s,s') / term T(c,s') / 全部 Triton 源码 = 派生出来的，不是手写的
```

`power` 不再是一个输入参数——它是推导的**输出**。

---

## 1. Milestone 12：只写代数，新增算子

两个算子在 demo 里只以代数形式声明，仓库里没有任何一个手写内核对应它们：

| 算子 | `g` | 推导出的 R | 推导出的 T | 最差相对误差 |
|---|---|---|---|---|
| `sumsq_div` | `c**2/s` | `s/s2` | `tl.where(mask, (v*v)*((1.0/den)), 0.0)` | `1.82e-06`（outlier 4×16384） |
| `lse_shifted` | `exp(c/2 - s/2)` | `exp(s/2 - s2/2)` | `exp(0.5*v + (-0.5)*den)` | `1.11e-06`（all_negative 32×512） |

同时验证了**拒绝路径**（这是框架正确性的一半）：

```
[PASS] stable_l1_ratio    refused: g 不能对 c 解析求逆（Abs → Piecewise 带 nan 分支）
[PASS] exp_abs_shift      refused: 同上
VERDICT: all auto-derived operators correct
```

`sumsq_div` 生成出来的完整源码（无一行手写）：

```python
@triton.jit
def sumsq_div(x_ptr, out_ptr, rows, ncols, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row < rows:
        base = x_ptr + row * stride_row
        m = 0.0
        acc = 0.0
        for off in range(0, ncols, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            mask = idx < ncols
            v = tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
            a = tl.abs(v)
            m_new = tl.maximum(m, tl.max(tl.where(mask, a, -float("inf")), axis=0))
            den = tl.maximum(m_new, 1e-30)
            scale = tl.where(m == 0.0, 1.0, (m / den))
            acc = acc * scale + tl.sum(tl.where(mask, (v * v) * ((1.0 / (den))), 0.0), axis=0)
            m = m_new
        tl.store(out_ptr + row, acc)
```

### 1.1 一个真实的方法论教训（`finalize` 是唯一必须由人声明的部分）

`sumsq_div` 第一版的声明是 `finalize = m*acc`，文档写的是「sum x²/max|x|」。
数值测试报 `rel_err = 3.4`——正好是 `max|x|` 这个因子。

- `g` 决定**融合是否合法**（条件 (3)(4)），求解器可以判定。
- `finalize` 决定**这个算子是什么意思**，机器无法从 `g` 推出来。
- 因为累加器不变量是 `acc = Σ_j x_j²/s_final`，正确写法是 `finalize = acc`（恒等）。

更值得记的是：**m12 与 m13 都用 `m*acc`，而 m12 的参考实现也是照着同一个误读写的，
所以 m12 是 PASS 的**。自洽的错不会被抓到；能抓住它的是「按算子文档而不是按公式
写参考实现」。这一条已写进两个文件里。

---

## 2. Milestone 13：把生成的内核装进 AOTI 产物

### 2.1 接入链路（全部是 torch 现成机制，没有手写 IR）

```
配方的代数 (g, finalize, producer)
  -> neptune_codegen 生成真的 Triton 源文件
  -> torch.library.triton_op("neptune::<op>")
  -> torch.export 分解成 triton_kernel_wrapper_functional
  -> Inductor 降级为 ir.UserDefinedTritonKernel
  -> AOTI 把生成的内核作为 .cubin 打进 .pt2
```

`decompose_custom_triton_ops = True`，所以 export 不会把内核降成普通 aten。

### 2.2 选择器自动命中

替换 pass 作用在 export 后的 ATen 图上，**三个算子**都被自动识别并整体替换成一个调用：

```
stable_l2  matched=stable_l2  graph now: ['aten.abs.default', 'aten.amax.default',
    'aten.clamp_min.default', 'aten.div.Tensor', 'aten.pow.Tensor_Scalar',
    'aten.sum.dim_IntList', 'aten.sqrt.default', 'aten.mul.Tensor',
    'aten.squeeze.dim', 'neptune.stable_l2']

sumsq_div  matched=sumsq_div  graph now: ['aten.abs.default', 'aten.amax.default',
    'aten.clamp_min.default', 'aten.mul.Tensor', 'aten.div.Tensor',
    'aten.sum.dim_IntList', 'aten.squeeze.dim', 'neptune.sumsq_div']

logsumexp  matched=logsumexp  graph now: ['aten.amax.default', 'aten.sub.Tensor',
    'aten.exp.default', 'aten.sum.dim_IntList', 'aten.log.default',
    'aten.add.Tensor', 'aten.squeeze.dim', 'neptune.logsumexp']
```

### 2.2.1 「生成通用」vs「识别按族」——必须说清楚的分工

`logsumexp` 是 PyTorch **已经**自动融合的那一个算子（`online_softmax`），把它一起跑
才是同类比较。它的图与另外两个不同：生产端是**带符号**的 `amax`（没有 `abs`），
消费端是 `exp → sum → log`。

这暴露出一个重要且诚实的区分：

- **生成是完全通用的**：新算子 = 一个新 sympy `g`，别的什么都不用写。
- **识别是按族的**：判定「`amax(x.abs())` + `sum(x*x/m)` 就是骨架 `g = c²/s`」
  本身就是一条模式规则，必须有人写下来。

生产端的分支正是族与族的分界：`max(|v|)` 配幂范数族，**带符号的 `max(v)`** 配指数折叠族。
Inductor 自己的 softmax 识别（`prepare_softmax_extra_check`）同样是手写的，这不是本模块的缺陷，
但不该被含糊过去。

### 2.3 产物证据（本次的核心结论）

```
    stable_l2
      baseline      : cubins=  1  artifact mentions 'stable_l2'   0x  in_ptr0 loads=8
      selected+gen  : cubins=  1  artifact mentions 'stable_l2'  41x  in_ptr0 loads=0
      size          : 433 KiB -> 426 KiB

    sumsq_div
      baseline      : cubins=  1  artifact mentions 'sumsq_div'   0x  in_ptr0 loads=8
      selected+gen  : cubins=  1  artifact mentions 'sumsq_div'  41x  in_ptr0 loads=0
      size          : 432 KiB -> 425 KiB

    logsumexp
      baseline      : cubins=  1  artifact mentions 'logsumexp'   0x  in_ptr0 loads=7
      selected+gen  : cubins=  1  artifact mentions 'logsumexp'  41x  in_ptr0 loads=0
      size          : 443 KiB -> 430 KiB
```

- `artifact mentions '<op>'`：在 `.pt2` 的**全部字节**（含 cubin 符号表、`aotinductor/`
  下的源码/元数据）里数生成内核的名字。基线 **0 次**，选中后 **41 次**。
  这是本里程碑的载重数字：它区分了「一个恰好很快的内核」和「产物里装着代数要求的内核」。
- `cubins >= 1` **单独不构成证据**：基线也是 1。
- `in_ptr0 loads` 从 8 降到 0：基线里那 8 处是 Inductor 自己生成的 triton 源码中的
  输入指针；选中后**一个 Inductor 生成的内核都不剩了**，整个计算塌缩成我们生成的那一个内核。

### 2.4 数值（eager 自定义算子 vs 分解式参考，float32）

```
[PASS] stable_l2  (64, 4096)    rel_err=2.386e-07
[PASS] stable_l2  (1024, 16384) rel_err=1.066e-06
[PASS] stable_l2  (8, 1000)     rel_err=1.191e-07
[PASS] sumsq_div  (64, 4096)    rel_err=3.317e-07
[PASS] sumsq_div  (1024, 16384) rel_err=1.059e-06
[PASS] sumsq_div  (8, 1000)     rel_err=2.379e-07
[PASS] logsumexp  (64, 4096)    rel_err=1.086e-07
[PASS] logsumexp  (1024, 16384) rel_err=9.365e-08
[PASS] logsumexp  (8, 1000)     rel_err=0.000e+00
```

### 2.5 为了让它在 torch 2.11 上跑起来，踩到的两个坑

1. **`triton_op` 的 schema 推断**。torch 2.11 的 `triton_op` 没有可用的 `schema` 参数，
   它无条件调用 `infer_schema`，而 `infer_schema` 会 `eval` 注解字符串。本模块用了
   `from __future__ import annotations`（注解是字符串），且为了「无 GPU 也能 import」
   把 `torch` 写成了函数内 import，于是 eval 找不到 `torch`：
   `ValueError: Unsupported type annotation torch.Tensor`。
   修法：注册时 `globals().setdefault("torch", torch)`，把名字放到 eval 会看的地方。
2. **pattern 替换必须整体替换**。FX 节点只有在没有任何 user 之后才能 `erase_node`；
   只换归约那一块会留下其余计算，生成的内核永远不是输出的唯一生产者。

---

## 2.6 修正（milestone 14 之后）

本文档 §2.3 的 41x 证据来自 **"trace 期间就调用生成算子"** 的路径（`Fused` 模块），
**不是**来自"导出后改图"。当时那条路径只打印了改好的算子列表，**没有走过 AOTI**。

milestone 14 把它量清楚了（`gpu_diag_paths.py`）：在 `torch.export` 之后插入该算子，
图里确实出现 `neptune.<op>`，但 ExportedProgram 从未登记它的 Triton kernel，
**产物里 0 次出现**；重新 export 那个 GraphModule 也还是 0。
只有 export **之前**就在 FX 图里改，才回到 41x。

所以"生成的内核进产物"这个结论仍然成立，但**成立的放置位置**是 export 之前。
详见 `DESIGN_general_module.md` §3.23 与 `exp24_recognition.txt`。

## 3. 回归

```
TOTAL: 101 passed, 0 failed, 101 checks
```

`gpu_m10_codegen_tests.py` 在加入 `as_triton_op` 之后仍然全绿。
`stable_l2` 在 (8,65536) 上 1.29x、(32,65536) 上 1.34x，输入读取遍数 4 → 1。

---

## 4. 仍然没做的（诚实边界）

1. **多生产端骨架**（论文 Eq.8/9，例如 RMSNorm+max）没有实现。
2. **`NEEDS_STATE`（Welford 类）的状态携带 codegen** 没有实现。
3. **替换点目前作用在 `torch.export` 之后**，属于「导出后、AOTI 编译前」的图改写。
   如果要在 `post_grad_custom_pre_pass` 里插入（真正意义上的编译器内 pass），
   需要在 pass 里手工构造 `triton_kernel_wrapper_functional` HOP，因为 triton 内核的
   捕获发生在 tracing 阶段。
4. **性能没有在带宽受限的大张量上量化**（当前尺寸下峰值带宽只有 ~186 GB/s，形状受限）。
