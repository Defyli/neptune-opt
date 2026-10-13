# 两趟方案：没有修复时怎么办 —— 以及一次"退出码审计"查出来的东西

这一轮做了两件事，第二件比第一件重要。

第一件是**实现**：给"不存在修复"的算子加了一条退路（两趟），覆盖率 20 → 26 / 28。
第二件是**审计**：为了让自动化能判成败，我给 GPU 脚本加了退出码和断言——然后立刻就查出
这些脚本**一直在报告成功**，其中 m13 的关键证据**从来没有真正跑出来过**。

## 一、机制：为什么是"两趟"，而不是官方那个 `ReduceToScan`

`NEEDS_STATE` 那 5 族的形状是 `g = (c - mean)^2`。问题不在累加器，在**项**：
项要的是**均值本身**，而把 `sum((c-s)^2)` 从旧的 `s` 修到新的 `s'` 需要
`sum((c-s)^2) = sum((c-s')^2) + n*(s'-s)^2` —— 需要**计数** `n`，这是本框架不携带的状态。

退路不是更多代数，而是**再读一遍这一行**：把"声明"这件事本身当成改动。

```python
register(Skeleton(
    name="variance",
    g=(c_sym - s_sym) ** 2,          # 项读的是 FINAL producer
    finalize=_FINAL_ACC / N,         # variance = 均方偏差
    producer_reducer="sum",
    two_pass=True,
    producer_finalize=_FINAL_M / N,  # 原始和 -> 均值
    intrinsic_params=((N, "ncols"),),
))
```

发出的内核：pass 1 归约 producer 并 finalize；pass 2 在 **producer 恒定的前提下**归约项，
所以**修复是恒等映射，且这是构造性的**——`repair_pair` 根本不被询问，`check_supported`
改问另一个问题（`producer_finalize` 是否存在、是否只依赖原始 producer）。代价是第二遍读；
收益是"两内核 + 一个物化中间量"变成"一个内核、零中间量"。

**和官方 `ReduceToScan` 的区别要说清楚**（免得把两件事混为一谈）。读了源码
（`src/tir/schedule/primitive/reduction.cc`，`RollingUpdate` 的第 3 步）：`ReduceToScan`
是一个 **TVM 调度**原语，把 producer 归约块变成 scan 块，从而让 `HasIncompleteBuffer` 成立、
producer 能被 fuse 到循环下——是为分块 / split-K 服务的。本项目改的是 ATen 图、每行发一个
program，没有循环嵌套可调度、也没有"部分结果"要合并，采用那个原语等于采用 TVM 的执行模型。
**所以这里没有采纳它**，而是用了适配本框架的机制。

也不是"官方能算而我们不能"：他们自己的 benchmark 只对 softmax / attention 用
`rolling_update`（`scripts/neptune_bench/ours/{softmax,attn_plain,attn_gqa,attn_fp8}.py`），
仓库里没有 variance / std 的调度。

### 两个守卫在两个 frame 里是**相反**的

这个不对称必须显式写下来，否则很容易"顺手统一"而引入静默错误：

| | 单趟 frame | 两趟 frame |
|---|---|---|
| `g` 提到"新鲜 producer"符号 | **拒绝**——此处"新鲜"指*流动*值，项会依赖顺序，无意义 | **必需**——此处"新鲜"是唯一的最终值，读它就是本方案的目的 |
| `g` 完全不读 producer | 拒绝——修复退化为恒等 | 拒绝——pass 1 会是死代码 |

同一个符号、两个 frame、相反结论，两者都对。第二行是关键：没有它，那个故意放在框架外的
对照族 `g = exp(c - |c|)` 在两趟 frame 里"完美分离"（比值为 1），会被算成**已覆盖**，而发出的
内核会归约一个谁都不读的 producer。

## 二、覆盖率 20 → 26 / 28，顺带修了两处**声明**问题

新方案落地后，探针里加了一条两趟尝试（在一趟全部失败之后才试——一趟更好，这是退路不是选项）。
第一次跑出来是 **27/28**，比预期多了一个。查下去发现两个问题**都在测量侧**：

1. **探针以前不走 `check_supported`**，直接调发射器。而两个"项不读 producer"的守卫恰好住在
   `check_supported` 里，于是两个本该被拒的对照族被算成已覆盖。修法与"用真发射器而不是
   重新实现一份"同一条原则：**从真实入口进**（`emit_kernel_source` 就是先 `check_supported`）。
2. **`p-norm (p=3)` 家族声明了符号 `p`**，而它自己的注释写着 "p is a compile-time constant"。
   探针于是把 `p` 当**运行时标量参数**声明，符号指数的分离 `(c/s)^p / (c/s')^p` 简化不成
   `(s'/s)^p`（实测简化后仍是 `exp(-p*(log(c/s) - log(c/s')))`，仍含 `c`）→ 被拒。
   改成具体的 `3` 后立刻导出 `s**3/s2**3`。**19 → 20**。

另外把 `two producers, one term` 这个家族名改了。它声明的是 `c * rp`，而 `rp` 在文件顶部就是
`s2 as rp` —— 也就是 `sum(c * mean(c))`，一个单 producer 的算子，两趟方案**本来就该**服务它。
旧名字断言了一件表达式没说的话，旧注释（"needs MultiProducerRepair"）指的是真正的双 producer
情形，而那个情形在这个**单 producer 的数据模型里根本无法声明**。已改名
`term reads the final producer` 并移出 "outside" 组。

最终：**26 / 28（92.9%）**，剩 2 个：`prod + ratio`（mul consumer 需要携带计数）和
`g = exp(c - |c|)`（对照族）。

## 三、`std` 的 `tl.sqrt(ncols)`：一个规则之外的空指针

`variance` 在 GPU 上一片干净（相对误差 ~1e-7），`std` 直接**编译失败**：

```
error: 'math.sqrt' op operand #0 must be floating-point-like, but got 'i32'
        tl.store(out_ptr + row, (tl.sqrt(acc)) * ((1.0 / tl.sqrt(ncols))))
```

原因不在 `std` 的声明里，也不在模板里：sympy 把 `sqrt(acc / N)` **存成** `sqrt(acc) / sqrt(N)`，
于是**整型**形状值 `ncols` 落到了平方根下面。修法是只给"被取根的 intrinsic"做浮点绑定
（`_intrinsics_needing_float`）：`std` 里 `N` 绑定为 `(1.0 * ncols)`，而 `variance` /
`rmsnorm_max` 的 `1.0 / ncols` 不需要转换，**8 个既有内核的代码逐行不变**。

同时补了一条纯 CPU 回归测试，直接盯住发出的源码（因为出错的位置在声明里看不见）。

## 四、退出码审计：本轮真正重要的发现

起因很小：我在验证循环里用 `exit=$?` 判定成败。给 m16/m17 加断言的同一次，我顺手检查了
老脚本——**m10 / m12 / m13 / m14 / m15 全都没有 `raise SystemExit`**，所以**永远退出 0**。

修完之后，问题立刻自己冒出来：

| 脚本 | 修之前 | 真实情况（修完后暴露） |
|---|---|---|
| `gpu_m10` | 打印 `449 passed, 44 failed` 却 exit 0 | 44 个失败全是 `std` 那个 `sqrt(int)`；修掉后 493 项 |
| `gpu_m12` | 打印 VERDICT，不设退出码 | 已补 |
| `gpu_m13` | exit 0 | **在第 [1] 节就 `KeyError('tiny')` 崩了，第 [4] 节的 AOTI 证据从来没跑出来** |
| `gpu_m14` | exit 0 | 只有 GPU 可用 + 3 处识别断言，`cubins/mentions` 只打印 |
| `gpu_m15` | exit 0 | 只有 GPU 可用，融合/`mentions`/`rel_err` 只打印 |
| `gpu_m16`/`m17` | — | 本来就设了退出码 |

顺带还查出 m10 里另外两处：

* **`gen load` 硬编码成 `1`**。对每个单趟内核它恰好是 1，所以从没人发现；两趟内核是 **2**，
  于是这张表会印出一个内核并不具备的数字。改成从发出的源码里数。
* **测试数据用 `hash((name, shape))` 播种**。字符串/元组的 `hash` 是**逐进程加盐**的
  （PYTHONHASHSEED），所以这套测试**每次跑的都是不同的随机数据**——这正是那个
  无法复现的 2.4e-3 失败出现一次就消失的原因。改成 `zlib.crc32`，测试从此可复现。

**口径更正**：我此前报告过 "m14/m15 exit 0"。在修之前，这个"证据"只等于"脚本没崩"，
不等于它们打印的那些数字成立。这条更正写在这里，也记在
[design.md](../design.md) 的验证表里。

## 五、两趟方案的性能：**不是普遍更快**，而且这一点必须说清楚

第一次跑 m17 时我断言的是"比分解形式快 1.05x"，结果是 4 个失败：

```
variance(8, 65536)   0.97x
variance(32, 65536)  0.97x
variance(1, 262144)  0.39x
```

改成"报告 + 单趟对照"之后的最终一次测量：

```
         shape   base ms    var ms   var x     l2 ms    l2 x  loads var/l2
     (8, 4096)    0.0347    0.0228   1.52x    0.0229   1.52x      2/1
    (8, 16384)    0.0563    0.0272   2.07x    0.0269   2.09x      2/1
    (8, 65536)    0.0571    0.0586   0.97x    0.0487   1.17x      2/1
   (32, 65536)    0.0551    0.0591   0.93x    0.0489   1.13x      2/1
   (1, 262144)    0.0516    0.1715   0.30x    0.1332   0.39x      2/1
```

`(1, 262144)` 那一行是关键：**单趟内核在那里也只有 0.39x**，所以这个损失属于**框架**
（每行一个 program、串行扫列），不属于本方案。两列的差才是"第二遍"真正的代价。

这不是测量噪声，是机制的直接后果：**两趟要读两遍**，所以只有在"基线那个额外的内核 + 物化
中间量"比"多读一遍"更贵时才可能赢；而**当行数太少时它会输得很惨**，因为本框架是**每行一个
program**（`(1, 262144)` 只有一行，发出的内核用单个 program 串行扫 256 轮，而 Inductor 会用
split reduction 铺满 GPU）。

所以 m17 现在**把单趟内核 `stable_l2` 放在同一张表里**做对照，把"方案的代价"和"框架的代价"
分开——而不是把框架的性质算在方案头上。断言也相应改成两条**说得出口**的命题：

* m10 的 `>1.05x` 只对**单趟**骨架断言（它替换的是"多内核 + 中间量"，这是它真正的主张）；
* 两趟只断言"不病态"（`>0.25x`），性能数字由 m17 报告。

再加一条我自己的测试设计错误：ill-conditioning 那一段最初用固定的 `1e-1` 阈值，实测
`offset=1e6, spread=1e-2` 时相对误差 0.4。但那个 case 里 `spread` 已经**低于 fp32 在该量级下的
一个 ULP**——数据本身就不含这个方差了，任何 fp32 算法都取不出来。现在把这种 case 显式标为
`unrepresentable` 并排除，断言只覆盖 fp32 能表示的 case。这是同一类错误的第三次出现
（前两次：`finalize` 误读、`k<0` 的参考式），所以它作为一条**教训**记下来而不是当作偶发。

## 六、实测（脚本现在会以非零码报告失败）

见 `exp31_two_pass_gpu.txt` 的原始日志。关键行：

最终一次完整验证（每个脚本都带真实退出码，`FAILED_SCRIPTS` 为空）：

```
tests/test_neptune_opt.py   26 passed            (CPU)
tests/test_recognize.py     exit 0               (CPU)
gpu_m10_codegen_tests       TOTAL: 493 passed, 0 failed, 493 checks
gpu_m12_generality          VERDICT: all auto-derived operators correct
gpu_m13_aoti_codegen        TOTAL: 0 failed checks   ← 第 [4] 节第一次真正跑出来
gpu_m14_recognition         exit 0
gpu_m15_integration         TOTAL: 0 failed checks
gpu_m16_affine              TOTAL: 145 passed, 0 failed, 145 checks
gpu_m17_two_pass            TOTAL: 117 passed, 0 failed, 117 checks
```

m13 第 [4] 节（此前从未产出）第一次给出的数字——基线**不含**生成内核的名字，改造后**含 41 次**：

```
stable_l2   baseline: cubins=1  mentions 'stable_l2'  0x  in_ptr0 loads=8
            selected: cubins=1  mentions 'stable_l2' 41x  in_ptr0 loads=0
```

## 七、还剩什么

* `prod + ratio`（`f = x*y`）：需要跨迭代携带计数，两个方案都不提供。
* 长行少行的性能：这是**框架**的性质（每行一个 program），不是本轮方案的；
  要改进需要 split reduction，也就是官方那套循环嵌套级的东西。
