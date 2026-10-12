# 仿射修复：把"反演"换成分离，以及它顺带查出来的三个问题

这一轮的目标是**读官方实现（`uiuc-arc/neptune`，TVM fork）并借鉴**。借鉴到的是一条核心
思路：**repair 不该靠"反演 g"求，而该靠"变量分离"求**。换成这个形式之后，代码本身反而查出了
三个问题——一个静默错误、一个隐藏 bug、一个我自己写错的测试参考。

## 一、原来错在哪：把 Eq.5 当字面意思用

论文把修复写成 `h(t, r, r') = g(r', g_c^{-1}(r, t))`（Eq.5）。读起来就是"把 term producer
在 `c` 上反演，再在新的 producer 值上重新作用一次"。我原来真就照这个做：`sympy.solve` 反演。

官方实现**根本不反演**。它的 `ReduceRepairer::HExprSolver` 走四步：

```
1. 分离变量：g(u…, x…) = gcomb(gu(u…), gx(x…))       # arith.sympy.separate_vars
2. 反演：    gcomb(a, b) = r  ->  a = gcomb^-1(r, b)
3. 构造：    h(r, x, x') = gcomb(gcomb^-1(r, gx(x)), gx(x'))
4. 证明：    h(f(y1,y2), x, x') == f(h(y1), h(y2), x, x')
```

第 1 步才是要点。落到我们的符号上，"分离"就直接是：

```
R = g(s', c) / g(s, c)          # 仅当 c 被消掉才接受
```

**为什么这就够**：累加器里放的是 `Σ_j g(s, c_j)`。把 producer 从 `s` 提升到 `s'`，需要把
**所有已经累加进去的项**同时乘上 `g(s',c_j)/g(s,c_j)`；只有当这个比**不依赖 j**（也就是 `c`
被消掉）时它才是一个"数"。所以分离就是全部义务，对加法 reducer 没有别的要证。

已在 `tests/test_neptune_opt.py::test_multiplicative_repairs_are_unchanged_by_the_affine_generalisation`
里钉成 golden 表：8 个骨架的斜率逐字符相同（`decay_renorm (N+s)/(N+s2)`、`linear_attn s/s2`、
`logsumexp exp(s-s2)`、`rmsnorm_max sqrt(eps+s)/sqrt(eps+s2)`、`stable_l2 s**2/s2**2` …），
`b` 全为 0。换推导**没有改掉任何一个修复**。

## 二、顺带查出的问题 1：`R = 1` 的静默退化

分离形式比反演更宽，宽出来的第一件事就是个坑。分类器里有个**故意放在框架外**的对照族
`g = exp(c - |c|)`。这个 `g` 完全不读 producer `s`，于是 `g(s',c)/g(s,c) = 1`，`c` 完美消掉，
分离**成功**。

而 `R = 1` 在代数上是**合法的**修复（条件 (3) 成立：`g(s,c)·1 = g(s,c) = g(s',c)`，因为 g 与 s
无关）。所以代数没有理由拒绝它——但这条路径会产出一个完全忽略 producer 的内核，而且让
**生成器与分类器公开矛盾**（分类器对它是 `NOT_INVERT`）。

> 处理方式：新增前置条件——`g` 必须提到 `s`，否则拒绝。理由是"这压根不是一个 (f,g,h) 折叠"，
> 而不是"算不出来"。宁可拒绝，也不静默发出一个忽略归约的内核。

## 三、顺带查出的问题 2：仿射修复（也是这轮真正的收益）

把判据写成"比值"而不是"反演"之后，**残余是"因子"还是"平移"一目了然**。当比值仍然含有 `c`
时，剩下的修复形状是仿射的：

```
h(t) = a(s,s')·t + b(s,s')
a = g'_c(s',c) / g'_c(s,c)          # 让两边共享同一个 c-依赖：匹配 c 的导数
b = g(s',c) - a·g(s,c)              # b 不含 c 就等于"没有 c-依赖剩下来"，即验收条件
```

接着**条件 (4) 会在两种 reducer 上给出不同答案**，这一点是决定性的：

| consumer `f` | 条件 (4) 要求 | 结论 |
|---|---|---|
| `x + y` | `a(x+y)+b == (ax+b)+(ay+b)` | **必须 `b = 0`**，偏移非法 |
| `max(x,y)` | `a·max(x,y)+b == max(ax+b, ay+b)` | `a > 0` 时**任意 `b` 都行**（max-同态） |

所以"加法式修复"恰好只在 max consumer 上合法——而工作清单里那两族卡住的 `max + additive shift` /
`max + scaled shift` 正好在这里。代码里对 `b != 0` 且 consumer 不是 max 的情况直接拒绝，理由就是
条件 (4)。

实测（`python tools/coverage_codegen.py`）：

```
total declared families          : 28
  ALGEBRA admits a fold (EXACT)  :  19
  CODEGEN emits a kernel         :  19  (67.9%)
```

**19/19**——凡是代数承认有折叠的族，生成器现在都发得出内核。改动前是 17（89% of EXACT）。

## 四、顺带查出的问题 3：floor 判据用错了代理

`denominator_expr` 原来按 `is_log` 决定要不要把新鲜 producer 值做地板（`tl.maximum(m_new, 1e-30)`）。
这个代理对**现有 8 个骨架全对**，但对**加法折叠是错的**：地板是**除法保护**，而加法折叠从不除，
这时地板不是 1e-30 的舍入问题，而是**实打实的错误**——因为那个值是被当**偏移量**用的。

实测（`tests/test_neptune_opt.py::test_producer_floor_is_applied_only_where_the_source_divides`）：

```
算子 g = c - s（max producer / max consumer），输入行 c = [-5, -1, -3]
参考值（恒为 0）：            0.0
不地板（新）：                0.0
地板（旧）：                -2.0      ← 误差是行自身量级，不是 1e-30
```

改法：按**派生表达式**判断，谁真的除以 producer 谁才地板（`_needs_denominator_floor`，对冻结的
skeleton 做了 memoize）。实测分档：

```
地板：decay_renorm, linear_attn, rmsnorm_max, stable_l2, sumsq_div
不地板：logsumexp, lse_biased, softcap_lse
```

剥掉注释逐行对比，8 个已注册内核的**代码**与改动前完全一致（只有解释性注释改了）。

## 五、顺带查出的问题 4：我自己写错的测试参考

`experiments/gpu_m16_affine.py` 第一次上 GPU：**88 passed / 23 failed**，23 个失败**全是**
`affine_plus k=-0.5`。查下去发现**内核是对的，参考错了**：

```
我用：  out = (1+k)·max(c)              ← 只在 k > 0 时成立
正确：  out = max_j(k·c_j) + max c      ← max_j(k·c_j) 在 k<0 时选的是 MINIMUM
```

本地复算确认（同样输入，两种参考都对内核求差）：

```
k= 2.0:  |kernel-true|=0.000e+00   |kernel-naive|=0.000e+00
k=-0.5:  |kernel-true|=0.000e+00   |kernel-naive|=1.437e+00
k=-3.0:  |kernel-true|=0.000e+00   |kernel-naive|=1.504e+01
```

这正是本项目反复出现的那类错误：**参考照"对算子的读法"写，而不是照定义写**（`finalize` 那次是
3.4 倍误差且测试照样通过）。修掉参考后 m16 = **145/145**，并且把 `k < 0` 也加进了 CPU 侧的数值
replay，两个符号都覆盖。

## 六、GPU 实测（RTX 5090 D v2，torch 2.11.0+cu128，triton 3.6.0）

被验证的正是最终提交的三份源码（md5 一致）：

```
### algebra_md5=8a5ce3f567b4da9718893ab39aef8498
### emitters_md5=520e7c95d2dee83f833cded9fb79282c
### template_md5=38dcac9993bef46629e2197d662f5888
```

| 项目 | 结果 |
|---|---|
| `tests/test_neptune_opt.py`（CPU 门禁，21 项） | 21 passed, 0 failed |
| `tests/test_recognize.py`（识别 8/8） | exit 0 |
| `experiments/gpu_m10_codegen_tests.py`（生成器正确性） | **395 passed, 0 failed** |
| `experiments/gpu_m14_recognition.py`（新算子进 AOTI 产物） | exit 0 |
| `experiments/gpu_m15_integration.py`（端到端 `.pt2`） | exit 0 |
| `experiments/gpu_m16_affine.py`（仿射修复，本轮新增） | **145 passed, 0 failed** |

m16 的关键几行（完整的原始日志见 `exp30_affine_gpu.txt`）：

```
[PASS] affine_plus/slope is 1              -- a = 1
[PASS] affine_plus/bias is s'-s            -- b = -s + s2
[PASS] affine_plus/producer is unfloored
[PASS] affine_shift k=None (16, 4096) all_negative  -- rel_err=0.000e+00
[PASS] one-element rows avoid the bias     -- err=0.000e+00 got=[9.0, -7.5, 0.0]
[PASS] wrong reference is detected         -- good=0.000e+00 bad=6.084e+00
```

m16 之前没有存在的必要：**8 个已注册骨架的 `b` 全是 0**，所以 m10 的 395 项检查里没有一项会碰到
`bias = tl.where(...)` 这一行。仿射路径此前是**零 GPU 覆盖**。

## 七、还剩下什么

覆盖率 19/28 之后，剩下 9 族的原因已经全部具名（`tools/coverage_codegen.py` 末尾就是工作清单）：

| 原因 | 数量 | 族 |
|---|---|---|
| `repair-not-multiplicative` | 6 | one-pass/two-pass variance、Welford、batch-norm stats、std … |
| `f-needs-carried-state` | 1 | prod + ratio |
| `g-ignores-producer` | 1 | 对照族（g 不读 producer） |
| `g-depends-on-fresh-value` | 1 | 对照族（占位符写了 `s'`，本身非良构） |

前两行是**真问题**，而且都是同一个：需要跨迭代携带状态。官方实现的 `ReduceToScan`
（把 producer 归约变成 scan 块）正是为此而设——它不需要修复对 reducer 分配。这已经从"可选项"
变成"唯一剩下的路"，也是唯一能触及 `NEEDS_STATE` 那 6 族的办法。

顺带记两条对比文档的更正：
* 借鉴"变量分离"**没有**解开双 producer 那一族（它的占位声明 `g = c·s'` 提到新鲜 producer 值，
  本身就不是 (producer, term) 对）；真正的双 producer 情形早就由 `producer_expr` +
  `producer_reducer` 表达，而那是另一条轴。
* 收益出在**没人预料到的轴**上（仿射修复）。教训：分离这个**形式**本身比它替换掉的反演更值钱，
  哪怕两者给出完全相同的答案。
