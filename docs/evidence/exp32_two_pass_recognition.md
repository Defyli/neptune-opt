# 让两趟方案真的能被用上：识别（以及顺带查出的三个 bug）

上一轮做出了两趟方案，但**它当时只存在于"显式算子库"里**：`neptune_opt.variance(x)` 能用，
而用户真正走的自动路径 `neptune_opt.apply(ep)` 对 `x.var(-1)` 一律拒绝。先量一下这个缺口：

```
decomposed var (.mean)   -> REFUSED: no registered skeleton matches (1 candidate pair(s))
decomposed var (sum/N)   -> REFUSED: no registered skeleton matches (1 candidate pair(s))
decomposed std           -> REFUSED: no registered skeleton matches (1 candidate pair(s))
torch.var                -> REFUSED: graph is not a producer/consumer reduction pair
```

一个只有显式 API 能用的能力，对这个项目要解决的问题（AOTI 自动生成的推理代码）基本没有价值。
所以这一步的目标是**让它可达**。

## 一、关键：把 `1/N` 写在图写出它的地方

我原来的声明是"先算原始和，再除以 N"：

```python
g=(c_sym - s_sym) ** 2,          # 项读的是 FINAL producer
finalize=_FINAL_ACC / _PARAM_N,  # 读出来再除 N
producer_finalize=_FINAL_M / _PARAM_N,   # 原始和 -> 均值
```

代数上完全正确，但**永远匹配不上**。原因是识别器的入口：`torch.mean` 是"对 ncols 求和"，
所以解释器把它看到的均值记成**操作数** `c / N`：

```python
reductions.append(("sum", v / NCOLS_SYM, sym, node))     # neptune_recognize/interp.py
```

而匹配器拿这个操作数和骨架的 `producer_expr` 比较（`_producer_pattern`）。于是"均值"要么以
`c / N` 的形式出现在 `producer_expr` 里，要么根本对不上——**我为此新增的那个 `producer_finalize`
字段，恰恰是让算子不可识别的原因**。

改法是把每个 `1/N` 放在图放置它的位置：

```python
g=(c_sym - s_sym) ** 2 / _PARAM_N,   # torch.mean() 实际求和的项
finalize=_FINAL_ACC,                 # 恒等：累加器就是算子的值
producer_expr=c_sym / _PARAM_N,      # 均值，按图的写法
```

于是 `producer_finalize` 被**删掉**了：它既冗余（`producer_expr` 早就能表达"均值"——
`rmsnorm_max` 一直用 `producer_expr=c**2/N` 描述"平方均值"），又阻断识别。发出的内核随之变成：

```
m_new = m + tl.sum(tl.where(mask, v * ((1.0 / (ncols))), 0.0), axis=0)      # pass 1 = 均值
den = m
acc   = acc + tl.sum(tl.where(mask, ((1.0/ncols)) * (v-den)*(v-den)), 0.0)  # pass 2
tl.store(out_ptr + row, acc)
```

删掉字段之后，**识别不需要任何匹配器改动**就通过了。这是一次"少即是多"的修正：我上一轮多加了
一个概念，它没有帮上忙，反而是唯一挡住识别的东西。

## 二、顺带查出的三个 bug

### 1. 匹配器的作用域（潜伏 bug，正好被这次改动踩亮）

`_match_candidate` 里 `solved` 只在 `if unknown:` 分支内绑定，却在分支外被使用：

```python
if unknown:
    solved: dict = {}          # 只在这里存在
    ...
if vanishes(...):
    root, tail_solved = _find_region_root(gm, env, sk, cand, solved, known_args)   # UnboundLocalError
```

**没有任何非 intrinsic 标量参数的 sum/sum 骨架**（`std`/`variance` 正是如此）会在这里崩。它一直
是休眠的，因为此前没有这样一个骨架能匹配上任何图——而这次改动正好让第一个匹配上了，于是立刻
炸出来。修法是把 `solved` 提到分支外；这同时还消掉了第二个隐患：旧写法下**上一轮循环的 `solved`
会泄漏到下一轮**，尾部检查可能看到别的算子的参数值。

### 2. 我自己把守卫弄丢了，覆盖探针立刻抓到

重写 `check_supported` 里 two_pass 那段（为了删 `producer_finalize`）时，我把"**项必须读 producer**"
的守卫一起删掉了。下一次跑覆盖探针：

```
non-invertible producer   outside   NOT_INVERT   YES   acc = acc + tl.sum(tl.where(mask, tl.exp(tl.
```

那个故意放在框架外的对照族 `g = exp(c - |c|)` 变成了"**已覆盖**"——因为它在两趟 frame 里平凡分离
（项根本不提 producer）。这正是探针存在的意义。守卫已恢复，并且补了一条**单元测试**盯住它，
下次不必依赖"某次跑到覆盖探针"才被发现。

### 3. 盯着一行已被删除的字面量的检查

`m17` 的形状检查（以及一个 CPU 测试）检查的是 `m = m * ((1.0 / (ncols)))` 这一行——它随
`producer_finalize` 一起消失了，于是 m17 报了 2 个 FAIL。改成按**位置**检查（pass 1 结束于 pass 2
开始之前），而不是钉字面量。

## 三、`aten.var.dim`：单算子展开，以及一个必须拒绝的默认值

`x.var(-1, unbiased=False)` 导出成**一个** `aten.var.dim` 节点，不是 producer/consumer 对。
所以解释器把它**展开**成代数需要的那一对（`mean(c)`，然后 `mean((c - mean)^2)`），产生的符号
和分解写法完全一致。

`unbiased` 是**读出来的，不是假定的**，这是正确性边界而不是细节：

| 图 | 是否融合 |
|---|---|
| `x.var(-1, unbiased=False)` | 是 —— `/N` |
| `x.var(-1, correction=0)` | 是 —— `/N` |
| `x.var(-1)`（**默认**） | **拒绝** —— `/(N-1)` |
| `x.std(-1)`（**默认**） | **拒绝** —— `/(N-1)` |

两种形式差一个 `N/(N-1)` 因子。把"没写这个 flag"当成"biased"，就会把**大多数人写方差的写法**
静默融进一个算出不同数字的内核里。四条都在 `tests/test_recognize.py` 里钉住。

## 四、我自己的流程错误（第三次，值得记下来）

在改 `docs/design.md` 和 `experiments/gpu_m15_integration.py` 时，我写的补丁形如：

```python
s = open(p).read()
s = s.replace(old1, new1)          # 或者干脆没赋值
open(p, "w").write(s)              # 写第一次
...
open(p, "w").write(s.replace(old2, new2))   # 用同一份 s 写第二次 -> 覆盖第一次
```

第二次写用的是**同一份内存里的 `s`**，于是第一次的编辑被静默覆盖。这已经发生三次
（m15 的 `check` helper 定义丢了一次、design.md 的两趟小节丢了两次），每次都表现为"脚本报告成功、
文件里却没有那段"。教训很简单：**一次读、一次累积、一次写**；改完立刻 grep 回验。

## 五、实测

识别（CPU，无 GPU）：

```
TOTAL: 19 passed, 0 failed        ← 原 9 项，新增 variance/std/var 算子形式/batch-norm 内嵌/两个负例
```

CPU 门禁与覆盖：

```
tests/test_neptune_opt.py   27 passed, 0 failed   ← 含"项必须读 producer"的新守卫测试
tools/coverage_codegen.py   26 / 28 families, 2 refusals (both labelled)
```

端到端（RTX 5090 D v2，`neptune_opt.apply(ep)` → AOTI，产物里数生成内核的名字）：

```
--- end to end from a plain `.var()` / `.std()` module
    var            fused=variance  mentions= 41x cubins=1 abs_err=2.384e-07 verified=True
    std            fused=std       mentions=121x cubins=1 abs_err=1.192e-07 verified=True
    decomposed     fused=variance  mentions= 41x cubins=1 abs_err=1.192e-07 verified=True
    batchnorm_like fused=variance  mentions= 41x cubins=2 abs_err=4.768e-07 verified=True
```

基线从不包含生成内核的名字（实测 0x），所以 `41x` / `121x` 就是"产物里确实带了被代数要求的内核"。
`batchnorm_like` 那一行说明**区域闸门**也在工作：方差是更大计算的一个中间量，被单独替换掉，
外面的标准化原样保留，数值不变。

完整日志见 `exp32_recognition_gpu.txt`（被验证的 revision：`emitters=b090e037...`、
`library=2f4de1e6...`）。全部 7 个 GPU 脚本 + 2 个 CPU 测试脚本真实退出码：

```
tests/test_neptune_opt.py   27 passed            (CPU)
tests/test_recognize.py     19 passed            (CPU)
gpu_m10_codegen_tests       TOTAL: 493 passed, 0 failed, 493 checks
gpu_m12_generality          VERDICT: all auto-derived operators correct
gpu_m13_aoti_codegen        TOTAL: 0 failed checks
gpu_m14_recognition         exit 0
gpu_m15_integration         TOTAL: 0 failed checks
gpu_m16_affine              TOTAL: 145 passed, 0 failed, 145 checks
gpu_m17_two_pass            TOTAL: 129 passed, 0 failed, 129 checks
FAILED_SCRIPTS: (empty)
```

## 六、还剩什么

* `prod + ratio`（`f = x*y`）：需要跨迭代携带计数，两趟方案也不提供。
* 长行少行的性能：属于**框架**（每行一个 program、串行扫列）。这一轮的性能表里
  `(1, 262144)` 仍然是两趟 0.30x / 单趟 0.39x —— 单趟也输，所以那不是方案的问题，
  要改进需要 split reduction，也就是官方那套循环嵌套级的东西。
