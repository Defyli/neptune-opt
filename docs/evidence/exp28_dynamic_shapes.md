# 动态形状：把"行长度"变成内核已有的运行时量

## 问题

`rmsnorm_max` 的骨架是 `producer_expr = c**2 / N`，而 `N` 是**行长度**。
我原来把 N 当成**标量参数、烘成源码字面量**，由识别器从输入形状里读出来。

于是符号形状下：`_last_dim_of` 拿不到整数 → 退化成自由符号 `$N` → 求解 `N = $N`
不是数值 → 被拒（实测报 `param-unsolved [N]`）。而且即便静态形状，烘字面量也意味着
**产物是形状特化的**：为 4096 列编的内核，拿去算 1000 列会静默算错。

## 修法：把行长度声明成"内核已经有的量"

内核签名本来就收到 `ncols`（以及 `rows`、`stride_row`）。所以新增了一个概念：

```
Skeleton.intrinsic_params = ((sympy_symbol, intrinsic_name), ...)
INTRINSICS = ("rows", "ncols", "stride_row")
```

`rmsnorm_max` 现在声明 `intrinsic_params=((N, "ncols"),)`，`params` 里只剩 `eps`。
发射时 `N` 直接渲染成 `ncols`：

```
m_new = m + tl.sum(tl.where(mask, (v * v) * ((1.0 / (ncols))), 0.0), axis=0)
```

识别侧对应地把 `.mean()` 读成 `sum / NCOLS_SYM`（一个规范符号，而不是从形状里读的数），
再把骨架的 intrinsic 参数替换成同一个符号。于是**静态与符号形状走完全同一条路径**，
识别器再也不用把 SymInt 具体化（原来 `int(shape[-1])` 正是崩溃点）。

## 实测：一个内核服务多个宽度

```
[E] shape independence: ONE generated kernel, several row lengths
    width= 4096  rel_err=1.572e-07  OK
    width= 1000  rel_err=1.713e-07  OK
    width=  333  rel_err=1.782e-07  OK
    width=    7  rel_err=2.024e-07  OK
```

旧设计下 1000/333/7 都会算错（除以烘死的 4096）。

## 实测：真·符号形状被识别

测试用 `torch.export` + `dynamic_shapes` 产生符号行长度，并**断言它确实是 SymInt**
（否则测试会退化成重测静态情形，毫无意义）：

```
exported placeholder shapes: [(8, s21)]        <- 符号行长度
recognize -> <rmsnorm_max params={'eps': 1e-06}>
```

用 `torch.export` 而不是符号 `make_fx`，是因为后者在 torch 2.11 上抛
`Expected Proxy, got [Proxy(arg0_1)]`（2.14 可以）。换路线后在**两个版本上都通过**。

## 顺带关掉一个真实隐患：归约维度从来没被检查过

解释器把 `sum`/`amax` 一律读成"行归约"，但**从来没有检查过 dim 参数**。
也就是说 `t.sum(dim=0)`（按列归约）会被误读成行归约，融合成一个算别的东西的内核——
正是我最担心的那类静默错误。现在 `_reduces_last_dim_only` 强制：只接受归约最后一维
（`dim=-1` 或 `dim=rank-1`），其余的整个图直接拒绝。新增测试覆盖。

## 回归

```
tests/test_neptune_opt.py   13 passed, 0 failed   (本地 2.14 与 GPU 2.11 均通过)
recognize_tests.py           9 passed, 0 failed
gpu_m10_codegen_tests.py   395 passed, 0 failed
gpu_m14_recognition.py       8/8 operators, 0 pattern rows
gpu_m15_integration.py      [A] 3/3 · [B] 无效(已封存) · [C] 通过 · [D] 2/2 · [E] 4/4 宽度
```

注：m10 的性能断言在共享机器上会抖动——第一次跑出现 2 个 1.03x（阈值 1.05），
立刻重跑就是 395/395。这两项是计时噪声，不是回归。
