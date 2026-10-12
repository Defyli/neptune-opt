# Milestone 15 — neptune_opt 的两个接入点

## 侦察结论（决定了方案）

torch 暴露的**全部** pass 钩子都在 AOTAutograd 之后：
`post_grad_custom_pre_pass` / `post_grad_custom_post_pass` / `joint_custom_pre_pass` /
`joint_custom_post_pass` / `pre_grad_custom_pass`（`_inductor/config.py:304-317`），
配置注释原话："pre/post passes are only applied before/after pattern_matcher in
post_grad_passes"。

而生成的内核只有在 **tracing 期间**才会被登记成 Triton HOP。所以那些钩子一个都用不了
（这也解释了 `_custom_pass` 路线此前一直是 0x）。

唯一能生效的位置是 AOTAutograd **之前**，那里有两个真实公开 API：
1. `optimize(model, args)` 改写后再 `torch.export` → AOTI
2. 自定义 `torch.compile` backend（`compile_fx` 内部会调 `aot_autograd`，所以先改写
   Dynamo 图就把算子放进了 tracing）

## 实测结果

[A] `optimize` → `torch.export` → AOTI: **3/3 算子通过**
```
stable_l2    fused=stable_l2                    mentions=41x  cubins=1  rel_err=2.433e-07  verified=True
logsumexp    fused=logsumexp                    mentions=41x  cubins=1  rel_err=1.079e-07  verified=True
rmsnorm_max  fused=rmsnorm_max(N=4096, eps=1e-06) mentions=41x cubins=1  rel_err=2.972e-07  verified=True
```
`rmsnorm_max` 的 N 与 eps 是从图里推断的（`.mean()` 形式）。

[B] `torch.compile(backend=...)`: **图改写成功、数值正确，但内核没进 codegen**
```
stable_l2    rel_err=1.216e-07  fused=stable_l2   hits in compiled/generated code=0
rmsnorm_max  rel_err=1.315e-07  fused=rmsnorm_max hits in compiled/generated code=0
```
所以这条路径的融合**是无效的**（算子会 fallback 到 Python 调用，比不融合更慢），
已用 `experimental=True` 开关封存，默认拒绝调用。

早先我一度以为它有效，依据是"Inductor cache 里出现内核名 273 次"——**那是假信号**：
Inductor 生成的注释会引用原图的节点名和 stack trace。分开统计后
"compiled/generated code" 命中为 0。这个教训写进了 m15 的注释里。

[C] 安全门：算子嵌在大图里时拒绝融合，且结果与原模型一致
```
fused=False  refused=True  result unchanged=True
```

## 本轮修掉的真 bug

1. **`rewrite` 会把图的输出整个替换掉。** 若输出是 `add(算子结果, 残差)`，它会静默
   删掉残差。现在加了区域门：算子尾部（finalize）只依赖两个归约结果、不含数据符号 `c`，
   所以从 consumer 向前走、只要表达式不含 `c` 就还在算子内——一旦结果与真实数据结合
   （残差加、matmul），该节点含 `c`，于是被排除、整张图被拒绝。
2. **`node.meta.get("val") or node.meta.get("example_value")`** —— `or` 会对 Tensor 取真值，
   报 "Boolean value of Tensor with more than one value is ambiguous"。
   这是我自己引入的；失败门把它变成了安全拒绝而不是崩溃，所以才被发现。
3. **区域门在 Dynamo 图上永远拒绝。** Dynamo 的 `output` 节点参数是**元组**，
   而 `make_fx` 直接放 tensor 节点。原来取 `out.args[0]` 当节点用，导致走向从未开始，
   同一个算子 make_fx 通过、torch.compile 全部被误判为"嵌在大图里"。
4. **`env` 里存的是图自己的符号 `$r0/$r1`**，而区域门用规范化后的 `S_RESULT` 判定，
   于是任何提到生产端的节点都被排除（实测 `mul` 的表达式是 `$r0*sqrt($r1)`，
   整个尾部被排除，连完整的整图算子都被误拒）。

## 最终回归

```
tests/test_neptune_opt.py   7 passed, 0 failed   (纯 CPU，无需 GPU/Triton)
recognize_tests.py          9/9
gpu_m10_codegen_tests.py    395 passed, 0 failed
gpu_m15_integration.py      [A] 3/3 通过  [B] 已查清无效并封存  [C] 安全门通过
```


---

# 追加：真正做区域提取（匹配 finalize）

## 为什么这不只是"功能"，而是先修一个静默错误

原来的区域门只检查尾部**不含数据符号 `c`**，并不检查它等于声明的 finalize。实测：

```
l2 (纯算子)         matched=stable_l2  region_root=squeeze
l2*2.0 (尾部多一步)  matched=stable_l2  region_root=mul_1     <- 外层那个 ×2！
```

也就是说 `rewrite` 会把 `l2*2.0` 的**外层乘法**当成区域根替换掉，**静默丢掉 ×2**。
`optimize` 的数值自检能拦住，但直接调 `NR.rewrite` 就会产出错代码。

## 做法

1. **把读出的常数声明成参数。** log 家族尾部是 `m + log(acc + tiny)`，`tiny` 声明为参数
   （默认 0）。硬编码 `1e-38` 永远无法等于图里的 `m + log(acc)`；声明后识别能解出 `tiny=0`。
   顺带给 `Skeleton` 加了 `defaults`，让"这个参数的默认值不是 1.0"能被表达。
2. **匹配 finalize 定位区域根。** 从 consumer 往后的每个节点，它的符号表达式若等于
   `finalize(acc, m)`（参数可推断），就是候选根；取拓扑序**最后**一个。
3. **只替换区域根**，再 DCE。区域外的计算原样保留。
4. **输出秩（keepdim）也是算子的一部分。** `(rows,)` 与 `(rows,1)` 和模型其余部分广播的
   方式不同，所以秩由区域根的形状决定，读不到形状就**拒绝**而不是猜。

## 实测（GPU）

```
[D] 嵌在大计算里的算子
    l2*2.0   fused=stable_l2  mentions=41x  cubins=2  rel_err=3.011e-07  verified=True
    l2+t     fused=stable_l2  mentions=41x  cubins=2  rel_err=3.743e-07  verified=True
```

×2 和 +t 都活着，且数值与自检都过。原来的 [C] 段从"拒绝"变成了"正确融合"。

识别回归：9/9，参数推断结果：
`logsumexp(tiny=0)`、`l2_biased(alpha=1.5, tiny=0)`、`softcap_lse(beta=0.5, tiny=0)`、
`decay_renorm(N=2)`、`rmsnorm_max(N=64, eps=1e-06)`。

## 顺带修掉的

参数求解统一成"**一次解一个，先符号后数值**"。原来数值回退只在"恰好剩一个未知数"时才走，
于是 `softcap` 的 `beta`（`tanh` 没有符号反函数）永远解不出来。现在 `_solve_for` 对每个
出现在方程里的未知数都先试符号、再试数值，两个场景用同一条路径。

## 最终回归

```
tests/test_neptune_opt.py   9 passed, 0 failed
recognize_tests.py          9 passed, 0 failed
gpu_m10_codegen_tests.py    395 passed, 0 failed
gpu_m14_recognition.py      8/8 operators, 0 pattern rows
gpu_m15_integration.py      [A] 3/3  [B] 无效（已封存）  [C] 正确融合  [D] 2/2 嵌入式融合
```
