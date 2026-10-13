# 拿真实的 `torch.nn` 模块来测：这个框架到底能服务什么

覆盖率 "26 / 28" 有一个我一直没正面处理的问题：**那 28 个族是我自己列的**。它衡量的是代数，
不是世界。所以这一步换一个更难的问题：

> 用户真正写下的代码（`nn.LayerNorm`、`nn.Softmax`、`torch.logsumexp`、`logits / T`），
> 这个模块能服务多少？不能的那些，**为什么**不能？

答案有好有坏，两份都记在这里。

## 一、实测表（`experiments/real_modules.py`）

```
    module                   as exported                  after core decompositions
    ------------------------------------------------------------------------------
    nn.LayerNorm             needs decomposition: layer_norm   needs decomposition: native_layer_norm
    nn.RMSNorm               needs decomposition: rms_norm      ONE reduction -- out of frame
    F.layer_norm             needs decomposition: layer_norm   needs decomposition: native_layer_norm
    nn.Softmax               needs decomposition: softmax      needs decomposition: _softmax
    nn.LogSoftmax            needs decomposition: log_softmax  needs decomposition: _log_softmax
    torch.logsumexp          needs decomposition: logsumexp    logsumexp
    log_softmax by hand      needs decomposition: logsumexp    logsumexp
    attention lse (t/8)      needs decomposition: logsumexp    lse_scaled
    RMSNorm + row max        rmsnorm_max                      rmsnorm_max
    stable L2 norm           stable_l2                        stable_l2
```

## 二、发现 1：真实模块必须先做 core-aten 分解

第一列几乎全是 `needs decomposition`：**真实模块导出成"一个高层 aten 算子"，不是
producer/consumer 对**。`torch.logsumexp` 是 `aten.logsumexp.default`，`nn.Softmax` 是
`aten.softmax.int`。识别器要找的"两个互相依赖的归约"，在原始导出图里根本不存在。

修法不是加模式，而是把图**分解**——`ExportedProgram.run_decompositions()`，也就是
AOTAutograd 本来要交给 Inductor 的那份图：

* `neptune_opt.export(model, args)` 现在默认 `decompose=True`；
* 已有 `ExportedProgram` 的话先 `ep = ep.run_decompositions()`；
* 并且当 `apply()` 拒绝、而分解之后本可以识别时，**报告会直说**：

```
skipped : needs-decomposition: would recognise 'logsumexp' after the core-aten decompositions:
          call ep = ep.run_decompositions() first, or use neptune_opt.export(..., decompose=True)
```

这一条是这轮最实用的修正。我之前所有证据都是拿"已经手工分解过的图"跑出来的，也就是说：
**当时的能力在一个真实用户手上几乎全都用不到**。

## 三、发现 2（更重要）：有些算子在框架之外，是结构性的

`nn.RMSNorm` 分解后是一类完全不同的东西：`mean(x²)` → `rsqrt` → `mul`。它**只有一个归约**，
消费端是逐元素运算。

而这个框架融合的是**两个互相依赖的归约**（producer 与 consumer）。一个孤立的归约喂给逐元素
运算，**无论再加多少代数都进不来**——不是模式缺失，是框架的适用域。同一句话适用于
`nn.LayerNorm` 和 softmax 的归一化。

所以解释器现在给出的是这个判定，而不是"没匹配上"：

```
only 1 reduction(s) found: this frame fuses a PRODUCER and a CONSUMER reduction, so a single
reduction with elementwise work around it (RMSNorm, LayerNorm, softmax's normalisation) is
outside it
```

**这条比覆盖率数字更worth 先知道**：它说明继续按"族数"投资是错的方向。框架真正赢的地方是那
个交集——同一行上两个互相依赖的归约。Inductor 把这种图编成**两个内核**（它无法融合两个相互
依赖的归约），而本框架编成一个。`rmsnorm_max`、`stable_l2`、`decay_renorm`、log-sum-exp 折
都属于这一类。

## 四、这个测试直接换来一个**真实**算子：`lse_scaled`

`attention lse (t/8)` 一开始是拒绝的。它的形状是 `logsumexp(t / T)`——**每个 attention
实现都这么写**，而我的注册表里没有带缩放的折。

加它需要**一条声明**（`neptune_codegen/library.py`）：

```python
register(Skeleton(
    name="lse_scaled",
    g=sp.exp(c_sym / _PARAM_T - s_sym),        # 生产者不用写
    finalize=_FINAL_M + sp.log(_FINAL_ACC + _PARAM_TINY),
    producer="signed", is_log=True,
    params=((_PARAM_T, "T"), (_PARAM_TINY, "tiny")),
))
```

而且 **T 是从图里解出来的**，不是配置进去的：

```
logsumexp(t/8)   -> lse_scaled  inferred={'T': 8.0, 'tiny': 0.0}
logsumexp(t/3)   -> lse_scaled  inferred={'T': 3.0, 'tiny': 0.0}
plain logsumexp  -> logsumexp   inferred={'tiny': 0.0}
```

**生产者是推导出来的**：指数折满足 `g = exp(phi(c) - s)`，于是 `phi = log(g) + s = c/T`。
这就是"加一个算子 = 一条声明、零 codegen"在真实算子上的一次验证（这一族是靠跑真实代码找到的，
不是靠扩自己的清单）。

## 五、为了能看到上面这些判定，解释器还补了三处

如实记录，因为每一处都是"看不见真实图"造成的：

1. **`torch.logsumexp` 的 NaN 守卫**。它分解成
   `where(eq(abs(amax), inf), 0.0, amax)`——把无穷大的 max 换成 0，避免 `exp(t-inf)` 出 nan。
   这跟参考分解顺带加的 `clamp_min` 属于同一类**数值守卫**，对有限输入是恒等，所以按恒等放行。
   守卫是**按图结构识别**的（`where` + `eq` + `abs`），而不是把 `abs` 加进忽略列表——
   `abs` 本身是真算子，通用地忽略它会静默改变函数含义。
   第一次实现没生效，原因是那个 `0.0` 是 **`scalar_tensor(0.0)` 节点**而非常量字面量，
   守卫表因此为空，拒绝理由还看起来毫不相干。
2. **参数物化的节点**（`empty` / `full_like` / `wrap_with_set_grad_enabled`）。导出模块用它们
   物化 weight/bias/eps。原来会在 `aten.empty.memory_format` 上拒绝，把一个"算子只有一个归约"
   的结构性判定，显示成一条关于内存分配的抱怨。
3. **`rsqrt` / `reciprocal`**：`rsqrt(x) = 1/sqrt(x)` 是真算子（RMSNorm/LayerNorm 就是这么写
   的）。理解它不会让那些算子变得可融合，但能让解释器走到上面那条结构性判定。

## 六、GPU 端到端：**这次测量本身出过两次错**，两个数字都记下来

做法：对每个"在框架内"的算子，把**两条路都打包**成 `.pt2`（不融合 / 融合），加载后比较数值与
延迟，并且**核对基线包里确实没有生成内核的名字**（`base` 列，实测 0）——否则就是拿同一个东西
在比。最终一次（256 × 16384，启动流水化）：

```
    module                   fused as   mentions  base    rel_err  plain ms  fused ms      x
    torch.logsumexp          logsumexp         41     0   9.35e-08    0.0281    0.0183   1.54x
    log_softmax by hand      logsumexp         41     0   2.38e-07    0.0311    0.0261   1.19x
    attention lse (t/8)      lse_scaled        41     0   9.82e-08    0.0286    0.0194   1.47x
    RMSNorm + row max        rmsnorm_max       41     0   4.45e-07    0.0277    0.0182   1.52x
    stable L2 norm           stable_l2         41     0   9.48e-07    0.0288    0.0184   1.57x
```

**端到端 1.19–1.57x**，数值误差 ~1e-7，基线包里没有生成内核（0 次）。

### 但第一版测量给出的是"0.89–1.05x"

第一次跑（64 × 4096，逐次调用）得到的是 0.89x / 1.03x / 1.05x —— 看起来像"优化基本没用、还偶尔
更慢"。那是**测量错**，不是结果：

* 单次调用 ~0.023 ms，和"Python 调度 + 一次启动"同量级，**内核根本不是瓶颈**，所以测的是外壳；
* 修正：换更大的形状（256 × 16384），并把 `reps` 次调用**连发**（启动流水化）再除以 reps。

两个数字都保留了，因为它们回答的是两个真问题：**当内核是瓶颈时赢 1.2–1.6x；当行很短时（内核
不是瓶颈）赢面在噪声里**。这对使用者是有用的一句话，而它只能靠两次测量都做出来才知道。

### 另外两个测量 bug（都是我的）

* **AOTI 包是静态形状的**。我拿 64 × 4096 编译、却用 256 × 16384 计时，结果是
  `CUDA error: an illegal memory access`，并且**污染了后续所有 case 的 context**（后面每个都报
  同一个错）。改成全程用同一个形状。
* **CUDA graph 捕获在这个场景下不可用**：AOTI 加载出来的包每次调用都会分配输出，捕获失败
  （`operation failed due to a previous error`）。已放弃并改用"连发 + 除 reps"的写法，原因写在
  `_batch_ms` 的 docstring 里，免得下次有人再试一遍。

## 七、这对项目方向意味着什么

* 继续按"族数"扩覆盖是**错的方向**：剩下那 2 族一个需要携带计数，另一个是故意保留的对照。
* 真正该问的是"**同一行上两个依赖归约**"这类算子在真实模型里有多少、值多少钱。已知的有：
  log-sum-exp 折（softmax/交叉熵/attention 分母，含温度缩放）、`rmsnorm_max` 这类归一化+归约的
  组合、以及 `stable_l2` 这类"先取尺度再统计"的算子。
* 反过来，**单归约 + 逐元素**（RMSNorm/LayerNorm/softmax 归一化本身）永远不在框架内。要在那里
  获益需要的是另一件事：把归约结果与输入在同一个内核里结合（也就是官方那种循环嵌套级的能力），
  那是另一种架构，不是更多代数。
