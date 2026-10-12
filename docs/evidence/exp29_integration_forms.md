# 调研：这个模块应该以什么形式接入 AOTI 最简洁？

结论先给：**最简形式是"对 `ExportedProgram` 调一次函数"**，而不是自定义 pass 钩子、也不是包装成 backend。
下面每一条都有源码或实测依据。

---

## 1. 先看 torch 到底提供了哪些"可接入点"

| 候选形式 | 源码位置 | 是否在 AOTAutograd **之前** |
|---|---|---|
| `torch._inductor.config.post_grad_custom_pre_pass` / `post_grad_custom_post_pass` | `_inductor/config.py:304-305` | ❌ 之后（注释原文："pre/post passes are only applied before/after pattern_matcher in **post_grad_passes**"） |
| `joint_custom_pre_pass` / `joint_custom_post_pass` | `_inductor/config.py:311-312` | ❌ 之后（joint graph 是 tracing 的产物） |
| `pre_grad_custom_pass` | `_inductor/config.py:317` | ❌ 之后（注释："pre-grad IR is 1. non-functional, 2. non-normalized"，指的是 Inductor 的 pre-grad，不是 AOTAutograd 之前） |
| `_functorch.config.joint_custom_pass` | `_functorch/config.py:440` | ❌ 之后 |
| `torch.export.export(...)` 的参数 | `export/__init__.py:59` | — 只有 `dynamic_shapes` / `strict` / `preserve_module_call_signature` / `prefer_deferred_runtime_asserts_over_guards`，**没有 pass 参数** |
| `_export/` 里有没有 pre-pass 注册表 | 全目录 grep `custom_pass\|pre_pass\|decomposition_table` | **没有任何结果** |
| `export/passes/` | `export/passes/__init__.py` | 只有一个 `move_to_device_pass` |
| `torch._dynamo` 有没有 pass 注册机制 | 全目录 grep | 没有 |
| `aoti_compile_and_package(...)` 签名 | `_inductor/__init__.py:64` | 只接受 `exported_program: ExportedProgram` |

**关键事实**：`aoti_compile_and_package` 只吃一个 `ExportedProgram`，
而 `torch.export` 既没有 pass 参数、也没有 pass 注册表。
所以"接入"本质上只能是**"以某种方式产出那个 ExportedProgram"**，不存在可以注册进去的钩子。

---

## 2. 为什么 Inductor 的钩子用不了（实测，不是推理）

我写了一个适配器（钩子收到的是 `torch.fx.Graph`，不是 GraphModule；`CustomGraphPassCallable
= Callable[[torch.fx.Graph], None]`，见 `_inductor/custom_graph_pass.py:125`），
在 `post_grad_custom_pre_pass` 里做同样的改写：

```
post_grad_custom_pre_pass  hook called=1  [('rewritten', 'stable_l2')]  cubins=0
```

**改写成功了**（`('rewritten', 'stable_l2')`），但产物 **`cubins=0`** ——
生成的内核根本没进 codegen。

原因：`triton_op` 是在 **AOTAutograd 的 functionalization 期间**被分解成 Triton HOP 的
（`FunctionalTensorMode` 上的 `functional_decomp`）。Inductor 的 post-grad 阶段在它**之后**，
图里塞进去的只是一个普通自定义算子 → Inductor 走 fallback → 运行时调 Python 实现。

**所以那 5 个 `*_custom_pass` 钩子一个都不能用**，无论它们看起来多像官方推荐的位置。

---

## 3. 实测：哪种形式真的能把内核送进产物

| 形式 | 产物里生成内核出现次数 | cubins |
|---|---|---|
| 模块在 **tracing 时**就调用该算子 | **41** | 1 |
| `export` → 改 `ep.graph_module` → AOTI（**推荐**） | **41**（连跑 4 次都是 41） | 1 |
| `make_fx` → 改 → `export` → AOTI（当前 `optimize`） | **41** | 1 |
| 在 Inductor 的 `post_grad_custom_pre_pass` 里改 | 1（且 cubins=**0**） | 0 |

推荐的形式之所以有效：**AOTI 内部会再跑一次 `aot_autograd`**（对 ExportedProgram 做
functionalization），而恰恰是这一步创建 HOP。所以"改完再交给 AOTI"就足够了——
**不需要重新 `export`**，也不需要 `make_fx`。

### 顺带纠正我自己两个错误

1. 我此前断言"改写必须发生在 export 之前，导出后改图是 0x"，
   并把这个结论写进了 README。**这个结论是错的。**
   重测（连跑 4 次）表明"导出后改 `ep.graph_module`"稳定给出 41x。
   当时的 0x 来自**另一件事**：旧版 `rewrite` 把**图的输出**直接换成算子调用，
   而新版只替换算子自己的区域（region root），输出端还留着一个 `squeeze`。
   `op 直接当图输出` → 0x（测到 3 次）；`op 是内部节点` → 41x（测到 5 次）。
   我还没能解释这个机制，但推荐的形式不依赖它。
2. `torch.ops.neptune.<op>` 拿到的是 **OpOverloadPacket**，而 FX / Inductor 图需要
   **OpOverload**。Inductor 会直接报 `neptune.stable_l2 is not an OpOverload`。
   已改为返回 `.default` 那个重载。这条对"能不能在 Inductor 里用"是硬约束。

---

## 4. 四种形式的人机工效对比

| 形式 | 用户要写什么 | 需要改模型 | 要求 | 实测可用 |
|---|---|---|---|---|
| **A. 对 ExportedProgram 调一次** | `ep = export(m, args)` → `neptune_opt.apply(ep)` | 否 | 无 | ✅ 41x ×4 |
| B. 合并成一个 helper | `ep = neptune_opt.export(m, args)` | 否 | 无 | ✅（A 的包装） |
| C. 改模块（当前 `optimize`） | `opt, rep = optimize(m, args)` → `export(opt)` | 否 | 需要 example inputs + 额外一次 eager trace | ✅ 41x |
| D. Inductor 自定义 pass | 设一个 config | 否 | 无 | ❌ cubins=0 |
| E. `torch.compile` backend | `backend=neptune_opt.backend` | 否 | 无 | ❌ 内核不进 codegen |
| F. 融合算子库 | 把 `x/sqrt(mean(x²))` 改成 `neptune_opt.rmsnorm_max(x)` | **是** | 无 | ✅（等价于"tracing 时调用"，41x） |

**A 比 C 更简单**，而且不只是"少写一行"：

- C 需要用户提供 **example inputs**，还要额外做一次 `make_fx` eager trace——
  那次 trace 会把数据相关的控制流**固化**下来（真模型里 `if seq_len > X` 这类分支会被烙死），
  与被 `torch.export` 正常追踪出来的图可能不同。
- A 直接作用在用户**已经拿到的**那个 `ExportedProgram` 上，语义完全继承用户的 export 流程。
- A 天然兼容 `dynamic_shapes`、`strict` 等 export 参数。

**F 是零风险的兜底**：不改写、不识别，用户显式调用。当识别失败时这是唯一的路。
两者不冲突——识别成功就用 A，失败就让用户用 F。

---

## 5. 建议的最终形态

```
neptune_opt.apply(ep) -> Report          # 就地改写 ExportedProgram 的图，主入口
neptune_opt.export(model, args, **kw)    # export + apply 的合并便利函数
neptune_opt.rmsnorm_max / stable_l2 / ...   # 显式算子库（识别失败时的兜底）
neptune_opt.backend                      # 保留但保持封存（实测无效）
```

用户侧代码从 3 行变成插入 1 行：

```python
ep = torch.export.export(model, args)      # 用户原有的 AOTI 流程
report = neptune_opt.apply(ep)             # ← 只加这一行
print(report.summary())
torch._inductor.aoti_compile_and_package(ep, package_path="m.pt2")
```

并且要在 README 里**撤回**"必须在 export 之前"这条（第 3 节说明），
改成"改写 `ExportedProgram` 的图即可；AOTI 会在内部重新跑 AOTAutograd，
那一步才会创建 Triton HOP"。

---

## 6. 还没查清的

1. **为什么"算子直接当图输出"会掉到 0x。** 测到 3 次 0x、5 次 41x，差异稳定地跟着
   "替换的是输出还是内部节点"。如果是 Inductor 对输出位置的 extern kernel 走了
   另一条路，那是个值得记下来的坑；但我没有时间隔离到最小复现。
2. **F 形式的算子库还没做**——目前 `op_handle` 可用，但没有面向用户的
   `neptune_opt.<op>(x)` 命名空间。
3. **A 形式还需要把 `apply` 实现出来**（`rewrite` 已经能工作，只差一个接收
   ExportedProgram 的薄封装 + 报告），以及把 `optimize` 改成基于 A 实现（可选）。
