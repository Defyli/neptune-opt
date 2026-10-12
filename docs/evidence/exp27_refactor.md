# 代码结构重整：把两个巨型文件拆成包

## 之前

```
neptune_codegen/__init__.py   933 行   <- 符号 + 骨架规格 + 代数 + 发射器 + 模块库 + 模板 +
                                           编译 + torch 接入 + 参考实现，全在一个文件
neptune_recognize.py          665 行   <- 解释器 + 参数求解 + 匹配 + 改写，全在一个文件
```

## 之后

```
neptune_codegen/
  symbols.py      60   sympy 符号（代数写在其上）
  skeleton.py    124   Skeleton 规格 + registry
  algebra.py     182   条件 (3)/(4) 的可执行形式：推导修复与项
  emitters.py    223   每一段 Triton 文本（派生而非模板）
  library.py     109   已注册算子（每个一行声明）
  template.py    133   渲染内核 + 编译成真正的 Triton 函数
  torch_op.py     97   暴露给 torch.compile / AOTInductor
  references.py   79   校验用的参考实现
  __init__.py          只做再导出（保留原公开 API）

neptune_recognize/
  interp.py      234   符号解释 FX 图 -> 代数谈论的那对表达式
  params.py       92   从图里反推骨架的标量参数
  match.py       279   这是哪个骨架？靠比较表达式决定
  rewrite.py      78   只替换算子自己的区域
  __init__.py          只做再导出
```

## 做法：AST 逐字抽取，不手抄

用 `ast` 取出每个顶层定义的行区间，**原样**搬进新文件，再按职责装配。
手抄 1600 行必然会引入笔误，这样能保证语义一字未改。

## 公开 API 回归

拆分前后对比 `dir(module)`：

```
CG 丢失：Callable, TranslationError, _Q, dataclasses, importlib, os, sp, sys, tempfile, to_triton
NR 丢失：CG, c_sym, dataclasses, sp, vanishes
```

绝大多数是**意外泄漏到公开命名空间的标准库/sympy 名字**，没人用（已 grep 全部脚本核对）。
其中 `TranslationError` / `to_triton` / `_Q` 是原文件**有意**再导出的，所以补回去了。
其余不带下划线的泄漏名不再导出——这是改善，不是回归。

## 拆分过程中暴露的真 bug

1. **`from ... import *` 会跳过下划线开头的名字。**
   `symbols.py` 里的 `_FINAL_ACC` / `_FINAL_M` 等都以 `_` 开头，于是
   `from neptune_codegen.symbols import *` 拿不到它们，`library.py` 一导入就
   `NameError: name '_FINAL_M' is not defined`。
   修法：给 `symbols.py` 显式 `__all__`。
2. **`template.py` 漏了一个 import**（`check_supported`）。生成路径在真实运行时才炸，
   m10 的 395 项测试立刻抓到 `NameError: name 'check_supported' is not defined`。
   修完后加了一项**公开 API 冒烟测试**（对每个已注册骨架调用每一个公开发射器），
   正是这类错误的廉价防网。

## 顺带修正一个测量方法错误

m15 的 [B] 段用 `os.environ["TORCHINDUCTOR_CACHE_DIR"] = ...` 切目录，
但 **Inductor 在 import 时就把这个环境变量读进 config 了**，之后改环境变量毫无作用——
所以那段其实一直在搜索**共享的**缓存目录，里面有 [A] 段留下的产物，
于是"未融合"的算子也报出 258 次命中。

这正是我一直在警告的那类假信号（之前"273 次命中"也是同一个原因）。
现在该段不再依赖缓存搜索：直接说明 [B] 路径**已实测无效**、只验证数值，
并把这个测量陷阱写进注释保留。

## 最终回归

```
tests/test_neptune_opt.py   10 passed, 0 failed   (纯 CPU，无需 GPU/Triton)
recognize_tests.py           9 passed, 0 failed
gpu_m10_codegen_tests.py   395 passed, 0 failed
gpu_m14_recognition.py       8/8 operators, 0 pattern rows
gpu_m15_integration.py      [A] 3/3 41x verified  [B] 无效（已封存）  [C] 通过  [D] 2/2 嵌入融合 verified
                            verified=True 计数 = 5
```
