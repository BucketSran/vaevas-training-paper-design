# tools/ — 工具栈深度学习笔记

> 这一层放**具体工具**的学习材料。和 SFT/GRPO **方法**类的概念笔记（在 `sft/`, `grpo/`）分开。

## 为什么单独放一层

工具会换（明年可能 verl 被新框架取代），方法论不太会换。把工具和方法分开:
- 工具笔记可以被替换/淘汰，不影响方法论笔记
- 方法论笔记不依赖具体工具的当下 API

## 当前文档

| 文件 | 工具 | 学完做什么 |
|---|---|---|
| [`llamafactory.md`](./llamafactory.md) | LLaMA-Factory SFT 框架 | **Phase 2 SFT 生产训练**（基于你远端已搭的真实环境）|
| [`vllm.md`](./vllm.md) | vLLM 推理引擎 | Phase 4 跑 eval 时直接用，Phase 3 GRPO 时知道底层在做什么 |
| [`verl.md`](./verl.md) | verl RL 训练框架 | Phase 3 GRPO 生产训练时用 |

未来可能加的:
- `llamafactory.md` — 如果 SFT 切到 LLaMA-Factory
- `trl.md` — 如果对 trl 内部行为有困惑
- `deepspeed.md` — 如果训练遇到 ZeRO 配置问题
- `wandb.md` — 实验日志系统

## 阅读顺序建议

按学习路径 B（当前路径）:

```
现在 (SFT 学习阶段)
  ├─ 先扫一眼 vllm.md §1-§4 (30 分钟，知道 vLLM 是什么)
  └─ 不需要 verl.md

GRPO 开始前
  ├─ vllm.md 全文 (90 分钟)
  └─ verl.md 全文 (2-3 小时)

GRPO 实际训练时
  ├─ 配 vLLM rollout 参数 → 翻 vllm.md §7-§8
  └─ 写 reward function → 翻 verl.md §6
```

## 与其他文档的关系

- [`../REFERENCES.md` 的"框架选型景观"章节](../REFERENCES.md#框架选型景观-framework-landscape) 是**高层选型决策**
- 本目录是**单个工具的深度学习**
- 高层 → "我们选 vLLM 不选 X，理由 Y"
- 本目录 → "vLLM 怎么工作，怎么用，常见坑是什么"
