---
layout: post
title: 环境构建是一种核心的 agentic capability
date: 2026-04-03 18:00:00
description: 
tags: research
categories: Research
img: 
---

## 环境构建是一种核心的 agentic capability

SWE 能力是重要的 Agent 能力，训练这个能力需要搭建 SWE 环境，但是在 Scaling 数据的时候，最大的难题是 Docker Build

我们提出可以训练 Agent 来完成这种 Docker Building 任务，增加 SWE 环境和数据的数量，同时训练模型成为好的 Docker Builder 的时候我们可以收集 Docker Building 轨迹，有了轨迹后可以和 SWE 数据一起训练，最终 SWE 和 Docker Building 的效果都变得更好了

很多 SWE agent 不是不会修 bug，而是连环境都搭不起来。真正卡住规模化训练和评测的，往往不是 reasoning，而是 Docker。

我们提出了 **DockSmith**：一个专门做 **Docker 环境构建** 的 agentic builder。

- **环境构建需要进行Scaling，这是训练 SWE Agent的基础**
- **环境构建不该只是前置预处理，它本身也是一种核心 agentic capability**

---

现在往往是从GitHub中收集大量的 SWE 的训练数据，**把一个静态仓库真正变成可执行的 Docker 环境**

但是，在构建 Docker 时其实问题很大：

- 依赖声明不完整
- 系统包缺失
- 编译链条复杂
- 测试入口不清楚
- 环境假设没有写在文档里

由于无法构建正确的 Docker 环境，很多 repository 无法作为合适的 SWE 训练数据，阻止了进一步的 Scaling

**环境构建不是一个杂务，而是整个 SWE data pipeline 的核心瓶颈。**

---

### DockSmith

核心思路：**把 Docker 环境构建本身当成一个可验证、可训练、可迁移的 agentic task。**

<a href="/blog/imgs/docksmith.pdf" target="_blank" rel="noopener noreferrer">
  <img src="/blog/imgs/docksmith.png" alt="docksmith" style="width: 50%; height: auto; display: block; margin: 1em auto;" />
</a>

DockSmith 不是只生成一个 Dockerfile 就结束，而是把环境构建做成一个带反馈闭环的多代理系统。整个 pipeline 包含四类 agent：

1. **Context Retrieval Agent**
   读取仓库中的 manifest、CI 配置、build script、测试入口等线索
2. **Dockerfile Agent**
   生成或修补 Dockerfile
3. **Eval Script Agent**
   生成实际运行环境和测试命令的脚本
4. **Test Analysis Agent**
   执行 build / test，并把原始日志整理成结构化失败信号

这四个 agent 在执行反馈驱动下不断迭代，直到：

- 环境构建成功
- P2P 和 F2P 条件均满足

在这个 Multi-Agent 框架下，我们利用来自真实 GitHub 仓库和真实 PR 的环境构建数据，蒸馏轨迹。在这些轨迹上训练出的模型是一个 **30B-A3B** 的专用 Docker-building 模型。

### 不只是“搭环境”

同时，我们把环境构建重新定义成了一种**可迁移的通用 agent 能力**。

因为一个强环境构建 agent 实际上会被迫学会很多更一般的技能：

- 读仓库结构
- 理解依赖关系
- 决定执行顺序
- 分析日志
- 逐步修复失败
- 避免重复无效尝试

这些能力显然不只对造 Docker 环境有用。

- **environment construction supervision 本身就能训练 agentic behavior**

- **environment construction** 能够 Scaling SWE 类型的训练数据

---

#### 实验结果

<img src="/blog/imgs/docksmith_result1.png" alt="Average Length" style="width: 50%; height: auto; display: block; margin: 1em auto;" />

DockSmith 在 **Multi-Docker-Eval** 手脚架上拿到了开源 SOTA：

- **39.72% Fail-to-Pass**
- **58.28% Commit Rate**

<a href="/blog/imgs/docksmith_result2.pdf" target="_blank" rel="noopener noreferrer">
  <img src="/blog/imgs/docksmith_result2.png" alt="Average Length" style="width: 50%; height: auto; display: block; margin: 1em auto;" />
</a>

Docker 训练还能反哺更一般的 agent 任务。

在把 Docker-building trajectories 和普通 SWE-solving trajectories 混合训练后：

- **SWE-bench Verified** 最多提升 **+2.25**
- **SWE-bench Multilingual** 最多提升 **+2.09**
- **Terminal-Bench 2.0** 最多提升 **+3.37**

而且不同任务的最佳 mixing ratio 还不一样：

- SWE.V / SWE.M 在大约 **1:1** 的 `SWE:Docker` token 比例附近最好
- Terminal-Bench 在 **1:0.5** 时增益最大

这个结果很说明问题：

- Docker-building trajectories 不是只教模型“怎么写 Dockerfile”
- 它更像是在教模型**如何在真实执行反馈里做长链条修复**

---

#### SWE 数据反过来也能帮 Docker Build

论文还做了反方向实验：

- 只用 Docker trajectories 训练
- 对比 `Docker + 0.5 SWE`

结果是加入 SWE 数据后，所有 benchmark 都提升：

- **Docker Building**：`34.43 -> 35.63`
- **SWE.V**：`33.50 -> 47.45`
- **SWE.M**：`18.25 -> 31.00`
- **Terminal**：`7.16 -> 10.11`

这说明两类数据是互补关系：

- Docker 数据强化环境、依赖、执行和恢复
- SWE 数据强化更广义的任务级验证和修复模式

---

**环境构建不是为了开始真正任务之前必须忍受的一段流程，它本身就是真正任务的一部分。**

---

### Links

- [🤗 Model / Data](https://huggingface.co/collections/8sj7df9k8m5x8/docksmith)
- [📄 Paper](https://arxiv.org/pdf/2602.00592)

---

### Citation

```bibtex
@article{zhang2026docksmith,
  title={DockSmith: Scaling Reliable Coding Environments via an Agentic Docker Builder},
  author={Zhang, Jiaran and Ma, Luck and Li, Yanhao and Wan, Fanqi and Qi, Di and Zhao, Xu and Hou, Jieyi and Xie, Zhe and Ren, Mengqiang and Wu, Xin and Huang, Zhewei and Chen, Liangyu and Ma, Yingwei and Han, Qi and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2602.00592},
  year={2026}
}
```
