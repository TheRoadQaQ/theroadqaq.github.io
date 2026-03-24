---
layout: post
title: REINFORCE 算法探究
date: 2026-03-20 12:00:00
description: 
tags: RL Learning
categories: Thoughts
img: 
---

# REINFORCE

REINFORCE 算法优化的目标是最大期望return，即

$$
\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[R(\tau) \right] = \sum_{\tau}{\pi_{\theta}(\tau) R(\tau)}

$$


对这个目标求导数得到

$$
\nabla_{\theta}\mathcal{L}(\theta) = \sum_{\tau}{R(\tau)\nabla_{\theta}\pi_{\theta}(\tau)}
$$

这个式子没错，但它不能作为损失函数，**无法便用采样来计算**

RL里拿不到所有轨迹 $\tau$，只能从当前策略采样一些轨迹。只有采样出来的轨迹，但是这个式子要的是所有的轨迹求和，现实中根本做不到。实际优化是，我们只能从 $\pi_{\theta}$ 中采样轨迹 $\tau$，因此最好写成期望的形式，得到采样的结果后直接用部分轨迹计算梯度的期望
	
代入重要的转换公式

$$
\nabla_{\theta}\pi_{\theta}(\tau) = \pi_{\theta}(\tau) \nabla_{\theta}[\log \pi_{\theta}(\tau)]
$$

得到

$$
\nabla_{\theta}\mathcal{L}(\theta) = \sum_{\tau}{R(\tau)} \pi_{\theta}(\tau) \nabla_{\theta}[log \pi_{\theta}(\tau)]
$$

再返回得到期望

$$
\nabla_{\theta}\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} R(\tau) \nabla_{\theta}[\log \pi_{\theta}(\tau)]
$$

这就是轨迹级别的 surrogate 函数了，虽然和目标不完全一致，但是导数是一致的，所以叫做 surrogate 函数

而轨迹概率可分解成state/action级别：

$$
\pi_{\theta}(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_{\theta}(a_t \mid s_t) p(s_{t+1} \mid s_t,a_t)
$$

其中，$p(s_0)$ 是初始状态分布, $\pi_{\theta}(a_t \mid s_t)$ 是策略网络，$p(s_{t+1} \mid s_t,a_t)$ 是环境转移概率。同时由于环境转移概率对模型参数不可导

因此，梯度为：

$$
\nabla_{\theta}\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ R(\tau) \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \right]
$$

这个公式就是 REINFORCE 算法 的基础。为了优化目标函数，我们从当前策略 $\pi_\theta$ 采样多条轨迹 $\tau$，对每条轨迹计算其回报 $R(\tau)$，然后用这些样本来构造梯度的蒙特卡洛估计：

$$
\nabla_{\theta}\mathcal{L}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} R(\tau^{(i)}) \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t^{(i)} \mid s_t^{(i)})
$$

也就是说，每条轨迹的回报 $R(\tau)$ 会作为权重，去加权对应轨迹中每一步动作的对数概率梯度，从而更新策略参数。直观上，高回报的轨迹会提高其对应动作的概率，低回报的轨迹则会抑制这些动作的概率。

#### 对 $R(\tau)$ 的进一步优化

同时可以进一步优化，考虑单独的每一个state的转移的目标梯度

$$
\mathbb{E}_{\tau \sim \pi_{\theta}}\left[ R(\tau) \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \right]
$$

$R(\tau)$ 是从 $\pi_{\theta}$ 中采样的一整条轨迹的return，而 $\log \pi_{\theta}(a_t \mid s_t)$ 只是一个state的转移概率，两者明显不对等，我们需要把 $R(\tau)$ 进一步细化。考虑

$$
R(\tau) \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) = R(\tau_{<t}) \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) + R(\tau_{\ge t}) \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)
$$

其中

$$
\tau_{<t}=(s_0,a_0,\dots,s_{t-1},a_{t-1},s_t)
$$

$$
\tau_{\ge t}=(a_t,s_{t+1},a_{t+1},\dots)
$$

前者可以直接被处理：

$$
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
R_{\tau<t}\nabla_\theta \log \pi_\theta(a_t \mid s_t)
\right] = \mathbb{E}_{\tau_{<t}\sim\pi_\theta}
\left[
\mathbb{E}_{\tau_{\ge t}\sim \pi_\theta(\cdot\mid \tau_{<t})}
\left[
R_{\tau<t}\nabla_\theta \log \pi_\theta(a_t \mid s_t)
\right]
\right] \\
= \mathbb{E}_{\tau_{<t}\sim\pi_\theta}\left[R_{\tau<t}
\left[
\mathbb{E}_{\tau_{\ge t}\sim \pi_\theta(\cdot\mid \tau_{<t})}
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
\right]\right]
$$

只看

$$
\mathbb{E}_{\tau_{\ge t}\sim \pi_\theta(\cdot\mid \tau_{<t})}
\nabla_\theta \log \pi_\theta(a_t \mid s_t) \\
$$

在条件 $\tau_{<t}$ 下，随机性只来自 $a_t$ 及其之后, 先只看 $s_t$ 到 $a_t$的转移：

$$
\sum_{a_t}
\pi_\theta(a_t \mid s_t)\,
\nabla_\theta \log \pi_\theta(a_t \mid s_t) = \sum_{a_t} \nabla_\theta \pi_\theta(a_t \mid s_t) = \nabla_\theta \sum_{a_t}  \pi_\theta(a_t \mid s_t) = \nabla_\theta 1 = 0
$$

所以

$$
\nabla_{\theta}\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[ R(\tau) \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \right] = 
\mathbb{E}_{\tau \sim \pi_{\theta}}\sum_{t=0}^{T-1}\left[ R(\tau_{\ge t}) \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \right]
$$

#### 从公式到代码

我们从原始优化目标得到了surrogate损失

原始目标是最大化采样轨迹的期望回报，并不显式包含 $\log$。

之所以在实现中使用 $\log \pi_\theta(a_t \mid s_t)$，是因为通过

$$
\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta
$$

可以把原目标的梯度改写成一个可用采样无偏估计的形式。

因此代码中的 policy loss 是一个 surrogate loss：它本身不等于原始目标，但它的期望梯度等于我们想要的 policy gradient。代码实现如下：


```python
def reinforce_surrogate_loss(log_prob, returns):
    """
    log_prob: [batch_size, seq_len]，已采样动作的 log π(a_t \mid s_t)
    returns:  [batch_size, seq_len]，对应时刻的 G_t
    mask: [batch_size, seq_len]，对每个state action的mask
    """
    loss = -((returns.detach() * log_prob) * mask).sum() / mask.sum()
    return loss
```