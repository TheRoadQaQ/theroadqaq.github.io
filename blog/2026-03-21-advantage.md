---
layout: post
title: Advantage定义探究
date: 2026-03-20 12:00:00
description: 
tags: learning
categories: Thoughts
img: 
---
# Baseline

回顾一下 REINFORCE 的公式

$$
\nabla_{\theta}\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\sum_{t=0}^{T-1}\left[ R(\tau_{\ge t}) \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \right]
$$

上面的Reinforce公式虽然无偏，但是方差较大。我们用

$$
R(\tau_{\ge t}) \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)
$$

更新 $t$ 步的概率，但是 $R(\tau_{\ge t})$的波动往往很大，尤其是长轨迹、稀疏奖励的情况（RLVR/Agentic RL）

因此可以不直接用return $R(\tau_{\ge t})$，而是用 return 减去一个参考值，也就是baseline $b(s_t)$，得到

$$
\nabla_{\theta}\mathcal{L}(\theta)
=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\sum_{t=0}^{T-1}
\left[
\big(R(\tau_{\ge t}) - b(s_t)\big)\nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)
\right]
$$

这个式子看起来像是“偷偷改了目标”，但其实**并没有引入偏差**，因为 baseline 对梯度期望的贡献为 0。

#### 为什么减去 baseline 仍然是无偏的

我们只看多出来的那一项：

$$
\mathbb{E}_{\tau \sim \pi_\theta}
\sum_{t=0}^{T-1}
\left[
b(s_t)\nabla_\theta \log \pi_\theta(a_t\mid s_t)
\right]
$$

对某个固定时刻 $t$ 来看：

$$
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
b(s_t)\nabla_\theta \log \pi_\theta(a_t\mid s_t)
\right]
$$

由于 $b(s_t)$ 只依赖于当前状态，不依赖于当前采样动作 $a_t$，因此可以像前面一样，把条件期望拆开：

$$
\mathbb{E}_{\tau_{<t}\sim \pi_\theta}
\left[
\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}
\left[
b(s_t)\nabla_\theta \log \pi_\theta(a_t\mid s_t)
\right]
\right]
$$

把 $b(s_t)$ 提出来：

$$
\mathbb{E}_{\tau_{<t}\sim \pi_\theta}
\left[
b(s_t)
\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}
\left[
\nabla_\theta \log \pi_\theta(a_t\mid s_t)
\right]
\right]
$$

而内部这一项前面已经证明过了：

$$
\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}
\left[
\nabla_\theta \log \pi_\theta(a_t\mid s_t)
\right]
=
\sum_{a_t}\pi_\theta(a_t\mid s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)
=0
$$

所以整体就是 0。于是：

$$
\mathbb{E}_{\tau \sim \pi_{\theta}}
\sum_{t=0}^{T-1}
\left[
b(s_t)\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)
\right]
=0
$$

因此我们可以合法地写成：

$$
\nabla_{\theta}\mathcal{L}(\theta)
=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\sum_{t=0}^{T-1}
\left[
\big(R(\tau_{\ge t}) - b(s_t)\big)\nabla_{\theta}\log \pi_{\theta}(a_t \mid s_t)
\right]
$$

也就是说，**减去任意一个与动作无关的 baseline，不会改变梯度的期望，只会影响方差。**

#### 直觉理解

$R(\tau_{\ge t})$ 可以理解为“这个动作之后最终拿到了多少回报”。

而 $b(s_t)$ 可以理解为“在这个状态下，通常来说我本来就能拿到多少回报”。

那么两者的差值：

$$
R(\tau_{\ge t}) - b(s_t)
$$

其实就是：**这个动作的结果，比这个状态下的平均水平好多少/差多少**。

- 如果高于 baseline，这个量是正的，就提高该动作的概率
- 如果低于 baseline，这个量是负的，就降低该动作的概率
- 如果差不多，那梯度就接近 0，说明这个动作没提供什么额外信息

这比直接拿原始 return 去乘 log-prob 更合理，因为模型真正需要知道的不是“这条轨迹值多少钱”，而是“这个动作相对平均水平到底好不好”。

# Advantage

前面引入 baseline 时，我们把 policy gradient 写成了

$$
\nabla_{\theta}\mathcal{L}(\theta)
=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\sum_{t=0}^{T-1}
\left[
\big(R(\tau_{\ge t}) - b(s_t)\big)\nabla_{\theta}\log \pi_{\theta}(a_t \mid s_t)
\right]
$$

这里最自然的问题就是：这个 $b(s_t)$ 到底该取什么？为什么大家最后都会把这个差值写成 advantage？

要讲清楚这一点，先把几个容易混淆的量分开。

#### 从时刻 $t$ 开始的实际回报：$G_t$

**对于一条已经采样出来的具体轨迹**，在时刻 $t$ 之后，我们可以把未来拿到的累计回报记作

$$
G_t = \sum_{t'=t}^{T-1} r_{t'}
$$

它表示的是：

> 在这一条实际采样出来的轨迹上，从当前时刻开始，后面真正拿到了多少回报。

所以 $G_t$ 是一个**样本值**，它对应的是“这一次实际发生了什么”，也可以直接把它理解成从时刻 $t$ 开始的 future return。

---

#### 状态价值：$V(s_t)$

和 $G_t$ 不同，$V(s_t)$ 表示的不是“这一次实际拿到了多少”，而是：

> 从状态 $s_t$ 出发，后面继续按照当前策略走，平均来说能拿到多少回报。

所以它刻画的是一个**平均水平**。

同一个状态 $s_t$，后续可能会采样出很多不同轨迹，因此也会有很多不同的 $G_t$；而 $V(s_t)$ 则是这些可能结果的平均。

可以把它理解成：

- $G_t$：这次真实发挥的结果
- $V(s_t)$：在这个状态下的平均发挥水平

这两个量的区别非常重要。  
**$G_t$ 是单次样本，$V(s_t)$ 是期望。**

---

#### 动作价值：$Q(s_t,a_t)$

如果再进一步，不只是固定状态 $s_t$，还固定当前采取的动作 $a_t$，那么我们关心的就是：

> 在状态 $s_t$ 下先做动作 $a_t$，之后再继续按当前策略往后走，平均能拿到多少回报？

这就是动作价值 $Q(s_t,a_t)$) 的含义。

它和 $V(s_t)$ 的区别在于：

- $V(s_t)$ 只固定了“现在在哪个状态”
- $Q(s_t,a_t)$ 不仅固定了状态，还固定了“这一步具体做了哪个动作”

因此，$Q(s_t,a_t)$ 比 $V(s_t)$ 更具体一些。

有了上面两个量之后，Advantage 就很好理解了：

$$
A(s_t,a_t)=Q(s_t,a_t)-V(s_t)
$$

它表示的是：

> 在状态 $s_t$ 下，采取动作 $a_t$ 这件事，相比这个状态下的平均水平，到底更好还是更差。

所以 advantage 本质上衡量的是“**这个动作相对平均水平的额外收益**”。

- 如果 $A(s_t,a_t) > 0$，说明这个动作比平均水平更好
- 如果 $A(s_t,a_t) < 0$，说明这个动作比平均水平更差
- 如果 $A(s_t,a_t) = 0$，说明这个动作差不多就是平均水平

这也是为什么它叫 advantage：它描述的是这个动作到底有没有“优势”。

---

#### 为什么 baseline 常常取 $V(s_t)$

前面我们说过，理论上只要 baseline 不依赖当前动作，它就不会改变 policy gradient 的期望，只会影响方差。

如果我们希望减掉的是“这个状态本身就有的那部分公共回报”，那么最合理的选择就是状态价值：

$$
b(s_t) = V(s_t)
$$

因为 $V(s_t)$ 正好表示：

> 不管这一步具体选什么动作，在这个状态下，平均来说本来就能拿到多少回报。

于是

$$
Q(s_t,a_t)-V(s_t)
$$

就恰好变成了：

> 这个动作的结果，相对这个状态下平均水平的偏离。

这正是我们更新策略时最想知道的信息。  
我们并不关心“这条轨迹总共值多少钱”，而更关心：

> 这一步动作到底有没有比平均水平更好一点。

---

#### 为什么实现里常写成 $G_t - V(s_t)$

看到这里，一个很自然的问题是：

Advantage 的定义是

$$
A(s_t,a_t)=Q(s_t,a_t)-V(s_t)
$$

但是 **$Q(s_t,a_t)$ 很难直接知道**，所以我们用采样完一次轨迹 $G_t$ 去近似 $Q(s_t,a_t)$

$$
A_t \approx G_t - V(s_t)
$$

这样一来，policy gradient 就可以写成

$$
\nabla_{\theta}\mathcal{L}(\theta)
=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\sum_{t=0}^{T-1}
\left[
A_t \nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)
\right]
$$

其中

$$
A_t \approx G_t - V(s_t)
$$

代替真正的 advantage，得到

$$
\nabla_{\theta}\mathcal{L}(\theta)
\approx
\mathbb{E}_{\tau \sim \pi_{\theta}}
\sum_{t=0}^{T-1}
\left[
\big(G_t - V(s_t)\big)\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)
\right]
$$

这比直接用 $G_t$ 做权重更合理，因为它减掉了“这个状态本来就该有的平均收益”，只保留了动作相对平均水平的那部分差异，因此方差会小很多。

---

# Actor-Critic

到这里也就能看清楚 $G_t$ 和 $V(s_t)$ 分别从哪来。

在 Monte Carlo REINFORCE 里

- $G_t$ 来自采样好的完整轨迹，直接从 reward 累加得到
- $V(s_t)$ 可以是手工设计的 baseline，也可以是额外训练出来的模型，也就是 value model

于是 advantage 的估计就是

$$
A_t \approx G_t - V(s_t)
$$

在 Actor-Critic 里

- actor 负责输出策略 $\pi_\theta(a_t\mid s_t)$
- critic 负责学习状态价值 $V_\phi(s_t)$

这时通常用 critic 的预测值替代真实的 $V(s_t)$，写成

$$
A_t \approx G_t - V_\phi(s_t)
$$

所以可以把 actor-critic 看成是在做两件事：

1. 用采样轨迹算出实际回报 $G_t$
2. 用 critic 去预测平均回报 $V_\phi(s_t)$

两者一减，就得到 advantage 的估计。

因此：

1. baseline 最自然的选择是 $V(s_t)$
2. advantage 的定义是 $Q(s_t,a_t)-V(s_t)$
3. 在实际采样训练时，因为 $Q(s_t,a_t)$ 难以直接获得，所以通常用 $G_t$ 去估计它
4. 最终实现里

$$
A_t \approx G_t - V(s_t)
$$

在 Monte Carlo policy gradient 中最常见的估计形式。







