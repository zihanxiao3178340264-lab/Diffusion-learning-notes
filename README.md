这是我的第一个仓库，用于记录扩散模型学习的笔记

# 概率视角下的生成模型

## 生成模型的概述

给定从真实分布$p(x)$采样的观测数据$x\sim p(x)$，通过参数估计得到一个由$\theta$控制逼近真实分布的$p_\theta(x)$，则称$p_\theta(x)$为**生成模型**。$x$一般认为是来自同一分布的多维向量。

$p(x)$代表数据真实的分布，观测的数据就是从真实分布中采样出来的。现实中真实分布几乎无法得到，绝大部分复杂数据不存在一个可以手写的表达式，且分布可能是非平稳的，如新闻用语、流行图片风格、网络话术等一直在变化，真实分布永远没法定格捕获。

考虑抛硬币的问题，在空气阻力、每次抛硬币的力度角度等的作用下，真实分布远没有正反面等概率$\dfrac{1}{2}$这么简单。但我们可以建立生成模型$p_\theta(x)$，直接简单的建模成$\theta$控制的伯努利分布，再利用参数估计的方法算得参数$\theta$。当然也可以在考虑上述那些作用的情况下，建立更精确的模型。

得到生成模型$p_\theta(x)$之后，就能生成各式各样，包括观测集中没有的数据了，这就是最终目的。



## $p_\theta(x)$的建模难点

关于图像等复杂情景的$p_\theta(x)$的建模肯定远比抛硬币复杂。这时该如何建模呢？一个直接的想法是利用复杂的神经网络去拟合$p_\theta(x)$。但是若强行让神经网络拟合成一些简单的分布，如正态分布等，则会出现严重欠拟合。所以对$p_\theta(x)$的建模时研究中最主要关心的问题。

本文主要介绍**隐变量模型**和**基于能量函数的模型**两种类型。



## 隐变量模型$LVM$($Latent\ Variable\ Models$)

真实分布$p(x)$不好直接分析或逼近，我们的想法是借助贝叶斯公式把它拆成：
$$
p(x)=\int p(x|z)p(z)dz
$$
其中引入的$z$就叫做隐变量，它服从任意指定先验分布$p(z)$。形如这样的模型就叫**隐变量模型**。由于这里隐变量只有$z$，故也叫单隐变量模型。

对于单隐变量模型，只需要使$p_\theta(x|z)$来逼近$p(x|z)$，理论上就可以利用公式$p_\theta(x)=\displaystyle\int p_\theta(x|z)p(z)dz$得到生成模型$p_\theta$。“逼近”的衡量标准通常采用$KL$散度，即最小化$KL[p_\theta(x|z)||p(x|z)]$。然而在很多情景里，这个式子也不容易处理。

我们稍作变化。考虑对联合密度$p(x,z)$逼近，这是由于$p(x,z)=p(x|z)p(z)$、$p_\theta(x,z)=p_\theta(x|z)p(z)$，这就相当于逼近拟合了$p(x|z)$。此时：
$$
\begin{equation}
\begin{split}
&KL(p(x,z)||p_\theta(x,z))\\
&=\iint p(x,z)ln\frac{p(x,z)}{p_\theta(x,z)}dzdx\\
&=\iint p(x)p(z|x)ln\frac{p(x,z)}{p_\theta(x,z)}dzdx\\
&=\int p(x)\left[\int p(z|x)ln\frac{p(x,z)}{p_\theta(x,z)}dz\right]dx\\
&=E_{x\sim p(x)}\left[\int p(z|x)ln\frac{p(x)p(z|x)}{p_\theta(x,z)}dz\right]\\
&=E_{x\sim p(x)}\left[\int p(z|x)\left[lnp(x)+ln\frac{p(z|x)}{p_\theta(x,z)}\right]dz\right]\\
&=E_{x\sim p(x)}\left[lnp(x)\int p(z|x)dz\right]+E_{x\sim p(x)}\left[\int p(z|x)ln\frac{p(z|x)}{p_\theta(x,z)}dz\right]


\end{split}
\end{equation}
$$
由于第一项中，
$$
\begin{equation}
\begin{split}
E_{x\sim p(x)}\left[lnp(x)\int p(z|x)dz\right]&=E_{x\sim p(x)}\left[lnp(x)\right]\\
&=C


\end{split}
\end{equation}
$$
其中$C$是一个常数，所可令：
$$
\begin{equation}
\begin{split}
\mathcal{L}
&=KL(p(x,z)||p_\theta(x,z))-C\\
&=E_{x\sim p(x)}\left[\int p(z|x)ln\frac{p(z|x)}{p_\theta(x,z)}dz\right]\\



\end{split}
\end{equation}
$$
最小化$\mathcal{L}$等价于最小化$KL(p(x,z)||p_\theta(x,z))$，也等价于最大化$-\mathcal{L}$。因此，$-\mathcal{L}$也被叫做**变分下限($ELBO$)**。

接着对$\mathcal{L}$再做一些变化：
$$
\begin{equation}
\begin{split}
\mathcal{L}
&=E_{x\sim p(x)}\left[\int p(z|x)ln\frac{p(z|x)}{p_\theta(x,z)}dz\right]\\
&=E_{x\sim p(x)}\left[\int p(z|x)ln\frac{p(z|x)}{p_\theta(x|z)p(z)}dz\right]\\
&=E_{x\sim p(x)}\left[-\int p(z|x)lnp_\theta(x|z)dz+\int p(z|x)ln\frac{p(z|x)}{p(z)}dz\right]\\
&=E_{x\sim p(x)}\left[E_{z\sim p(z|x)}\left[-lnp_\theta(x|z)\right]+KL(p(z|x)||p(z))\right]
\end{split}
\end{equation}
$$
可以看到我们引入了真实后验$p(z|x)$，表示的是观测到$x$，与之对应的隐变量$z$的真实分布。于是还需要用神经网络拟合$p_\theta(z|x)$，叫做**编码器($Encoder$)**；与之对应，似然函数$p_\theta(x|z)$叫做**解码器($Decoder$)**或**生成器($Generator$)**。



