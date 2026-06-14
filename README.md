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
其中$C$是一个常数，所以可令：
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
可以看到我们引入了真实后验$p(z|x)$，表示的是观测到$x$，与之对应的隐变量$z$的真实分布。于是还需要用神经网络拟合出$p_\theta(z|x)$用来逼近$p(z|x)$，这种神经网络叫做**编码器($Encoder$)**；与之对应，用于拟合出似然函数$p_\theta(x|z)$的神经网络叫做**解码器($Decoder$)**或**生成器($Generator$)**。

对于优化函数$\mathcal{L}$的第一项，$E_{z\sim p(z|x)}\left[-lnp_\theta(x|z)\right]$叫做**重构损失MSE**，$p_\theta(x|z)$是解码器拟合出的似然函数，最小化$-lnp_\theta(x|z)$就是最大化似然函数。对于第二项，$KL(p(z|x)||p(z))$叫做**$KL$正则项**，是希望真实后验分布$p(z|x)$与先验分布$p(z)$尽量接近，这样就能使先验分布采样的$z$对应真实的$x$，进而放入解码器中生成。

接下来就有两个要求：

1、$p_\theta(x|z)$的形式要容易计算出极大似然，且均值要容易得到，方便生成。

2、选择合适的后验分布$p(z|x)$形式，使得能与先验$p(z)$的$KL$散度容易计算。

到此为止，我们把各种单隐变量模型在统一的观点下进行了推导介绍。对于具体的模型需要做不同的假设和分析，这将在下文讨论。

### 自编码器($AE$)

自编码器的思想核心在于要求隐变量$z$和$x$能唯一对应。首先考虑**要求2**，既然能够唯一对应，则可以利用狄拉克函数的筛选特性，将真实后验分布表示为$p(z|x)=\delta(z-C(x))$，其中$C$为真实编码器。这个真实后验分布可以用神经网络拟合：$p_\theta(z|x)=\delta(z-C_\theta(x))$。对于先验分布，我们可以假设服从$p(z)=\dfrac{1}{D},0<z<D$，$D$为正数。至此已经满足了要求2。

再考虑**要求1**，关于似然函数的拟合与上面一样利用狄拉克函数：$p_\theta(x|z)=\delta(x-G_\theta(z))$，其中$G_\theta$是解码器。至此要求1也已经满足。

为了计算$\mathcal{L}$也即（5），我们对刚刚选取的分布做变换。狄拉克取样函数可以表示为正态分布方差趋于0的极限，于是似然函数化为$p_\theta(x|z)=\delta(x-G_\theta(z))=\lim\limits_{\sigma\to0}N(G_\theta(z),\sigma^2)$，后验分布化为$p_\theta(z|x)=\delta(z-C)=\lim\limits_{\hat{\sigma}\to0}N(C(x),\hat{\sigma}^2)$。这样代入损失函数，$\mathcal{L}$的第一项也即**重构损失项**化为：
$$
\begin{equation}
\begin{split}
&E_{x\sim p(x)}\left[E_{z\sim p(z|x)}\left[-lnp_\theta(x|z)\right]\right]\\
&=E_{x\sim p(x)}\left[E_{z\sim p(z|x)}\left[-ln\frac{1}{\sqrt{2\pi}\sigma}e^{\frac{(x-G_\theta(z))^2}{2\sigma^2}}\right]\right]\\
&=E_{x\sim p(x)}\left[E_{z\sim p(z|x)}\left[{\frac{(x-G_\theta(z))^2}{2\sigma^2}}\right]\right]-ln\frac{1}{\sqrt{2\pi}\sigma}\\
&=\frac{1}{2\sigma^2}E_{x\sim p(x)}\left[E_{z\sim p(z|x)}\left[{(x-G_\theta(z))^2{}}\right]\right]-ln\frac{1}{\sqrt{2\pi}\sigma}\\
\end{split}
\end{equation}
$$
对于$KL$正则项：
$$
\begin{equation}
\begin{split}
&KL(p(z|x)||p(z))\\
&=\int\delta(z-C(x))ln\frac{\delta(z-C(x))}{p(z)}dz\\
&=\int\delta(z-C(x))ln{\delta(z-C(x))}dz-\int\delta(z-C(x))lnp(z)dz\\
&=\int\delta(z-C(x))ln{\delta(z-C(x))}dz-lnp(C(x))\\
&=\int\delta(z-C(x))ln{\delta(z-C(x))}dz+lnD\\
\end{split}
\end{equation}
$$
对于前一项有，
$$
\begin{equation}
\begin{split}
&\int\delta(z-C(x))ln{\delta(z-C(x))}dz\\
&=\int\frac{1}{\sqrt{2\pi}\hat{\sigma}}e^{-\frac{(z-C(x))^2}{2\hat{\sigma}^2}}ln\frac{1}{\sqrt{2\pi}\hat{\sigma}}e^{-\frac{(z-C(x))^2}{2\hat{\sigma}^2}}dz\\
&=\int-\frac{(z-C(x))^2}{2\hat{\sigma}^2}\frac{1}{\sqrt{2\pi}\hat{\sigma}}e^{-\frac{(z-C(x))^2}{2\hat{\sigma}^2}}dz+\\
&ln\frac{1}{\sqrt{2\pi}\hat{\sigma}}\int\frac{1}{\sqrt{2\pi}\hat{\sigma}}e^{-\frac{(z-C(x))^2}{2\hat{\sigma}^2}}dz\\
&=-\frac{1}{2}-ln\sqrt{2\pi}\hat{\sigma}

\end{split}
\end{equation}
$$
所以$KL(p(z|x)||p(z))=-\frac{1}{2}-ln\sqrt{2\pi}\hat{\sigma}+lnD$，这是一个与$x、\theta$无关的常量，在优化过程中可以不用考虑。

至此，自编码器$AE$的目标函数就推导出来了，即为重构损失项MSE$\dfrac{1}{2\sigma^2}E_{x\sim p(x)}\left[E_{z\sim p(z|x)}\left[{(x-G_\theta(z))^2{}}\right]\right]-ln\dfrac{1}{\sqrt{2\pi}\sigma}$。

### 变分自编码器$(VAE)$

与自编码器AE假设每个$x$唯一对应一个$z$不同，变分自编码器VAE的思想在于：假设每个$x$对应一个分布（这个分布是隐变量$z$的后验分布$p(z|x)$），再由这个分布采样出一个$z$，从而重构出$\hat{x}$，实现还原$x$。

为了生成的时候方便采样，我们设先验分布为标准正态分布$p(z)=N(0,1)$，则真实后验分布应该为$\mu=C(x)$的正态分布$p(z|x)=N(\mu,\sigma^2)$，含义就是每个$x$对应一个正态分布后验。对于这个后验$p(z|x)$，可以直接用神经网络拟合函数$C_\theta(x)$，进而拟合出$p_\theta(z|x)=N(C_\theta(x),\sigma^2)$。损失函数$\mathcal{L}$的第一项和$AE$完全一致，即MSE重构损失项。关键差异在第二项$KL$正则项上：
$$
\begin{equation}
\begin{split}
&KL(p(z|x)||p(z))\\
&=\int N(\mu,\sigma^2)ln\frac{N(\mu,\sigma^2)}{N(0,1)}dz\\
&=\int\frac{1}{\sqrt{2\pi}{\sigma}}e^{-\frac{(z-\mu)^2}{2{\sigma}^2}}ln\frac{\frac{1}{\sqrt{2\pi}{\sigma}}e^{-\frac{(z-\mu)^2}{2{\sigma}^2}}}{\frac{1}{\sqrt{2\pi}{}}e^{-\frac{z^2}{2}}}\\
&=\int\frac{1}{\sqrt{2\pi}{\sigma}}e^{-\frac{(z-\mu)^2}{2{\sigma}^2}}ln\frac{1}{\sigma}e^{\frac{1}{2}[z^2-\frac{(z-\mu^2)^2}{\sigma^2}]}dz\\
&=\frac{1}{2}\int\frac{1}{\sqrt{2\pi}{\sigma}}e^{-\frac{(z-\mu)^2}{2{\sigma}^2}}[-ln\sigma^2+z^2-\frac{(z-\mu)^2}{\sigma^2}]dz\\
&=\frac{1}{2}(-ln\sigma^2+\mu^2+\sigma^2-1)
\end{split}
\end{equation}
$$
至此，变分自编码器$VAE$的目标函数也推导出来了。可以看出VAE的目标函数与AE的目标函数相比，不仅有MSE项，还有KL正则项也需优化。

### 扩散模型($Diffusion$)

