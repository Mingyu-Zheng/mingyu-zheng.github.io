---
layout:     post
title:      "「机器学习」 Transformer模型简介"
subtitle:   "Introduction to the transformer model"
date:       2023-03-02 12:00:00
author:     "Azaan"
header-img: "img/post-bg-machine-learning.jpg"
katex: true
tags:
    - 机器学习
---





# 背景问题

Transformer最早起源于自然语言处理（NLP）问题，这一问题不同于卷积神经网络所擅长处理的图像问题，一句话的输入长度/大小，以及不同词汇输入顺序等因素都对这句话的理解起到了非常重要的作用。

例如下面这段话：

> The restaurant refused to serve me a ham sandwich because it only cooks vegetarian food. In the end, they just gave me two slices of bread. Their ambiance was just as good as the food and service.

我们可以看到以下一些问题：

- 首先，编码后的输入将会很大。这句话中包含了37个单词，如果每个单词用一个长度为1024的隐向量来表示，那么整个输入长度为 37888，如果是更长的一段文本，那么输入长度会达到数十万甚至上百万。
- 并且，不同的输入是不等长的。我们不能限制一段文本有严格固定的字数，因此我们必须处理可变长的输入，而这一点就意味着我们很难直接应用一个全连接的网络。而且这一点也表明，我们可能需要让不同位置之间的这些单词共享参数（就像卷积神经网络在图像的不同位置都是应用同一套卷积核一样）。
- 还有，单词是具有歧义的。仅仅从语法角度来说，it 可以指代 restaurant，也可以指代 sandwich，为了理解文本，单词 it 应当和 restaurant 联系起来，这也就是后面Transformer中引入的注意力机制。

# 自注意力机制

一个自注意力块（self-attention block）我们记作 $\mathbf{sa[\bullet]}$，它有 $N$ 个输入 $\mathbf{x_n}$，每个输入都是 $D\times 1$ 维的（即列向量），然后返回 $N$ 个相同大小的输出。在NLP的概念中，一个输入 $\mathbf{x_n}$ 代表一个单词或一个词组。

对于一个自注意力块来说，首先对每个输入计算 $values$：


$$
\mathbf{v_n=\beta_v + \Omega_v x_n}
$$


在这里 $\mathbf{\beta_v}$ 和 $\mathbf{\Omega_v}$ 分别代表偏移（bias）和权重（weight），之后第 $n$ 个输出 $\mathbf{sa[x_n]}$ 是一个 $\mathbf{v_n}$ 的加权和：


$$
\mathbf{sa[x_n]}=\sum_{m=1}^{N} a[\mathbf{x_m,x_n}]\mathbf{v_m}
$$


其中的标量权重 $a[\mathbf{x_m,x_n}]$ 是 $\mathbf{x_m}$ 对 $\mathbf{x_n}$ 的注意力（attention），这 $N$ 个权重都是非负的并且其和为1。

![](https://azaan-zheng.github.io/img/machine-learning/20230302/1.jpg)

如上图所示，三个输入 $\mathbf{x_1,x_2,x_3}$ 分别计算 $values$，然后各自以不同的权重（红色部分）相加作为输出，分别得到 $\mathbf{sa[x_1],sa[x_2],sa[x_3]}$。

### 注意力权重计算

在前面的推导中我们知道，对于每个输入 $\mathbf{x_m}$，我们可以独立地得到对应的 $value$：$\mathbf{\beta_v+\Omega_v x_m}$，这一过程是比较熟悉的；对于不同输入，我们可以计算其之间的注意力权重 $a[\mathbf{x_m,x_n}]$，我们下面将重点探讨这部分权重的计算。

我们对输入进行以下两个线性变换：


$$
\mathbf{q_n=\beta_q+\Omega_q x_n} \\
\mathbf{k_n=\beta_k+\Omega_k x_n}
$$


在这里我们将 $\mathbf{q_n}$ 和 $\mathbf{k_n}$ 分别称为 $query$ 和 $key$。之后我们计算二者之间的点积并且通过 softmax 函数进行处理：


$$
a[\mathbf{x_m,x_n}]=softmax_m[\mathbf{k^T_{\bullet}q_n}] = \frac{exp[\mathbf{k^T_m q_n}]}{\sum_{i=1}^{N}exp[\mathbf{k^T_i q_n}]}
$$


$query$ 和 $key$ 是从信息检索领域所继承的说法，我们可以解释为，一个点乘的结果相当于计算了查询内容 $query$ 和关键字 $key$ 之间的相似性。通过这个权值，我们能够分析这个输入对应提取的是什么位置的信息，从而对应地作用到 $value$ 上，生成对应的结果。

### 矩阵形式

![](https://azaan-zheng.github.io/img/machine-learning/20230302/2.jpg)

上图表示了矩阵形式下自注意力块的运算过程。如图所示，我们对输入的 $N$ 个向量都分别进行前述运算，最终运算的过程相当于对于大小为 $D\times N$ 的矩阵 $\mathbf{X}$ 的运算。$value$，$query$ 和 $key$ 可分别计算如下：


$$
\mathbf{V[X]=\beta_v 1^T + \Omega_v X} \\
\mathbf{Q[X]=\beta_q 1^T + \Omega_q X} \\
\mathbf{K[X]=\beta_k 1^T + \Omega_k X}
$$


这里 $\mathbf{1}$ 是一个 $N\times 1$ 的全1矩阵用来同一维度，相应的，自注意力计算可以表示为：


$$
\mathbf{Sa[X]=V[X]\vdot Softmax[K[X]^T Q[X]]}
$$


这里 $\mathbf{Softmax[\bullet]}$ 是对所得矩阵的每一列进行作用（如果将 $\mathbf{q_n}$ 对应一个列向量，那么可以将 $\mathbf{K}$ 与其作用的结果也对应为一个列向量）。

### 位置编码

不难看出我们上述的分析过程并没有考虑不同输入之间的先后次序关系，但是就像我们之前的例子所说的，相同的单词在不同的位置所表达的含义可能是不一样的，这表明一个输入的顺序信息是决定其表达含义的非常重要的因素，因此我们引入位置编码（position encoding）来引入位置信息。

**绝对位置编码（absolute position embeddings）**：矩阵 $\mathbf{\Pi}$ 被添加到输入 $\mathbf{X}$ 之中，对位置信息进行编码，矩阵中的每一列都是唯一的，因此包含了关于输入序列中位置的信息。这个矩阵可以手工选择，也可以学习。可以被添加到网络输入，或者添加到每一层之中。有时，它在对 $query$ 和 $key$ 进行计算的时候被添加到 $\mathbf{X}$ 中，但是在计算 $value$ 的时候并不添加。

**相对位置编码（relative position embeddings）**：有时其输入可以是一个完整的句子或者许多句子，单词的绝对位置远不如两个输入之间的相对位置重要。相对位置编码为两个位置之间学习一个参数 $\pi_{a,b}$，并使用它来修改注意力矩阵（如相加或相乘等方式）。

















### 参考文献

1. Understanding Deep Learning



