## Analysis
Focus on papers and related resources that provide theoretical or empirical analysis for deep learning optimization algorithms.

## A.Convergence Analysis
从优化的视角研究优化算法是最为自然、最为传统的研究方向，针对很多问题的研究甚至在深度学习提出之前就已经相当充分了。其主要研究对象为优化算法若干迭代序列的收敛性与收敛速度。本部分针对研究的算法分为GD、SGD、动量方法及自适应方法两大类。

This part focus on the conventional convergence analysis in optimization, which covers several directions:
* (1) Convergence Analysis of GD, SGD and Momentum
* (2) Convergence Analysis of Adaptive Methods

#### A.1. Convergence analysis of GD, SGD and Momentum

#### A.2. Convergence analysis of Adaptive Gradient Methods
* **On the convergence of a class of Adam-type algorithms for non-convex optimization**，*Xiangyi Chen, Sijia Liu, Ruoyu Sun, Mingyi Hong*，ICLR 2019，2018
* **On the convergence proof of AMSGrad and a new version**，*Tran Thi Phuong,  Le Trieu Phong*，IEEE Access，2019
* **A Simple Convergence Proof of Adam and Adagrad**，*Alexandre Défossez, Léon Bottou, Francis Bach, Nicolas Usunier*，arXiv preprint arXiv:2003.02395，2020
* **Towards practical Adam: Non-convexity, convergence theory, and mini-batch acceleration**，*Congliang Chen, Li Shen, Fangyu Zou, Wei Liu*，JMLR，2022

## B.Loss Landscape Analysis & Learning Dynamics
随着深度学习优化算法的日趋复杂，从传统的优化角度理解优化器愈显乏力，Loss Landscape Analysis（暂时没看到较好的翻译）为理解深度学习优化提供了超越传统收敛分析的新见解和理论。其抛弃了传统优化分析框架下只能得知梯度及某一时刻Loss的受限视角，转而将损失函数置于更高维度下将其可视化出来。广义而言，其可以研究模型结构、初始化、优化算法、训练策略等因素对整体深度学习效果的影响。在此框架下，我们可以更加方便地研究不同优化算法性能各异的理论原因。Learning Dynamics（学习过程的动力学分析）借鉴了一定动力系统与物理学中的概念，将深度学习的优化视作一个动态过程，研究Loss Landscape中特殊地形（sharp minima, flat minima, 高原...）对优化的影响等。由于本部分重点在Learning Dynamics，一些经典的Loss Landscape Analysis这里不收集。

This part focus on the learning dynamics of deep-learning optimization, which covers several directions:

* (1) Learning Dynamics of GD, SGD and Momentum
* (2) Learning Dynmaics of Adaptive Methods

#### B.1. Learning Dynamics of GD, SGD and Momentum

#### B.2. Learning Dynmaics of Adaptive Methods
* **Towards Theoretically Understanding Why SGD Generalizes Better Than ADAM in Deep Learning**，*Pan Zhou, Jiashi Feng, Chao Ma, Caiming Xiong, Steven Chu Hong Hoi, Weinan E*，NeurIPS 2020，2020  [[NeurIPS PDF]](https://proceedings.neurips.cc/paper/2020/file/f3f27a324736617f20abbf2ffd806f6d-Paper.pdf)

