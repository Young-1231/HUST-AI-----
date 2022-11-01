# MLP

## 公式推导

### Cross Entropy Loss
令$h$代表最后隐藏层的输出值,$q$代表真实标签的one hot向量,$p$代表网络预测的概率分布
$$L(p,q)=-\sum\limits_{i}p_i\log(q_i)$$
其中$$
q = \text{Softmax}(h)
$$

## 三层MLP推导
样本为$(x_1,y_1),\cdots,(x_N,y_N)$, 样本数量为$N$, 可得$l=-\sum\limits_{i=1}^{N}y_i^T\log\left(\text{Softmax}\left(W_3\sigma(W_2\sigma(W_1x_i+b_1)+b_2)+b_3\right)\right)$
定义$a_{1,i}=W_1x_i+b_1$, $h_{1,i}=\sigma(a_{1,i})$, $a_{2,i}=W_2h_{1,i}+b_2$, $h_{2,i}=\sigma(a_{2,i})$, $a_{3,i}=W_3h_{2,i}+b_3$
利用[矩阵求导术](https://zhuanlan.zhihu.com/p/24709748)中知识可得
$$
\frac{\partial l}{\partial a_{3,i}}=\text{Softmax}(a_{3,i})-y_i
$$
结合[矩阵求导术](https://zhuanlan.zhihu.com/p/24709748)中相关知识可得，$$dl=  \text{tr}\left(\sum\limits_{i=1}^{N}\frac{\partial l}{\partial a_{3,i}}^Tda_{3,i}\right)=\text{tr}\left(\sum\limits_{i=1}^{N}\frac{\partial l}{\partial a_{3,i}}^TdW_3h_{2,i}\right) + \text{tr}\left(\sum\limits_{i=1}^{N}\frac{\partial l}{\partial a_{3,i}}^TW_2dh_{2,i}\right)+\left(\sum\limits_{i=1}^{N}\frac{\partial l}{\partial a_{3,i}}^Tdb_3\right)$$
从第一项可得$\frac{\partial l}{\partial W_3}=\sum\limits_{i=1}^N \frac{\partial l}{\partial a_{3,i}}h_{2,i}^T$, 从第二项可得$\frac{\partial l}{\partial h_{2,i}}=W_3^T\frac{\partial l}{\partial a_{3,i}}$, 从第三项可得$\frac{\partial l}{\partial b_3}=\sum\limits_{i=1}^{N}\frac{\partial l}{\partial a_{3,i}}$。对$\frac{\partial l}{\partial h_{2,i}}$使用sigmoid函数微分的复合公式，可得$\frac{\partial l}{\partial a_{2,i}}=\frac{\partial l}{\partial h_{2,i}}\odot \sigma'(a_{2,i})$. 
同理可得 $\frac{\partial l}{\partial W_2}=\sum\limits_{i=1}^{N}\frac{\partial l}{\partial a_{2,i}}h_{1,i}^T$, $\frac{\partial l}{\partial h_{1,i}}=W_2^T\frac{\partial l}{\partial a_{2,i}}$, $\frac{\partial l}{\partial b_2}=\sum\limits_{i=1}^N \frac{\partial l}{\partial a_{1,i}}$
再次利用sigmoid微分复合公式可得 $\frac{\partial l}{\partial a_{1,i}}=\frac{\partial l}{\partial h_{1,i}}\odot \sigma'(a_{2,i})$
可得$\frac{\partial l}{\partial W_1}=\sum\limits_{1}^{N}\frac{\partial l}{\partial a_{1,i}}x_{1,i}^T$, $\frac{\partial l}{\partial b_1}=\sum\limits_{i=1}^{N}\frac{\partial l}{\partial a_{1,i}}$

## 实验

1. 固定学习率(lr=0.1), batch_size=32, 三层, sigmoid, 无weight decay
![本地路径](img\mlp.png)

2. 固定学习率(lr=0.1), batch_size=32, 三层, tanh, 无weight decay
![本地地址](img\mlp_tanh.png)

3. 固定学习率(lr=0.1), batch_size=1,  三层，tanh, 无weight decay
![本地路径](img\mlp_tanh_2.png)

4. 固定学习率(lr=0.1), batch_size=32, 三层，tanh, weight_decay=0.001
![本地路径](img\mlp_tanh_3.png)

5. 固定学习率(lr=0.1), batch_size=32, 三层, tanh, 无weight decay, 在最后一个fc层前加上dropout = 0.5
![本地路径](img\mlp_tanh_dropout.png)

6. StepLR(step_size=20, gamma=0.8),batch_size=32,三层，tanh, 无weight decay
![本地路径](img/mlp_tanh_lr.png)

7. 固定学习率(lr=0.1), batch_size=32, 五层, tanh, 无weight decay
![本地路径](img/mlp_tanh_5layer.png)

8. 固定学习率(lr=0.1), batch_size=32, 五层, tanh, 无weight decay, dropout=0.5
![本地路径](img/mlp_tanh_5layer_dropout.png)

9. 固定学习率(lr=0.1), Adagrad, 三层, tanh, 无weight decay
![本地路径](img/mlp_tanh_adagrad.png)

10. 固定学习率(lr=0.1), Adam, 三层， tanh, 无weight decay
![本地路径](img/mlp_tanh_adam.png)

## 分析
1. StepLR vs 固定学习率
- 在此项任务上采用固定学习率的表现更好
- StepLR在测试集上的正确率曲线具有一个明显的随学习率衰减的上升支.
  
1. Sigmoid vs tanh
- 总体来说learning curve很相近，在测试集上的正确率也相近
> Sigmoid 和 tanh均采用了数值稳定形式, 避免前向传播时上溢

3. batch_size=32(mini-batch SGD) vs batch_size=1(vanilla SGD)
- vanilla SGD由于每次只接受一个样本作为输入，具有较大的随机性，learning curve表现出来相对震荡, 并且正确率也比较低

4. 3layers vs 5layers
- 5层网络因为具有更强的学习能力，较早的在训练集上收敛，导致存在相对的更大的过拟合现象

5. dropout
- dropout可以理解为: 1. 训练多个子网络,做ensemble 2. 添加随机噪声, 增强泛化能力
- dropout机制的引入对于三层网络的提升较少，而在五层网络中可以看到是可以一定程度上提升在测试集上的准确率

6. Adam vs Adagrad
- 在其它条件相同时,在本任务上Adagrad相比Adam的表现更好，收敛到一个更好的状态，Adam的learning curve较为震荡
- Adam的震荡与初始学习率设定应该有关
