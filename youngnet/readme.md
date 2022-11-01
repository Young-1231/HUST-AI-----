# youngnet


## 参考
> [PyTorch](https://github.com/pytorch/pytorch.git)
> [PyDyNet](https://github.com/Kaslanarian/PyDyNet.git)

## 文件说明

* activations
    - 实现Sigmoid, tanh, ReLU激活函数

* functional
    - 在前向传播和backpropagation时所用到的函数

* dropout
    - 实现dropout机制

* layers
    - 目前仅实现Linear层
  
* loss
    - 实现Cross Entropy Loss, MSE loss, NLL loss

* lr_scheduler
    - 目前仅实现StepLR(固定步长衰减)学习率调节策略

* optim
    - 实现SGD, Adagrad, Adam

* preprocess
    - 实现 zscore normalization, min-max scale预处理操作

* utils
    - 实现dataloader, onehot encoder和简单分类任务下分类准确率的计算工具

* init
    - 实现xavier和kaiming权重初始方法

* module
    - Module
