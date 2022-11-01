# Self-attention
---
## Self-attention的理解和代码实现

### 核心公式
核心的公式如下: $$
\text{Attention}(Q,K,V)= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
核心即是通过$Q$和$K$计算得到attention weight, 再作用于$V$得到输出
其中$Q,K,V$分别为三个矩阵，并且三者`.shape[1]`(第二个维度)分别为$d_q,d_k,d_v$, 并且$d_q=d_v$, 图中的**scale**操作即为除以$\sqrt{d_k}$这一缩放因子。
而$Q,K,V$是输入$X$分别乘以三个不同的矩阵计算而来
$$
\begin{aligned}
X W^Q = Q \\ XW^K = K \\ XW^V = V
\end{aligned}    
$$
借助self attention,只需要对原始输入进行一些矩阵乘法便能得到最终包含有不同注意力信息的向量。

## encoder-decoder机制理解和代码实现

encoder: 接受一个长度可变的序列作为输入，并将其转化为具有固定形状的上下文向量
decoder: 将encoder输出的上下文向量作为初始隐状态，并接受一定输入，最终输出长度可变的序列

encoder实现: 
```python
class Encoder(nn.Module):
    """
    encoder基类
    """
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

decoder实现
```python
class Decoder(nn.Module):
    """
    decoder基类
    """        
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    # 将encoder的输出转化为为decoder的初始隐状态
    def init_state(self, encoder_outputs, *args):
        raise NotImplementedError
    
    def forward(self, X, *args):
        raise NotImplementedError
```

encoderdecoder实现
```python
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder 
    
    def forward(self, enc_x, dec_x, *args):
        encoder_output = self.encoder(enc_x, *args)
        dec_state = self.decoder.init_state(encoder_output, *args)
        return self.decoder(dec_x, dec_state)
```
