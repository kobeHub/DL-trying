# Train RNNs

训练循环神经网络与普通的神经网络类似，也是使用反向传播算法，但是由于参数被所有的时间戳所共享，所以每一步的训练都依赖于上一步的结果。故采用**BBTT**算法进行训练。

对于一个展开的RNNs网络而言，对其进行BPTT，根据上篇所言的展开图

![unfold](http://media.innohub.top/180614-unfold.png)

为了计算的方便将每一个时间戳 $t$ 的输出$o_{t-1}$记作 $\hat{y}$ .   对于每一个时间戳的操作如下：

$\begin{aligned}  s_t &= \tanh(Ux_t + Ws_{t-1}) \\  \hat{y}_t &= \mathrm{softmax}(Vs_t)  \end{aligned} $

相应的损失函数用交叉熵来定义：

$\begin{aligned}  E_t(y_t, \hat{y}_t) &= - y_{t} \log \hat{y}_{t} \\  E(y, \hat{y}) &=\sum\limits_{t} E_t(y_t,\hat{y}_t) \\  & = -\sum\limits_{t} y_{t} \log \hat{y}_{t}  \end{aligned}  $

$y_t$ 对应在时间戳t下的实际的单词，$\hat{y}$ 是模型的预测值，将整个序列作为一个训练样例，所以总误差就是每个单词训练得到的误差。我们的目标是计算相对于我们训练的参数$U, V ,W$ 的误差梯度，然后使用随机梯度下降将误差值降到最小。

为了计算这些梯度，可以使用分化的链式法则。可以使用$E_3$作为一个样例：

## 对于 $V$ 的梯度计算

$\begin{aligned}  \frac{\partial E_3}{\partial V} &=\frac{\partial E_3}{\partial \hat{y}_3}\frac{\partial\hat{y}_3}{\partial V}\\  &=\frac{\partial E_3}{\partial \hat{y}_3}\frac{\partial\hat{y}_3}{\partial z_3}\frac{\partial z_3}{\partial V}\\  &=(\hat{y}_3 - y_3) \otimes s_3 \\  \end{aligned}  $

在上面公式中，$z_3 = V s_3$ ,符号$\otimes$ 是向量叉乘符号。

##对于$W, U$的梯度计算

$\begin{aligned}  \frac{\partial E_3}{\partial W} &= \frac{\partial E_3}{\partial \hat{y}_3}\frac{\partial\hat{y}_3}{\partial s_3}\frac{\partial s_3}{\partial W}\\  \end{aligned}  $

对于$W$求导时，由于$s_3 是 s_2$ 的函数，$s_2 是s_1$ 的函数。进行求导操作时，会进行连续的链式操作

$\begin{aligned}  \frac{\partial E_3}{\partial W} &= \sum\limits_{k=0}^{3} \frac{\partial E_3}{\partial \hat{y}_3}\frac{\partial\hat{y}_3}{\partial s_3}\frac{\partial s_3}{\partial s_k}\frac{\partial s_k}{\partial W}\\  \end{aligned}  $

所以在计算E3对于Ｗ的导数时，需要一致求导到Ｅ0

![te](http://www.wildml.com/wp-content/uploads/2015/10/rnn-bptt-with-gradients.png)

## BBTP代码实例：

```python
def bptt(self, x, y):
    T = len(y)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation: dL/dz
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            # Add to gradients at each previous step
            dLdW += np.outer(delta_t, s[bptt_step-1])              
            dLdU[:,x[bptt_step]] += delta_t
            # Update delta for next step dL/dz at t-1
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]
```





$\frac{7x+5}{1+y^3}$  $z = z_1 $ $\cdots$ $\alpha$ $sd^{cd}$  $\int ^2_3 x^2 {\rm d}x$    $\lim_{n\rightarrow+\infty} n$  $sum \frac{1}{i^2}\quad\prod \frac{1}{i^2}$  