#Recurrent Netural Networks 基础

[TOC]

## 1.简介

RNNs 循环神经网络，是十分流行的模型，在进行自然语言处理(NLP)方面展现了极大的潜力．本文基于 

[recurrent neural network based language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf) .主要应用有两个方面，一是可以根据现实世界的语言使用情况，对于任意语句进行评分，提供了一套可以度量语法语义的正确性的依据，这样的模型一般作为机器翻译系统的一部分使用；第二个模型可以根据情景而生成文字，在莎士比亚的基础上训练模型就可以产生莎士比亚式的文章．[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 展示了基于RNN的字符识别模型的使用．

## 2.什么是RNNs

### Contrast

RNNs是用来处理序列数据的．在传统的神经网络模型中，从输入层到隐含层再到输出层，是一种前向反馈神经网络(**Feed-forward Neural Networks**) ,因为所有的信号都在单个方向上传递，所以只可以对单个对象进行处理，而不能获取序列化的数据信息．序列化数据，包含大量信息，而且具有复杂的时间依赖性

**Humans aren’t built to just do linear or logistic regression, or recognize individual objects. We can understand, communicate, and create.** 

和前向反馈网络一样，RNNs也是基于神经元构建的，区别在与神经元的连接方式．前向反馈神经网络，以层的结构构建，signals只能按照唯一的方向进行传播（from input to output）也不允许循环的存在；RNNS与自身连接，这样就可以考虑时间的因素了，因为一个神经元的状态会受到previous step神经元的状态的影响．具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。理论上，RNNs能够对任何长度的序列数据进行处理。但是在实践中，为了降低复杂性往往假设当前的状态只与前面的几个状态相关.

![constranst](http://media.innohub.top/180614-rnn.png)

| Feed-forward                        | RNNs                                     |
| ----------------------------------- | ---------------------------------------- |
| computational unit(neuron)          | computational unit(neuron)               |
| DAG(非循环有向图)                   | Loops                                    |
| Signals are passed in one direction | Signals are sent back to the same neuron |
| Each layer has their own variables  | All steps share the same variables       |

### RNNs 的优点

+ **Take advantage of sequential information of data (texts, genomes, videos, etc.)Generally reduce the total number of parametersForm the backbone of NLP**
+ **Generally reduce the total number of parametersForm the backbone of NLP**
+ **Form the backbone of NLP**

## 3. SRNs  

Simple Recurrent Neural Network, Elman's SRNs.在其早期的模型中，当前step下计算的隐藏层是，是当前step的输入与上一次step的隐藏层的二元函数．在Elman之前，Jodan也建立了类似的模型，但是当前step的隐含层是当前step的输入以及previous step 的输出的二元函数．

![srns](http://media.innohub.top/180614-srn.png)

可以将循环神经网络展开为序列的形式，对于SRNs每一个隐含层接受两个输入：上一层隐含层，以及本层的input．而展开后每一个标记代表一个step,而所有的step共享同一个weights,所以可以大大减少参数的数量．

![unfold](http://media.innohub.top/180614-unfold.png)

对于SRNs网络展开后的运行流程如下，包含每一个step的隐含层包含两个输入:当前的输入数据以及上一步的状态

![context](http://media.innohub.top/180615-context.png)

## 4. BPTT

**Back-Progpagation Through Time**　通过时间的反向传播．

在一个前向反馈的神经网络中，或者是在一个卷积神经网络中，errors会从损失函数反向传播到所有层．这些错误用于更新参数(weights, bias) ，依照所确定的更新法则（例如：gradient descent ,Adam ,...）

### 反向传播的区别

在一个RNNs神经网络中，errors会从损失函数反向传播到所有的时间戳(timestep),它与`FFN`

的区别在于：

+ 在FFN网络的每一层都要有独立的参数，在循环神经网络中，所有的时间戳(**TImestep**)共享相同的参数．对于每一个训练样本／批次（**sample/sequence**），需要使用所有时间戳的梯度来更新参数
+ 一个前向反馈神经网络具有确定数量的层数，但是一个RNNs,可能有不确定的时间戳数量，而时间戳的数量取决于所处理的序列的长度．

如果需要处理的序列对象很长，比如说有1k个词，就要有1k个时间戳，那么进行反向传播的过程就会消耗大量的计算资源；而且也可能会导致梯度的指数式增加或者减少，分别被称为梯度爆炸（**Gradient explode**）,梯度消失（**Gradient vanish**）[*According to:*](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)

### 解决方案

为了避免对于所有时间戳的参数的更新，经常限制时间戳的数量，也就是使用切割的基于时间的方向传播(**Truncated BPTT**) 可以加速每次的参数更新，缺点在于，由于仅使用了有限数量的时间戳参与运算，所以网络模型不能从一开始就学习所有的依赖以及特征．

在tensorflow中，RNNs是以展开的网络结构创建的，在non-eager模式下，这种解决方式意味着在计算前有一定数量的时间戳被确定

**A fixed number of timesteps**

+ **Won’t be able to capture the full sequential dependenciesIn **
+ **In non-eager TensorFlow, have to make sure all inputs are of the same length**

RNNs模型的局限性在于在实际应用中，该模型不能很好地捕捉长期的依赖

```
"I grew up Iin Franch ... I speak fluently"
->需要返回以获取信息
```

## 5. Gated Recurrent Unit (LSTM & GRU)

### LSTM

门控循环单元包含两个常见的模型。用以解决RNNs处理长期依赖的短板。**Long  Short-Term Memory**(LSTM)长短期记忆单元，在近３年得到了广泛关注，实际上是一个很老的概念，在上世纪９０年代被两个德国的研究人员提出用以解决梯度消失问题。与大多数AI问题的解决一样，由于计算力的提升以及数据量的增多，使得这一想法付诸实践。

模型的基本单元称为cell,可以把cells看作是黑盒用以保存当前输入xtxt之前的保存的状态ht−1ht−1，这些cells更加一定的条件决定哪些cell抑制哪些cell兴奋。它们结合前面的状态、当前的记忆与当前的输入。已经证明，该网络结构在对长序列依赖问题中非常有效。[*More detials*](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 

LSTM模型的单元使用了一种称为门控的机制，来控制每个时间戳进入和发出的神经单元的信息。包含了４个门，记作`i, o, f, c` 分别对应　`input, output, forget, candidate/new memory`  

+ **input gate:** 控制当前允许通过的数据量
+ **forget gate:** 控制上一层需要考虑的数据量
+ **output gate:** 控制本次隐含层向上下一层暴露的信息的数量
+ **candidate gate:** 与基本RNNs模型相似，该门根据当前层的输入以及上一层的状态控制计算该隐含层的候选状态
+ **final gate:** 该单元的内部存储器将候选的隐藏状态，与`input/forget` gate 信息结合起来，然后使用输出门计算最终存储单元的状态，以决定将最终存储单元的多少信息输出作为当前的步骤的隐藏状态

![iofc](http://media.innohub.top/180616-lstm.png)

[其基本计算思路如下：](https://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf)

注意　f ◦ g =  f(g(x))   是离散数学中复合函数的表达形式 

![process](http://media.innohub.top/180616-process.png)

直观上看待，这些门可以看做是，在每一个时间戳上，控制数据进出的控制单元。所有的门具有相同的维度。

### GRU

门控循环单元是另外一中常用的用于处理长期序列的RNNs变体。相比于LSTM其结构更为简单，将长短期记忆单元的输入控制们以及遗忘控制门相结合，归一化为一个“update gate”.同时整合了  candidate /new cell 的状态以及隐藏层的状态。这样使得GRU比标准的LSTM模型要简单得多。而在一些基准测试任务中，他的表现已经与LSTM的表现一致。结构的简单意味着需要更少的计算量，那么会节省大量的计算时间，但是该模型并没有表现出计算时间的优越性。

**基本结构如下：**

![gru](http://media.innohub.top/180616-gru.png)

**计算过程：**

![gru_pro](http://media.innohub.top/180616-grup.png)

##*Reference:*

[BPTT](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)

[intro to LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[Deep learning for NLP](https://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf)

