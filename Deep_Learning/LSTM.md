<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# 了解LSTM神经网络

[TOC]

## RNN 递归神经网络
人类是不能马上开始进行思考的。 当你在阅读这篇文章时，你基于之前的文字了解阅读时的每一个词。你也不会完全摈弃所有的东西重新开始思考。你的思想具有持续性。传统的神经网络不能做到这一点，这也是它的一个主要缺点。举个例子，你想辨别影片放映时每一帧正在发生的事情。传统的神经网络很难通过之前发生的事情去判断之后将要发生的事。
递归神经网络却解决了这个问题。它们是一个自循环的网络，同时也允许信息的保留。

![Recurrent Neural Networks have loops.](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png)

以上图一个小的神经网络A为例，输入是\\(x_t\\)，输出是\\(h_t\\)。通过循环可以将上一步的信息再重新传递到下一步。这些循环式RNN充满了某种神奇的力量。然而如果再仔细想想，它们其实本没有和正常的神经网络有多大的区别。一个递归神经网络可以想象成一个神经网络的多个拷贝，一个一个将信息传递给后继者。我们可以将循环展开：

![An unrolled recurrent neural network.](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

这一如同链的特性使RNN和序列、列表紧密相关起来，也是神经网络用来处理这些数据的天然结构。这相当有用！在过去的几年中，RNN被成功运用解决了多种问题包括：语音识别、语言模型、翻译、图像识别...不甚枚举。其中有一篇Andrej Karpathy's的精彩博文[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)让我印象深刻。
这些成功的运用主要归功于LSTM，一个远远优于标准RNN的特殊类别。几乎所有与RNN有关的令人兴奋的结果都离不开LSTM。

## 长期记忆的问题
将之前的信息链接到当前的任务中是RNN其中一个吸引点，如使用之前的视频帧来抱住理解当前帧。如果RNN能做到这一点那是相当有用的。不过可不可以还得看情况。
有时候我们只需要最近的一些信息来完成当前的任务。譬如，一个语言模型通过前几个词语预测下面一个词语——`“the clouds are in the sky”`当中的`sky`。很显然我们只需要前面几个单词“the clouds are in the”无他。在这样的情况下，相关信息的分歧很小。RNN可以通过这样使用之前的信息。

![RNNs can learn to use the past information.](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-shorttermdepdencies.png)

但有的时候，我们需要更多的上下文。如预测`“I grew up in France… I speak fluent French.”`中的最后一个词。最近的信息表明，下一个单词可能是语言的一种。但是我们想进一步缩小范围，我们需要获取上文中的`France`。而此时相关联的信息与当前位置相距甚远。
不幸的是，当距离增加，RNN变很难学会链接相关信息了。

![Unfortunately, as that gap grows, RNNs become unable to learn to connect the information](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-longtermdependencies.png)

理论上，RNN完全可以处理这样的长期依赖。我们可以通过精心调整参数来解决这个问题。然而现实当中却极为不易。这个问题又再一次地在Hochreiter和Bengio[(1994)](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf)等人的论文中提及。幸运的是，LSTM不存在这样的问题！

## LSTM神经网络
长短期记忆神经网络（LSTM）是一类可以处理长期记忆的特殊RNN神经网络。它由Hochreiter与Schmidhuber在[1997](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)年提出，随后又被很多人进行了优化与改良。它在各种问题上都表现出色以至于被广泛使用。
LSTM可以从容处理长期信息，同时也是他们的默认行为。所有的RNN都具有重复的链式神经网络结构。在标准的RNN模型中重复模型可以表达为一个简单的`tanh`层。

![The repeating module in a standard RNN contains a single layer.](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)

LSTM模型也拥有类似的结构，只是更为复杂。有之前的一层结构转变成了四层。

![The repeating module in an LSTM contains four interacting layers.](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

下面来详细解释一下上面的结构。首先我们了解一下上述图中的图例都是什么意思。

![notation](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM2-notation.png)
上图中，每一条从输出到另一个输入的直线代表一个完整向量。红色圆圈代表点操作，如向量加法，黄色矩形代表一个神经网络层。线合并则表示向量的合并。

## LSTM的中心思想
LSTM的关键在于状态变化，由一条水平直线从图顶部穿过。细胞状态如果一个传送带，它穿越了整个链，只进行了简单的线性运算，使得信息可以在这个过程中轻易保持不变。

![cell state](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png)

LSTM通过门运算可以对胞进行信息的删除与增加。门结构可以选择性地使信息通过，它包括一个sigmoid层和一个乘运算。Sigmod层输出0到1之间的数，表示多少比例的信息可以通过。0表示“全部不能通过”，而1表示“可以全部通过”。一个LSTM结构有这样的三个门用于保护和控制胞状态。

## LSTM分步
LSTM的第一步是决定我们可以将什么信息去除。这个操作由一个遗忘门进行。通过输入\\(h_{t-1}\\)和\\(x_t\\)来在状态\\(C_{t-1}\\)中输出一个0和1之间的数。0表示“全部不能通过”，而1表示“可以全部通过”。让我们继续之前语言模型预测的例子。在这个问题中，胞需要包括当前主语的性别一边选取正确的代词，而当我们遇到新的主语时我们需要忘记之前的主语性别。

![forget gate](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png)

下一步是决定我们将储存哪些信息。这个过程包含两部分，第一部分一个sigmoid“输入门”决定更新哪些值，另一个tanh层产生新的待用向量\\(\tilde{C}_t\\)，之后我们将这两部分合并以更新状态。在我们的例子中，我们希望用新主语的性别去替代原来的主语。

![input gate](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png)

经过上述两部以后我们便能将状态\\(C_{t-1}\\)更新到新状态\\(C_t\\)。首先我们将前一状态\\(C_{t-1}\\)乘以\\(f_t\\)，忘掉我们决定忘掉的部分。然后加上新的缩放后的值\\(i_t*\tilde{C}_t\\)。在本次例子中，我们将去除旧主语的性别并加入新的信息。

![forget and update](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png)

最后我们需要决定输出什么，输出基于过滤后的胞状态。首先使用sigmod层来决定应该输出胞状态的哪一部分，然后用tanh层将输出置为-1到1之间同时乘以sigmoid门的结果。对于语言模型这个例子来说，优于只看到了一个主语，因此我们需要输出关于动词的一些信息，如主语是单数还是复数。

![output](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png)

## 长短期记忆变种
到目前所描述的是一个标准的LSTM模型，然而并不是所有的LSTM都和上述结构相同。事实上，每一个运用在论文中的模型都或多或少有些区别。
其中一个流行变种由Gers和Schmidhuber[(2000)](ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf)提出，其中增加了`peephole connections`，通过门结构监控胞状态。

![peephole connections](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-peepholes.png)

上图显示，每个门结构都增加了一个`peephole`。其他论文则或有异同。
另一个变种则进行了遗忘门与输出门的耦合。该结构用以共同进行决策而不是让遗忘和添加分开进行。这样我们只在要进行对应的输入时才进行遗忘，也仅在遗忘之前部分时才计入新值。

![optional](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-tied.png)

一个变化更显著的变种是GRU（Gated Recurrent Unit），由[Cho](http://arxiv.org/pdf/1406.1078v3.pdf)等人在2014年提出。该模型将遗忘门和输入门合并为一个“更新门”。同时将胞状态和隐藏状态合并。最终模型比标准的LSTM模型简单，因此广为流行。

![GRU](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)

当然还有其他变种，如[Yao, et al. (2015)](http://arxiv.org/pdf/1508.03790v2.pdf)。当然也有使用其他方法来解决长期记忆的问题，如[Koutnik, et al. (2014)](http://arxiv.org/pdf/1402.3511v1.pdf)。
那么哪个变种效果最好呢？[Greff, et al. (2015)](http://arxiv.org/pdf/1503.04069.pdf)进行了一个较为全面地比较，发现这些模型效果普遍相似。[Jozefowicz, et al. (2015)](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)测试了超过10000的RNN结果发现有些结构在某些问题上比LSTM效果更好

## 结论
LSTM的确在多个领域卓有成效，如果将他们以公式的形式解出来实在是有些吓人。此文希望通过新的形式使读者能一步步接触到LSTM的精华之所在。LSTM是RNN模型的一大步。那么有人会问还会有下一个极大地进展吗？学者们的普遍回答是“是的”。其主要思想是让RNN在每一步都与一个更大集合的信息进行对照。例如，当你在使用RNN进行图像识别时，它可以将图像的一部分与输出的每个词进行一一匹配。事实上，[Xu, et al. (2015)](http://arxiv.org/pdf/1502.03044v2.pdf)已经做了类似有趣的实验，而这些也值得我们继续探究。不只在RNN上的研究，[Kalchbrenner, et al. (2015)](http://arxiv.org/pdf/1507.01526v1.pdf)的Grid LSTMS也令人期待。当然也还有用于生成模型的RNN研究，如[ Gregor, et al. (2015)](http://arxiv.org/pdf/1502.04623.pdf), [Chung, et al. (2015)](http://arxiv.org/pdf/1506.02216v3.pdf)和[Bayer & Osendorfer (2015)](http://arxiv.org/pdf/1411.7610v3.pdf)。过去的几年是RNN领域令人兴奋的时节，当然今后将会出现更多！