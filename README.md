## sentence_representation_matching

该项目主要是文本匹配相关模型，包含使用SimCSE、ESimCSE、PromptBert三种无监督文本匹配模型和SBert、CoSent两种有监督文本匹配模型。

### 无监督文本匹配

#### 1. SimCSE

利用Transformer Dropout机制，使用两次作为正样本对比，以此来拉近正样本，推开负样本。

参考：

1）https://github.com/princeton-nlp/SimCSE

2）https://github.com/KwangKa/SIMCSE_unsup

3）https://arxiv.org/pdf/2104.08821.pdf

#### 2. ESimCSE

在SimCSE的基础上，通过重复句子中部分词组来构造正样本，同时引入动量对比来增加负样本。

参考：https://arxiv.org/pdf/2109.04380.pdf

#### 3. PromptBert

使用Prompt方式来表征语义向量，通过不同模板产生的语义向量构造正样本，同一批次中的其他样本作为负样本。

参考：https://arxiv.org/pdf/2201.04337.pdf

### 有监督文本匹配

#### 1. SBert

使用双塔式来微调Bert，MSE损失函数来拟合文本之间的cosine相似度。

模型结构：

![SBERT Siamese Network Architecture](pics/sbert.png)

参考：https://www.sbert.net/docs/training/overview.html

#### 2. CoSent

构造一个排序式损失函数，即所有正样本对的距离都应该小于负样本对的距离，具体小多少由模型和数据决定，没有一个绝对关系。

损失函数：

![cosent_loss](pics/cosent_loss.svg)

参考：

1）https://spaces.ac.cn/archives/8847

2）https://github.com/shawroad/CoSENT_Pytorch
