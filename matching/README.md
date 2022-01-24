## Sentence_Matching

该项目主要是文本匹配服务，根据channel不同调用不同模型来计算两句话的相似度。

1. channel为0：使用SimCSE模型
2. channel为1：使用ESimCSE模型，并使用同一句话通过dropout作为正样本对，引入动量对比增加负样本对。
3. channel为2：使用ESimCSE模型，并重复一句话中部分词组构造正样本对，引入动量对比增加负样本对。
4. channel为3：同时使用ESimCSE和SimCSE模型，加上多任务损失函数。
5. channel为4：使用PromptBERT模型。
6. channel为5：使用SBert模型。
7. channel为6：使用CoSent模型。
