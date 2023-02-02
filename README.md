# BERT-pytorch源码阅读记录和各模块调试流程
Source code：Google AI 2018 BERT pytorch implementation\
调试者：wanghesong2019
# 项目背景
 Google 这个代码写的真是漂亮， 结构清晰；但原始repository只给出了2句简单的命令行，只是执行下它们很难让我清晰了解到bert模型的内部运行机制，故下决心对该项目代码一行行啃；
 阅读心得以注释的方式加在了代码中；调式过程单独起一个py文件，供自己和看官参考；
# 整体框架
![BERT-pytorch框架](https://raw.githubusercontent.com/wanghesong2019/BERT-pytorch/master/img/1.PNG)
在上面bertEmbedding-loss-train模式中，bertEmbedding无疑是最重要的，因为其不仅包含bert的3种embedding，还实现了bert的2种pre-training目标，具体构成如下：\
![BERT-pytorch框架](https://raw.githubusercontent.com/wanghesong2019/BERT-pytorch/master/img/2.PNG)
