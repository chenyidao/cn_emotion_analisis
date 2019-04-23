# cn_emotion_analisis
中文情感分析

一.	数据获取
1.1 预训练词向量：
北京师范大学中文信息处理研究所与中国人民大学 DBIIR 实验室的研究者开源的"chinese-word-vectors" 
使用数据："chinese-word-vectors"知乎Word + Ngram
链接地址：https://github.com/Embedding/Chinese-Word-Vectors
1.2 语料：
谭松波老师的酒店评论语料

二.	运行环境
Windows10
JDK 1.8
tensorflow

三.	实现内容
3.1 使用gensim加载预训练中文分词embedding
3.2 加载语料，进行分词、tokenize、padding(填充)、truncating(修剪)等数据处理 
3.3 使用tensorflow的keras接口来建模。(补充：搭建LSTM模型，第一层是Embedding层，只有把tokens索引转换为词向量矩阵后才能使用神经网络对文本进行处理。而keras提供了Embedding接口，避免了繁琐的稀疏矩阵操作。)
3.4 建立权重的存储点，保存模型
3.5 结合flask框架进行结果的前端显示
四.	关键部分
4.1模型构建过程总结：
 
五.	参考资料
网易云deeplearning.ai课程
https://zhuanlan.zhihu.com/p/26306795


六.	实验结果

