#encoding:utf-8
# 首先加载必用的库
import numpy as np
import re
import jieba # 结巴分词
import os
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")
# 我们使用tensorflow的keras接口来建模
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 使用gensim加载预训练中文分词embedding
def get_cnmodel():
    cn_model = KeyedVectors.load_word2vec_format('G:\pycharm_program\deeplearning\sgns.zhihu.bigram', binary=False)
    return cn_model
# cn_model = get_cnmodel()

# 获得样本的索引，样本存放于两个文件夹中，
# 分别为 正面评价'pos'文件夹 和 负面评价'neg'文件夹
# 每个文件夹中有2000个txt文件，每个文件中是一例评价

def get_comment():
    pos_txts = os.listdir('G:/pycharm_program/deeplearning/pos')
    neg_txts = os.listdir('G:/pycharm_program/deeplearning/neg')
    train_texts_orig = [] # 存储所有评价，每例评价为一条string

    # 添加完所有样本之后，train_texts_orig为一个含有4000条文本的list
    # 其中前2000条文本为正面评价，后2000条为负面评价

    for i in range(len(pos_txts)):
        with open('G:/pycharm_program/deeplearning/pos/'+pos_txts[i], 'r', errors='ignore') as f:
            text = f.read().strip()
            train_texts_orig.append(text)
            f.close()
    for i in range(len(neg_txts)):
        with open('G:/pycharm_program/deeplearning/neg/'+neg_txts[i], 'r', errors='ignore') as f:
            text = f.read().strip()
            train_texts_orig.append(text)
            f.close()
    return train_texts_orig
# comment = get_comment()

# 进行分词和tokenize
# train_tokens是一个长长的list，其中含有4000个小list，对应每一条评价
def get_train_tokens(cn_model):
    train_tokens = []
    train_texts_orig = get_comment()
    for text in train_texts_orig:
        # 去掉标点
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
        # 结巴分词
        cut = jieba.cut(text)
        # 结巴分词的输出结果为一个生成器
        # 把生成器转换为list
        cut_list = [ i for i in cut ]
        for i, word in enumerate(cut_list):
            try:
                # 将词转换为索引index
                cut_list[i] = cn_model.vocab[word].index
            except KeyError:
                # 如果词不在字典中，则输出0
                cut_list[i] = 0
        train_tokens.append(cut_list)
    return train_tokens

# train_tokens = get_train_tokens(cn_model)
# print(train_tokens)
# print(len(train_tokens))

def get_num_max_tokens(train_tokens):
    # 获得所有tokens的长度
    num_tokens = [ len(tokens) for tokens in train_tokens ]
    num_tokens = np.array(num_tokens)
    # 取tokens平均值并加上两个tokens的标准差，
    # 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖95%左右的样本
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    return num_tokens,max_tokens
# num_tokens,max_tokens = get_num_max_tokens(train_tokens)
# print(max_tokens)
# print(num_tokens)

def get_embedding_matrix(cn_model):
    # 只使用前50000个词
    num_words = 50000
    embedding_dim = 300
    # 初始化embedding_matrix，之后在keras上进行应用
    embedding_matrix = np.zeros((num_words, embedding_dim))
    # embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
    # 维度为 50000 * 300
    for i in range(num_words):
        embedding_matrix[i, :] = cn_model[cn_model.index2word[i]]
    embedding_matrix = embedding_matrix.astype('float32')

    return embedding_matrix
# embedding_matrix = get_embedding_matrix(cn_model)
# print(embedding_matrix)
# print(embedding_matrix.shape)

def get_train_pad(train_tokens,max_tokens):
#     print(train_tokens)
    num_words = 50000
    # 进行padding和truncating， 输入的train_tokens是一个list
    # 返回的train_pad是一个numpy array
    train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                                padding='pre', truncating='pre')
    # 超出五万个词向量的词用0代替
    train_pad[ train_pad>=num_words ] = 0
    return train_pad

# train_pad = get_train_pad(train_tokens,max_tokens)
# print(train_pad[31])
# print(train_pad.shape)

def get_train_target():
    # 准备target向量，前2000样本为1，后2000为0
    train_target = np.concatenate((np.ones(2000),np.zeros(2000)))
    return train_target
# train_target = get_train_target()
# print(train_target)

# 进行训练和测试样本的分割

def split_train_test(train_pad,train_target):
    # 90%的样本用来训练，剩余10%用来测试
    X_train, X_test, y_train, y_test = train_test_split(train_pad,
                                                        train_target,
                                                        test_size=0.1,
                                                        random_state=12)
    return X_train,X_test,y_train,y_test
# X_train,X_test,y_train,y_test = split_train_test(train_pad,train_target)
# print(train_pad)

def train_model(embedding_matrix):
    num_words = 50000
    embedding_dim = 300
    max_tokens = 236
    # 用LSTM对样本进行分类
    model = Sequential()
    # 模型第一层为embedding
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_tokens,
                        trainable=False))
    model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
    model.add(LSTM(units=16, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    # 我们使用adam以0.001的learning rate进行优化
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    return model
# my_model = train_model(embedding_matrix)

def get_callbacks(model):
    # 建立一个权重的存储点
    path_checkpoint = 'sentiment_checkpoint.keras'
    checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                                 verbose=1, save_weights_only=True,
                                 save_best_only=True)
    # 尝试加载已训练模型
    try:
        model.load_weights(path_checkpoint)
    except Exception as e:
        print(e)
    # 定义early stoping如果3个epoch内validation loss没有改善则停止训练
    earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    # 自动降低learning rate
    lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                     factor=0.1, min_lr=1e-5, patience=0,
                                     verbose=1)
    # 定义callback函数
    callbacks = [
        earlystopping,
        checkpoint,
        lr_reduction
    ]
    return callbacks

# callback = get_callbacks(my_model)

def start_train(model, callback):
    # 开始训练,20个批次，每个批次大小为128
    model.fit(X_train, y_train,
              validation_split=0.1,
              epochs=10,
              batch_size=128,
              callbacks=callback)
# start_train(model,callbacks)

#准确度
def get_accuracy(model,X_test,y_test):
    result = model.evaluate(X_test, y_test)
    print('Accuracy:{0:.2%}'.format(result[1]))

# get_accuracy(model,X_test,y_test)

def predict_sentiment(model,text):
    print(text)
    # 去标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [ i for i in cut ]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            cut_list[i] = 0
    # padding
    tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
                           padding='pre', truncating='pre')
    # 预测
    result = model.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('是一例正面评价','output=%.2f'%coef)
    else:
        print('是一例负面评价','output=%.2f'%coef)

def test(model,text_list):
    for text in text_list:
        predict_sentiment(model,text)


# 测试
if __name__ == '__main__':
    # 词向量模型
    cn_model = get_cnmodel()
    # 语料
    comment = get_comment()
    # 处理语料，转换为词向量模型的索引值
    train_tokens = get_train_tokens(cn_model)

    num_tokens, max_tokens = get_num_max_tokens(train_tokens)
    embedding_matrix = get_embedding_matrix(cn_model)
    # 填充
    train_pad = get_train_pad(train_tokens, max_tokens)
    # 前2000条是1，后两千条是0
    train_target = get_train_target()
    # 划分数据集
    X_train, X_test, y_train, y_test = split_train_test(train_pad, train_target)
    # 训练模型
    my_model = train_model(embedding_matrix)
    callback = get_callbacks(my_model)
    start_train(my_model, callback)

    test_list = [
        '酒店设施不是新的，服务态度很不好',
        '酒店卫生条件非常不好',
        '床铺非常舒适',
        '房间很凉爽，空调冷气很足',
        '酒店环境不好，住宿体验很不好',
        '房间隔音不到位',
        '晚上回来发现没有打扫卫生',
        '因为过节所以要我临时加钱，比团购的价格贵'
    ]
    test(my_model,test_list)