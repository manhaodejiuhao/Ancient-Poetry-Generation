# coding: UTF-8
# 本程序用gensim训练字向量
from data import *
from parameter import *
from gensim import models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class charvec:
    # 初始训练字向量的模型，如果已经训练过了就加载
    def __init__(self, istrain = False):
        if istrain:
            self.model = models.Word2Vec(poem_Data.poems, sg=1, size=100, window=5, min_count=2, negative=3, sample=0.001, hs=1,
                             workers=4)
            self.model.train(sentences=poem_Data.poems, total_examples=len(poem_Data.poems), epochs=20)
            self.model.save(charvec_path)
        else:
            self.model = models.Word2Vec.load(charvec_path)

    # 关键字补全（输入一个列表，返回一个列表，返回的列表是意思相近的关键字）
    def gene_simi_chars(self, characters):
        res = characters
        if len(characters) == 1:
            # 将字库排序，排序标准是与输入列表的相似度
            simi_list = self.model.most_similar(positive=characters, topn=4750)
            l = []
            count = 0
            # 拓展为四个关键字
            for i in range(4000):
                if simi_list[i][0] in poem_Data.key_word:
                    l.append(simi_list[i][0])
                    count += 1
                if count > 3:
                    break
            for i in range(3):
                res.append(l[i])
        elif len(characters) == 2:
            simi_list = self.model.most_similar(positive=characters, topn=4750)
            l = []
            count = 0
            for i in range(4000):
                if simi_list[i][0] in poem_Data.key_word:
                    l.append(simi_list[i][0])
                    count += 1
                if count > 2:
                    break
            for i in range(2):
                res.append(l[i])
        elif len(characters) == 3:
            simi_list = self.model.most_similar(positive=characters, topn=4750)
            l = []
            count = 0
            for i in range(4000):
                if simi_list[i][0] in poem_Data.key_word:
                    l.append(simi_list[i][0])
                    count += 1
                if count > 1:
                    break
            res.append(l[0])
        return res


# 整理字向量库并用来初始化embedding,并初始化停用词表
stop_word=[]
file = open('stop_words.txt', 'r', encoding='utf-8')
for line in file:
    stop_word.append(line[0])
c = charvec().model
word_vectors = np.zeros(shape=[len(poem_Data.word_ID) + 1, 100], dtype=np.float32)
# print("word_vectors.shape", word_vectors.shape)
l = list(poem_Data.word_ID.keys())
# print(l[6])
# print(c[l[6]])
for i in range(len(poem_Data.word_ID)):
    tmp = c[l[i]]
    word_vectors[i] = tmp


# 生成训练用到的三个数据集
def generateBatch(istrain=True, reload=True):
    if istrain:
        poemsVector = poem_Data.trainVector
    else:
        poemsVector = poem_Data.testVector
    # 因为训练集数据加载的比较慢(5min左右),所以预先处理好用到时只需加载
    if reload:
        X = np.load(path_x)
        Y = np.load(path_y)
        Z = np.load(path_z)
        return X, Y, Z
    # random.shuffle(poemsVector)
    batchNum = (len(poemsVector) - 1) // batch_size
    # X为输入诗句向量，Y为输出，Z为关键字的标签集
    X = []
    Y = []
    Z = []  # 关键词标签
    for i in range(batchNum):
        print(i)
        batch = poemsVector[i * batch_size: (i + 1) * batch_size]
        length = 26
        temp = np.zeros((batch_size, length), dtype=np.int32)
        temp2 = np.zeros((batch_size, 4), dtype=np.int32)
        for j in range(batch_size):
            temp[j] = batch[j]
            for k in range(len(temp[j])):
                if temp[j][k] == 4183:
                    temp[j][k] = 10
            # 用textrank进行关键字的提取
            l = []
            for ii in temp[j]:
                word = poem_Data.word_numtoID[ii]
                if word=='[' or word==']' or word==',' or word=='。' or word == '，':
                    continue
                else:
                    l.append(ii)
            for iii in range(len(l)):
                l[iii] = poem_Data.word_numtoID[l[iii]]
            #print(j)
            #if j == 24:
               # r = 0
            res = textrank(l)
            for jj in range(len(res)):
                res[jj] = poem_Data.word_ID[res[jj]]
            temp2[j] = res
        X.append(temp)
        Z.append(temp2)
        temp2 = np.copy(temp)
        temp2[:, :-1] = temp[:, 1:]
        # 保存数据集
        Y.append(temp2)
        x = np.array(X)
        y = np.array(Y)
        z = np.array(Z)
        np.save(path_x, x)
        np.save(path_y, y)
        np.save(path_z, z)
    return x, y, z


# 参数为20个汉字
def textrank(l):
    l_num = []
    for i in l:
        l_num.append(poem_Data.word_ID[i])
    l_vec = np.zeros(shape=[20, 100], dtype=np.float32)
    for i in range(20):
        l_vec[i] = word_vectors[l_num[i]]
    # 相似矩阵
    sim_mat = np.zeros(shape=[20, 20], dtype=np.float32)
    for i in range(20):
        for j in range(20):
            if i != j:
                sim_mat[i][j] = cosine_similarity(l_vec[i].reshape(1,100), l_vec[j].reshape(1,100))[0][0]
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    rank_s = sorted(scores.items(), key=lambda d: d[1], reverse=True)
    res = []
    count = 0
    for i in range(20):
        if count >= 4:
            break
        tmp = rank_s[i][0]
        che = False
        for j in res:
            if l[j] == l[tmp]:
                che = True
        if l[tmp] in stop_word:
            che = True
        if che:
            continue
        else:
            res.append(rank_s[i][0])
            count += 1
    res = sorted(res, reverse=False)
    if len(res) == 3:
        res.append(res[2])
    elif len(res) == 2:
        res.append(res[1])
        res.append(res[1])
    for i in range(4):
        res[i] = l[res[i]]
    return res


if __name__ == '__main__':
    l = list('空山不见人但闻人语响返景入深林复照青苔上')
    print(textrank(l))