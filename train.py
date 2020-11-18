# coding: UTF-8
from parameter import *
from rnn_model import *
from wordvec import*
from yayun import *


def train(traindata, reload = True):
    # 初始化计算图
    lstm_model = model(traindata, batch_size * 4)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # 进行模型的加载
        if not os.path.exists(checkpointsPath):
            os.mkdir(checkpointsPath)

        if reload:
            checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
            # 如果有模型就加载，没有就从头训练
            if checkPoint and checkPoint.model_checkpoint_path:
                saver.restore(sess, checkPoint.model_checkpoint_path)
                print("restored %s" % checkPoint.model_checkpoint_path)
            else:
                print("no checkpoint found!")
        # 开始训练

        for epoch in range(epochNum):
            X, Y, Z = generateBatch()
            epochSteps = len(X)  # equal to batch
            for step, (x, y, z) in enumerate(zip(X, Y, Z)):
                # 重新处理数据集，变成5*100向量形式
                batch_s = len(x)
                new_x = np.zeros(shape=[4*batch_s, 6], dtype=np.int)
                new_y = np.zeros(shape=[4 * batch_s, 6], dtype=np.int)
                new_z = np.zeros(shape=[4 * batch_s, 4], dtype=np.int)
                for i in range(batch_s):
                    k = i*4
                    new_x[k] = x[i][0:6]
                    new_x[k + 1][1:6] = x[i][7:12]
                    new_x[k + 1][0] = poem_Data.word_ID['[']
                    new_x[k + 2] = x[i][12:18]
                    new_x[k + 3][1:6] = x[i][19:24]
                    new_x[k + 3][0] = poem_Data.word_ID['[']
                    new_z[k] = z[i]
                    new_z[k + 1] = z[i]
                    new_z[k + 2] = z[i]
                    new_z[k + 3] = z[i]
                new_y[:, :-1] = new_x[:, 1:]
                a, loss, gStep = sess.run([lstm_model.trainOP, lstm_model.cost, lstm_model.addGlobalStep],
                                          feed_dict={lstm_model.gtX: new_x, lstm_model.gtY: new_y, lstm_model.gtZ:new_z})
                print("epoch: %d, steps: %d/%d, loss: %3f" % (epoch + 1, step + 1, epochSteps, loss))
                if gStep % saveStep == saveStep - 1:  # prevent save at the beginning
                    print("save model")
                    saver.save(sess, os.path.join(checkpointsPath, type), global_step=gStep)


# 根据映射的概率提取相应的字
def probsToWord(weights, words, rhythm=0):
    # ratio是任取的，在1~0均匀分布，出现在概率大的区间的概率也会很大
    prefixSum = np.cumsum(weights)  # prefix sum
    ratio = np.random.rand(1)
    index = np.searchsorted(prefixSum, ratio * prefixSum[-1])
    if index[0] >= len(words):
        index[0] = np.random.rand(len(words) - 1)[0]
    index = index[0]
    if rhythm == 0:
        return words[index]
    else:
        weight_dict = {i: weights[0][i] for i in range(len(weights[0]))}
        word_dict = sorted(weight_dict.items(), key=lambda d:d[1], reverse=True)
        sorted_words = [w[0] for w in word_dict]
        for w in sorted_words:
            ww = words[w]
            k = get_yun(ww)
            if k == 0:
                continue
            elif k == rhythm:
                return ww
        return words[index]


# 检查是否为五言绝句，删除非五言绝句
def examine_poems(poems, generate_num):
    wrong_id = []  # 错误诗的序号
    for i in range(generate_num):
        poem = poems[i]
        if len(poem) != 28:
            wrong_id.append(i)
            continue
        for j in [x * 7 for x in range(4)]:
            for k in range(5):
                if poem[j + k] in ['，', '。', '\n']:  # 是否是字
                    wrong_id.append(i)
                    continue
            separate = {0: '，', 1: '。'}
            if poem[j + 5] != separate[j // 7 % 2]:  # 标点是否正确
                wrong_id.append(i)
                continue
            if poem[j + 6] != '\n':  # 是否正确换行
                wrong_id.append(i)
    right_poems = []
    for i in range(generate_num):
        if i in wrong_id:
            continue
        right_poems.append(poems[i])
    generate_num = len(right_poems)
    return right_poems, generate_num


# characters是关键字对应数字列表
def generate(traindata, characters):
    print("genrating...")
    poems = []
    rhythm = 0
    for ki in range(4):
        print("ki :", ki)
        tf.reset_default_graph()
        # 初始化计算图
        lstm_model = model(traindata, 1)
        # 检索关键字
        '''for i in characters:
            if not (i in poem_Data.key_word_num):
                print('输入的关键词不在关键词库中')
                return None'''
        _characters = np.array([characters])
        with tf.Session() as sess:
            # 加载训练好的模型
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
            if checkPoint and checkPoint.model_checkpoint_path:
                saver.restore(sess, checkPoint.model_checkpoint_path)
                print("restored %s" % checkPoint.model_checkpoint_path)
            else:
                print("no checkpoint found!")
                exit(1)
            # 先计算出第一个状态
            state = sess.run(lstm_model.stackCell.zero_state(1, tf.float32))
            # x为feed
            x = [[lstm_model.traindata.word_ID['[']]]
            print("feed:", x)

            for i in range(generateNum):
                print("开始生成第", i, "首诗")
                # 根据上一个状态、attention与这时的输入计算出输出的概率，由此找到对应的词，进行生成诗句
                probs1, state = sess.run([lstm_model.probs, lstm_model.finalState], feed_dict={lstm_model.gtX: x,
                                                                                               lstm_model.gtZ: _characters,
                                                                                               lstm_model.initState: state})

                word = probsToWord(probs1, lstm_model.traindata.word_vca)    # 根据映射的概率提取相应的字
                poem = ''
                num = 0
                print("probs1", probs1)
                print(probs1.shape)
                print("word", word)

                while word not in [' ', ']']:
                    poem += word
                    num += 1
                    if num >= 5:
                        break

                    x = [[lstm_model.traindata.word_ID[word]]]
                    # print(word)
                    probs2, state = sess.run([lstm_model.probs, lstm_model.finalState], feed_dict={
                        lstm_model.gtX: x, lstm_model.gtZ: _characters, lstm_model.initState: state})

                    if num == 4 and ki % 2 == 1:
                        word = probsToWord(probs2, lstm_model.traindata.word_vca, rhythm)
                        print("word.No1", word, num)
                        r = 0
                    elif num == 4 and ki == 0:
                        word = probsToWord(probs2, lstm_model.traindata.word_vca)
                        print("word.No2", word, num)
                        rhythm = get_yun(word)  # 获取韵脚
                        print("rhythm 1", rhythm)
                        while rhythm == 0:
                            word = probsToWord(probs2, lstm_model.traindata.word_vca)
                            print("word.No3", word, num)
                            rhythm = get_yun(word)
                            print("rhythm 2", rhythm)
                    else:
                        word = probsToWord(probs2, lstm_model.traindata.word_vca)
                        print("word.No4", word, num)
                poems.append(poem)
    poem = ''
    poem += poems[0]
    poem += '，'
    poem += poems[1]
    poem += '。'
    poem += '\n'
    poem += poems[2]
    poem += '，'
    poem += poems[3]
    poem += '。'
    poem += '\n'
    return poem


# 该函数根据得到的图片标签生成有关的诗句
def label_poem(label):
    # 图片对应的关键字
    key_list = label_key_dict[label]
    rand_np = np.random.randint(len(key_list), size=5)
    poem_list = []
    keywords = []
    # 生成五首诗句并存入列表最后return
    for i in range(generate_totalNum):
        print("label_poem :", i)
        characters = list(key_list[rand_np[i]])
        print("分类后的结果的字典对应中文字", characters)
        keywords.append(characters)
        key_num = []
        print("poem_Data.word_vca", poem_Data.word_vca)
        print(len(poem_Data.word_vca))
        print(len(poem_Data.word_dict))
        print("poem_Data.word_ID", poem_Data.word_ID)
        # 拓展关键字
        _characters = charvec().gene_simi_chars(characters)  # 关键字补全（输入一个列表，返回一个列表，返回的列表是意思相近的关键字）
        print("_characters是扩展后的关键字", _characters)
        for i in _characters:
            key_num.append(poem_Data.word_ID[i])
            # print("key_num", key_num)
            # print("i", i)
        # print()
        poems = generate(poem_Data, key_num)
        p = list(poems)
        p.insert(6, '\n')
        p.insert(20, '\n')
        poems = ''.join(p)
        poem_list.append(poems)
    poem_list, generate_num = examine_poems(poem_list, generate_totalNum)
    print("生成%d首诗\n" % generate_num)
    for i in range(generate_num):
        print(keywords[i])
        print(poem_list[i])
    return keywords, poem_list



def api_html(l):
    res = []
    for i in range(4):
        res.append(l[0 + i*7:6 + i*7])
    return res


if __name__ == "__main__":
    keywords, poems = label_poem("cloud")
    print(keywords, poems)

    # train(poem_Data)