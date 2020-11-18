# coding: UTF-8
from parameter import *
from rnn_model import *
import data
from wordvec import*


# 参数为五个字一句诗的列表+起始符
def perplexity(poem, key_word):
    res = 1
    key_words = [0, 0, 0, 0]
    tf.reset_default_graph()
    # 初始化计算图
    lstm_model = model(poem_Data, 1)
    for i in range(6):
        poem[i] = poem_Data.word_ID[poem[i]]
    for i in range(4):
        key_words[i] = poem_Data.word_ID[key_word[i]]
    key_words = np.array(key_words)
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
        x = [[lstm_model.traindata.word_ID['[']]]
        for ki in range(5):
            probs1, state = sess.run([lstm_model.probs, lstm_model.finalState], feed_dict={lstm_model.gtX: [[poem[ki]]],
                                                                                           lstm_model.gtZ: [key_words],
                                                                                       lstm_model.initState: state})
            res *= probs1[0][poem[ki + 1]] # 概率相乘
        return res


# 接受两个列表
def evaluate(keys, poems):
    score = {0: 1, 1: 1, 2: 1, 3: 1}
    for i in range(len(keys)):
        key = keys[i]
        res = 1
        for j in range(4):
            poem = '['
            poem += poems[i][0+j*7:5+j*7]
            res *= perplexity(list(poem), key)
        score[i] = res
    score = sorted(score.items(), key=lambda d: d[1], reverse=True)
    return poems[score[0][0]]



if __name__ == '__main__':
    keys = list('山峰树木')
    p1 = list('[峯然一弭苔')
    p2 = list('[迢迢夜飞来')
    p3 = list('[曾见树云处')
    p4 = list('[久断岧鹤来')
    p = []
    p.append(p1)
    p.append(p2)
    p.append(p3)
    p.append(p4)
    res = 1
    for i in range(4):
        res *= perplexity(key_word=keys, poem=p[i])
    print(res)
