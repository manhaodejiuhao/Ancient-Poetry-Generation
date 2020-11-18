def clarify_yayun():
    with open("字表拼音.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        zhengti = ["hi", "ri", "zi", "ci", "si"]  # 整体认读音节的后两个字母
        u_to_v = ["yu", "xu", "qu"]  # 读作v写作u
        for line in lines:
            c = line[0]
            # _i指末尾i个字母组成的韵母
            _1 = line[-2]
            _2 = line[-3: -1]
            _3 = line[-4: -1]
            if _1 == "a":
                yun_mu[1].append(c)
            elif _2 == "ai":
                yun_mu[2].append(c)
            elif _2 == "an":
                yun_mu[3].append(c)
            elif _3 == "ang":
                yun_mu[4].append(c)
            elif _2 == "ao":
                yun_mu[5].append(c)
            elif _1 == "o" or (_1 == "e" and _2 != "ie" and _2 != "ye" and _2 != "ue"):
                yun_mu[6].append(c)
            elif _2 == "ei" or _2 == "ui":
                yun_mu[7].append(c)
            elif _2 == "en" or _2 == "in" or _2 == "un":
                yun_mu[8].append(c)
            elif _3 == "eng" or _3 == "ing" or _3 == "ong":
                yun_mu[9].append(c)
            elif (_1 == "i" and _2 not in zhengti) or _2 == "er":
                yun_mu[10].append(c)
            elif _1 == "i" and _2 in zhengti:
                yun_mu[11].append(c)
            elif _2 == "ie" or _2 == "ye":
                yun_mu[12].append(c)
            elif _2 == "ou" or _2 == "iu":
                yun_mu[13].append(c)
            elif _1 == "u" and _2 not in u_to_v:
                yun_mu[14].append(c)
            elif _1 == "v" or _2 in u_to_v:
                yun_mu[15].append(c)
            elif _2 == "ue":
                yun_mu[16].append(c)
        sum = 0
        for lst in yun_mu.values():
            sum += len(lst)


'''def output():
    with open("押韵分类结果.txt", "w", encoding="utf-8") as file:
        for i in range(1, 17):
            file.write(str(i))
            file.write(": ")
            for c in yun_mu[i]:
                file.write(c)
                file.write(" ")
            file.write("\n")'''


def get_yun(c):  # 返回所属韵的类别
    for i in range(1, 17):
        if c in yun_mu[i]:
            return i
    return 0

yun_mu = {i: [] for i in range(1, 17)}  # 韵母
clarify_yayun()


if __name__ == "__main__":
    clarify_yayun()
    # output()
