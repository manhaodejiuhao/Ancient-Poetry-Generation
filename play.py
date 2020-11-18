# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:47:20 2020

@author: PC
"""
from parameter import *
from wordvec import *
import data
import rnn_model
import train
import cifar100vgg as cfvg
import estimate as es


def poem(file_path):
    #print("请输入图片的路径:")
    #path = input()
    path=file_path
    label = cfvg.pic_to_label(path, show_pictures=False)
    print("label: ", label)
    keywords, poems = train.label_poem(label)
    res_poem = es.evaluate(keywords, poems)
    #print("最终的版本:\n")
    #print(res_poem)

    return res_poem
