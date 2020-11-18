# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:38:29 2020

@author: PC
"""


from keras.backend import clear_session

 

# 销毁当前TF图并创建一个新的TF图，避免旧模型/图层的混乱。

clear_session()