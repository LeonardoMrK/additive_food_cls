#!/usr/bin/env python
# coding: utf-8
# @Author  : Mr.K
# @Software: PyCharm Community Edition
# @Time    : 2019/11/18 13:56
# @Description: #此脚本用来载入训练好(维基语料)的w2v模型，并将制定词语转化为字向量后加权平均得到词向量

import gensim
import numpy as np
from para_setting import *

#载入模型
model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path, binary=True,unicode_errors='ignore')


#输入参数是词表，返回是词向量表
def Wordvector_W2V(keywords_list):
    result_list=[]
    for each_ele in keywords_list:
        #print(each_ele)
        result_list.append(model.wv[each_ele])
    return result_list

#在上一步基础上，将每个字向量相加求平均得到词向量,维度为300
def calcu_avr_function(list):
    result_list=[]
    sum=0
    for i in range(len(list)):
        sum+=list[i]
    avr=sum/len(list)
    result_list.append(avr)
    return result_list


# b=['氯', '化','钠']#词表
# a=Wordvector_W2V(b)#先把词表中的每个字嵌入为向量
# print(a)
# print(calcu_avr_function(a))#然后在相加求平均



def new_function(ele_list):#输入是一个拆为字的词list，对每个字嵌入后相加求平均，得到词向量
    sum=0
    for i in range(len(ele_list)):#依次处理词表中的每个元素
        #print(ele_list[i])
        sum+=(model.wv[ele_list[i]])#对每个元素做嵌入，并相加
    avr=sum/len(ele_list)
    return avr

# print(new_function(b))