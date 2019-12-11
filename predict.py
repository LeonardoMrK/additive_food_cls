#!/usr/bin/env python
# coding: utf-8
# @Author  : Mr.K
# @Software: PyCharm Community Edition
# @Description: 使用训练好的分类模型预测

import time
from processor import *
from para_setting import *
from w2v_embedding import *
import numpy as np



time_start = time.time()#计时
print('开始第一步处理...')
#step1:读取语料
rawdata=info_data_read(text_path2)#读取待分类语料
rawdata_ele=word_2_chara(rawdata)#将待分类语料中的每一个词拆为字
print(rawdata_ele)


print('开始第二步处理....')
#setp2:将正负数据集进行词嵌入获取向量
result_list = []#词向量集合，里面每一个都是由字向量相加平均后的词向量
for each in rawdata_ele:
    result_list.append(new_function(each))
new_vectors_list=np.array(result_list)#将数据格式转为ndarray



print('开始第三步处理....')
#step3:调用模型，对数据进行分类




# svm=cls_model('SVM',new_vectors_list)#实例化支持分类的类
# svm.predict()


lr=cls_model('LR',new_vectors_list)#逻辑回归
lr.predict()

# nb=cls_model('NB',new_vectors_list)#朴素贝叶斯
# nb.predict()


# dt=cls_model('DT',new_vectors_list)#决策树
# dt.predict()


# knn=cls_model('KNN',new_vectors_list)#K近邻
# knn.predict()


# rf=cls_model('RF',new_vectors_list)#随机森林
# rf.predict()

time_end = time.time()
a=(time_end - time_start) / 60
print('任务完成，总共耗时为：%f分钟'%(a))











