#!/usr/bin/env python
# coding: utf-8
# @Author  : Mr.K
# @Software: PyCharm Community Edition
# @Description: 主函数：直接使用维基语料的word2vec模型进行字嵌入、词向量加权求平均计算词向量、SVM分类器训练
#数据0类是添加剂，1类是食品

import time
from w2v_embedding import *
from processor import *
import numpy as np

time_start = time.time()#计时
print('开始第一步处理...')
#step1:读取语料
rawdata=info_data_read(text_path)#读取原始语料
rawdata_ele=word_2_chara(rawdata)#将原始语料中的每一个词拆为字
print(rawdata_ele)#一共1034个数据：[['吊', '白', '块'], ['腐', '竹'], ...，['粉', '丝'], ['竹', '笋']]


print('开始第二步处理....')
#setp2:将正负数据集进行词嵌入获取向量
result_list = []#词向量集合，里面每一个都是由字向量相加平均后的词向量
for each in rawdata_ele:
    result_list.append(new_function(each))
new_vectors_list=np.array(result_list)#将数据格式转为ndarray



print('开始第三步处理....')
#step3:划分训练集测试机，训练分类器并测试
# svm=cls_model('SVM',new_vectors_list)#实例化支持分类的类
# svm.split_train_test_function(0.6)#将数据集分为训练集与测试集合
# svm.train()#训练模型,可在path.py中设置SVM分类器核函数:'poly''rbf''linear''sigmoid'
# svm.evaluate()#评价模型
# svm.plot_roc()#绘制roc曲线

#以下为其他分类器代码
lr=cls_model('LR',new_vectors_list)#逻辑回归
lr.split_train_test_function(0.5)#将数据集分为训练集与测试集合
lr.train()#训练模型
lr.evaluate()#评价模型
# lr.plot_roc()#绘制roc曲线

# nb=cls_model('NB',new_vectors_list)#朴素贝叶斯
# nb.split_train_test_function(0.6)#将数据集分为训练集与测试集合
# nb.train()#训练模型
# nb.evaluate()#评价模型


# dt=cls_model('DT',new_vectors_list)#决策树
# dt.split_train_test_function(0.5)#将数据集分为训练集与测试集合
# dt.train()#训练模型
# dt.evaluate()#评价模型


# knn=cls_model('KNN',new_vectors_list)#K近邻
# knn.split_train_test_function(0.5)#将数据集分为训练集与测试集合
# knn.train()#训练模型
# knn.evaluate()#评价模型


# rf=cls_model('RF',new_vectors_list)#随机森林
# rf.split_train_test_function(0.5)#将数据集分为训练集与测试集合
# rf.train()#训练模型
# rf.evaluate()#评价模型


time_end = time.time()
a=(time_end - time_start) / 60
print('任务完成，总共耗时为：%f分钟'%(a))

