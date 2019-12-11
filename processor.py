#!/usr/bin/env python
# coding: utf-8
# @Author  : Mr.K
# @Software: PyCharm Community Edition
# @Time    : 2019/11/18 14:19
# @Description:

import os
import gensim
import jieba
import numpy as np
from para_setting import *
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from para_setting import *


def info_data_read(filepath):#读取语料
    with open(filepath,mode='r',encoding='utf-8') as f:
        s=f.read()
        s1=s.split()
        lines=[]
        for i in s1:
            lines.append(i)
        return lines


def term2ele(each_list):#将词拆为字
    temp_list=[]
    for each_term in each_list:
        for each_ele in each_term:
            temp_list.append(each_ele)
    return temp_list


def word_2_chara(list):#输入的是词语list，每一个词语为一个单位
    temp=[]
    for each in list:
        temp.append(term2ele(each))
    return  temp

class cls_model(object):
    def __init__(self,flag,data):
        self.flag=flag
        self.data=data

    def split_train_test_function(self,ration):
        total_num = len(self.data)
        train_num = int((total_num / 2) * ration)
        # test_num=int(total_num-train_num)
        train_x = []
        train_y = []
        text_x = []
        text_y = []
        print('处理语料总量为', len(self.data))
        print('训练集数量为', train_num * 2)
        for i in range(0, int(total_num / 2)):
            if i <= train_num - 1:
                train_x.append(self.data[i])
                train_y.append(0)
            if i > train_num - 1:
                text_x.append(self.data[i])
                text_y.append(0)
        for i in range(int(total_num / 2), total_num):
            if i <= train_num + (total_num / 2) - 1:
                train_x.append(self.data[i])
                train_y.append(1)
            if i > train_num + (total_num / 2) - 1:
                text_x.append(self.data[i])
                text_y.append(1)
        self.x_train=train_x
        self.y_train=train_y
        self.x_test=text_x
        self.y_test=text_y
        # return self.x_train,self.x_test,self.y_train,self.y_test

    def train(self):
        """
        训练函数
        :return:
        """
        if self.flag=='SVM':
            classifier = OneVsRestClassifier(SVC(kernel=kernel_func, probability=True, C=1.0, random_state=0, gamma=0.2))
            classifier.fit(self.x_train, self.y_train)
            self.score = classifier.decision_function(self.x_test)
            print('模型已保存到%s'%(model_save_path))
            joblib.dump(classifier, model_save_path)
        if self.flag=='RF':
            clf = RandomForestClassifier()
            clf.fit(self.x_train, self.y_train)
            print('模型已保存到%s'%(model_save_path))
            joblib.dump(clf, model_save_path)
        if self.flag=='NB':
            clf = GaussianNB()
            clf.fit(self.x_train, self.y_train)
            print('模型已保存到%s'%(model_save_path))
            joblib.dump(clf, model_save_path)
        if self.flag == 'DT':
            clf = DecisionTreeClassifier()
            clf.fit(self.x_train, self.y_train)
            print('模型已保存到%s'%(model_save_path))
            joblib.dump(clf,model_save_path)
        if self.flag == 'LR':
            clf = LogisticRegression()
            clf.fit(self.x_train, self.y_train)
            self.score = clf.decision_function(self.x_test)
            print('模型已保存到%s'%(model_save_path))
            joblib.dump(clf, model_save_path)
        if self.flag == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(self.x_train, self.y_train)
            print('模型已保存到%s'%(model_save_path))
            joblib.dump(clf, model_save_path)

    def evaluate(self):
        """
        评估函数
        :return:
        """
        clf=joblib.load(model_save_path)
        result_set = [clf.predict(self.x_test), self.y_test]
        temp_path1 = os.path.join(out_put_path, 'eva_true_lable.txt')#真实值
        temp_path2 = os.path.join(out_put_path, 'eva_predict_lable.txt')#模型预测值（用于评价模型）
        np.savetxt(temp_path1, result_set[1], fmt='%.4e')
        np.savetxt(temp_path2, result_set[0], fmt='%.4e')
        print('%s分类器评价指标如下：'%(self.flag))
        print('Accuracy:\t', accuracy_score(result_set[1], result_set[0]))
        print('Precision:\t', precision_score(result_set[1], result_set[0]))
        print('Recall:\t', recall_score(result_set[1], result_set[0]))
        print('f1 score:\t', f1_score(result_set[1], result_set[0]))

    def plot_roc(self):
        num_Y_test = len(self.y_test)
        Y_test = np.array(self.y_test)
        Y_test = Y_test.reshape(num_Y_test, 1)
        ROC_curve_plot(Y_test, self.score)

    def predict(self):
        classifier = joblib.load(model_save_path)
        y_score = classifier.decision_function(self.data)
        result_set = [classifier.predict(self.data), y_score]
        #result_set = [classifier.predict(self.data)]#有的分类器不支持decision_function方法
        temp_path3 = os.path.join(out_put_path, 'app_predict_result.txt')#实际使用时预测的结果
        temp_path4 = os.path.join(out_put_path, 'app_predict_prob.txt')#实际使用时预测的概率
        np.savetxt(temp_path3, result_set[0], fmt='%.4e')
        np.savetxt(temp_path4, result_set[1], fmt='%.4e')
        print('预测结果已保存到目录%s下' % (temp_path3))
        print('预测结果概率已保存到目录%s下' % (temp_path4))




def ROC_curve_plot(y_test, y_score):
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    #plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()




