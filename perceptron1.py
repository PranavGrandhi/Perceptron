# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:00:15 2019

@author: Pranav Grandhi
"""

import numpy as np
import pandas as pd
import matplotlib as plt

class Perceptron:
    
    def __init__(self, no_features, learning_rate = 2, epochs = 1, bias = 0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = bias
        self.weight_mat = np.random.randint(-1,1,size = no_features)
        
    def predict(self, feature):
        answer = np.dot(self.weight_mat, feature) + self.bias
        
        if answer >= 0:
            return 1
        elif answer < 0:
            return -1
    
    def fit(self, features, T):
        for i in range(self.epochs):
            for xn,t in zip(features, T):
                predict = self.predict(xn)
                if predict != t:
                    if t == 1:
                        self.weight_mat[0] += self.learning_rate * xn[0]
                        self.weight_mat[1] += self.learning_rate * xn[1]
                    elif t == -1:
                        self.weight_mat[0] -= self.learning_rate * xn[0]
                        self.weight_mat[1] -= self.learning_rate * xn[1]
            plt.pyplot.scatter(features[:,0],features[:,1],c=T)
            x = np.linspace(-4,4,100)
            plt.pyplot.plot(x,(-x * (self.weight_mat[0]/self.weight_mat[1])) + self.bias)
                        
    def no_misclassification(self,features,T):
        count = 0 ;
        for xn,t in zip(features, T):
            predict = self.predict(xn)
            if predict != t:
                count += 1
        
        return count
        

data = pd.read_csv('dataset_1.csv')

data = data.drop('0', axis = 1)
data = np.asarray(data)

features = data[:, [0,1]]
T = data[:, 2]

for k in data:
    if k[2] == 0:
        k[2] = -1
        
perceptron = Perceptron(2)

print("No of misclassifications initially =", perceptron.no_misclassification(features, T))
plt.pyplot.scatter(features[:,0],features[:,1],c=T)
x = np.linspace(-4,4,100)
slope = (perceptron.weight_mat[0]/perceptron.weight_mat[1])
plt.pyplot.plot(x,(-x * slope) + perceptron.bias)

perceptron.fit(features,T)  
print("No of misclassifications after fitting =", perceptron.no_misclassification(features, T))    

perceptron.fit(features,T)  
print("No of misclassifications after fitting =", perceptron.no_misclassification(features, T))
    
perceptron.fit(features,T)  
print("No of misclassifications after fitting =", perceptron.no_misclassification(features, T))