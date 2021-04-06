# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:24:38 2021

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np
#from config import config

def plot(train_ious,test_ious):
    x = np.arange(len(train_ious))
    plt.plot(x,train_ious,label='Train loss', c='r')
    plt.plot(x,test_ious,label='Test loss', c='b')
    #plt.axhline(config["threshold loss"],0,len(Loss_train),label='loss threshold',c='k')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()