# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 12:45:05 2021

@author: Admin
"""
import torch
from load_data import load_data
from learning_function import learning_function
from torchsummary import summary
from plot import plot
from Unet import UNet
from mean_teacher import MT
from config import config
import transform

#####################################################################################################
######################################## load data ##################################################
#####################################################################################################
l_train = load_data("data", "l_train")
u_train = load_data("data", "u_train")
test    = load_data("data", "test")

#####################################################################################################
################################## transformation  ##################################################
#####################################################################################################
transform_fn = transform.transform(*config["transform"])

#####################################################################################################
#################################### student model ##################################################
#####################################################################################################
S_model = UNet(2,transform_fn)
#summary(S_model, (3, 480 ,640))


#####################################################################################################
#################################### Teacher model ##################################################
#####################################################################################################
T_model = UNet(2,transform_fn)
T_model.load_state_dict(S_model.state_dict())
ssl_obj = MT(T_model, config["ema_factor"])


Loss_train,Loss_test = learning_function(S_model,ssl_obj,l_train,test, u_train)

plot(Loss_train,Loss_test)