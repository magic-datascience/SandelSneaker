#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 12:47:46 2020

@author: connorhall

The objective of this script is predict whether a picture is a shoe or a sandel.  It should predict
1 if it's a Sandel and 0 if it's a Shoe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_curve as rc
import sklearn.model_selection as skms
import sklearn.svm as svm
import sklearn.preprocessing as pre
import copy

xtrain = pd.read_csv('data_sneaker_vs_sandal/x_train.csv')
xtest = pd.read_csv('data_sneaker_vs_sandal/x_test.csv')
ytrain = pd.read_csv('data_sneaker_vs_sandal/y_train.csv')
ytrain = ytrain.squeeze()

# #####################  Feature Transformations!  ######################################

xtrain = xtrain.to_numpy()

# add all the pixels together in one row of the image and then square that value- then take the max value
# hopefully this can indicate if there are any long bright white strips going though the image with is more 
# common in shoes
whitestrip_detect_cols = np.zeros((len(ytrain),28))

for i in range(0,784,28):
    
    whitestrip_detect_cols[:,int(i/28) ] = np.square(np.sum(xtrain[:,i:i+28], axis = 1))
    
whitestrip_detect_col = np.max(whitestrip_detect_cols, axis = 1)

whitestrip_detect_col = whitestrip_detect_col.reshape(-1, 1)

#normalize
norm = pre.Normalizer(norm = 'max')

horz_whitestrip_detect_col = norm.transform(whitestrip_detect_col.T).T

# Calculate Column for the average value of all pixels in each photo

mean_col = xtrain.mean(axis = 1)

mean_col = np.expand_dims(mean_col ,axis = 1)

## White area detecter 

ident = np.ones(5)
out = []
for row in xtrain:
    pic_values = []
    row = np.reshape(row,(28,28))
    
    for i in range(24):
        for j in range(24):
            
            pic_values.append(np.sum(ident * row[i:i+5,j:j+5]))
            
    out.append(np.max(pic_values))

out = np.asarray(out).reshape(-1,1)
white_area_detect = norm.transform(out.T).T

# max height see below for test
xheight = 784 - ((xtrain!=0).argmax(axis=1))
xheight = np.expand_dims(xheight ,axis = 1)

#maxspread

xflip = np.flip(xtrain, axis = 1)
xspread =  784 -((xflip!=0).argmax(axis=1)) - ((xtrain!=0).argmax(axis=1))
xspread = np.expand_dims(xspread ,axis = 1)

##Tall Short binary:  1(aiming for sandels) if it's tall or short, 
##                    0 (aimming for shoes) if it's medium 
#                       (roughly 80% threshold)

small = (xspread < 290)
tall = (xspread > 480)
BinaryTallShort = (small * 1) + (tall * 1)


#finally add the extra columns
xtrain = np.concatenate((xtrain
                        ,mean_col
                        ,horz_whitestrip_detect_col
                        ,white_area_detect
                        ,xheight
                        ,xspread
                        ,BinaryTallShort)
                        ,axis = 1)


###########################################################  
c_opts = [.01 , .1, 1, 10, 100, 1000]
score_c = dict()
averages_c = [] 

for c in c_opts:
    svc = svm.SVC(C= c)
    
    scores = skms.cross_validate(svc, xtrain, ytrain, cv=5,
                                  scoring=('accuracy'),
                                  return_train_score=False)
    score_c[c] = scores  

    averages_c.append(np.average(score_c[c]['test_score']))
    
    # the winner is c = 10 or above 
##############################################################################
   
kern = ['linear', 'poly', 'rbf', 'sigmoid'] 

score_kern = dict()
averages_kern = []

for k in kern:
    svc = svm.SVC(C= 10 , kernel = k)
    
    scores = skms.cross_validate(svc, xtrain, ytrain, cv=5,
                                  scoring=('accuracy'),
                                  return_train_score=False)
    
    score_kern[k] = scores
      
    averages_kern.append(np.average(score_kern[k]['test_score']))   
    
    The winner is 'rbf'
########################################################################

#ROC curves

ind_val = random.sample(range(12000), 2000)

xvalid = xtrain.iloc[ind_val,:]
yvalid = ytrain.iloc[ind_val]

xtrain = xtrain.drop(ind_val)
ytrain = ytrain.drop(ind_val)

xtrain = xtrain.to_numpy()

svc = svm.SVC(C = 10, probability=True)
svc.fit(xtrain, ytrain)
y_score = svc.predict_proba(xvalid)
fpr_svm, tpr_svm, thresholds_svm = rc(yvalid, y_score[:,1])

y_score = svc.predict_proba(xtrain)
fpr_svm_tr, tpr_svm_tr, thresholds_svm = rc(ytrain, y_score[:,1])

plt.figure(0)
plt.title('ROC Curves Heldout Set')
plt.plot(fpr_svm, tpr_svm, label = 'Support Vector Machine')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.savefig('ROC_Curves.jpg') 

###########################################################################################3
# Print out a few examples of photos that we are misclasified by my model in order to learn from

yhva = svc.predict(xvalid)

ytt = copy.deepcopy(yvalid)
ytt[ytt == 0] = 999
    
ytf = copy.deepcopy(yvalid)
ytf[ytf == 1] = 888
    
ypre_t = copy.deepcopy(yhva)
ypre_t[ypre_t == 0] = 777
    
ypre_f = copy.deepcopy(yhva)
ypre_f[ypre_f == 1] = 666
    
FP = (ypre_t-ytf) == 1

FN = (ytt - ypre_f)==1

xfalse_positive = xvalid[FP]
yfp = yvalid[FP]

xfalse_negative = xvalid[FN]
yfn = yvalid[FN]

fp_ind = random.sample(range(len(xfalse_positive)),9)
fn_ind = random.sample(range(len(xfalse_negative)),9)

fig, axes = plt.subplots(
            nrows=3, ncols=3,
            figsize=(3 * 3, 3 * 3))

for ii, row_id in enumerate(fp_ind):

    cur_ax = axes.flatten()[ii]
    
    x = xfalse_positive.iloc[row_id,:]
    x = np.asarray(x)
    
    cur_ax.imshow(x.reshape(28,28), 
                  interpolation='nearest', 
                  vmin=0, vmax=1, cmap='gray')
    
    cur_ax.set_xticks([])
    
    cur_ax.set_yticks([])
    
    cur_ax.set_title('y=%d' % yfp.iloc[row_id])

plt.savefig("Shoes we thought were Sandels.jpg")
plt.show()

fig, axes = plt.subplots(
            nrows=3, ncols=3,
            figsize=(3 * 3, 3 * 3))


for ii, row_id in enumerate(fn_ind):

    cur_ax = axes.flatten()[ii]
    
    x = xfalse_negative.iloc[row_id,:]
    x = np.asarray(x)
    
    cur_ax.imshow(x.reshape(28,28), 
                  interpolation='nearest', 
                  vmin=0, vmax=1, cmap='gray')
    
    cur_ax.set_xticks([])
    
    cur_ax.set_yticks([])
    
    cur_ax.set_title('y=%d' % yfn.iloc[row_id])

plt.savefig("Sandels's we thought were Shoes.jpg")
plt.show()

#########################################################################################
## Final Prediction Output

xtrain = pd.read_csv('data_sneaker_vs_sandal/x_train.csv')
xtest = pd.read_csv('data_sneaker_vs_sandal/x_test.csv')
ytrain = pd.read_csv('data_sneaker_vs_sandal/y_train.csv')
ytrain = ytrain.squeeze()

svc = svm.SVC(C= 10 , kernel = 'rbf')
svc.fit(xtrain, ytrain)
y_score_final = svc.predict_proba(xtest)

    
