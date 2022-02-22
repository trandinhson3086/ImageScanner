# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:44:36 2020

@author: trandinhson3086
"""

import sys
import os
import numpy as np

from fastai.vision import load_learner, open_image


class test_model(object):
    def __init__(self):

        #all model
        self.model=self.get_classifier("10731_model.pkl")
        
    def softmax(self,x):
        e_x=np.exp(x-np.max(x))
        return e_x/e_x.sum(axis=0)
    
    def read_image(self, impath):
        return open_image(impath)
    
    def get_classifier(self, weights=None):
        model=load_learner(*os.path.split(weights))
        return model
    
    def predict_images(self, model, img):
        pred_class, pred_idx, outputs=model.predict(img)
        classes=model.data.classes
        tmp={}
        softmaxed=self.softmax(outputs.numpy())
        
        for o in softmaxed.argsort()[-3:][::-1]:
            tmp.update({classes[o]:softmaxed[o].item()})
        
        pred=max(tmp, key=lambda k: tmp[k])
        return tmp
    
    def porn_model_test(self, imageName):
        im=self.read_image(imageName)

        
        preds=self.predict_images(self.model, im)
        
        return preds