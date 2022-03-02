import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
#from fastai.vision.widgets import *
from fastai.vision import load_learner, open_image
# from fastbook import *
from pathlib import Path
import numpy as np
import streamlit as st
from gradcam import *
st.title("SmartVision")
st.header("AI Image Scanner system for Nudity Image Detection")
# For newline
st.write('\n')

class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(*os.path.split(filename))
        self.img = self.get_image_from_upload()
        
        if self.img is not None:
            self.display_output()
            self.get_prediction()

    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Image",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return uploaded_file
        return None

    def display_output(self):
        st.image(self.img, caption='Uploaded Image')

    def get_prediction(self):
        def softmax(x):
            e_x=np.exp(x-np.max(x))
            return e_x/e_x.sum(axis=0)

        if st.button('Classify'):
            print('here :', self.img.name)
            img1=open_image(self.img)
            pred, pred_idx, probs = self.learn_inference.predict(img1)
            
            classes=self.learn_inference.data.classes
            tmp={}
            softmaxed=softmax(probs.numpy())
            
            for o in softmaxed.argsort()[-3:][::-1]:
                tmp.update({classes[o]:softmaxed[o].item()})
            
            pred=max(tmp, key=lambda k: tmp[k])
            st.write(f'Prediction: {tmp}')
            
            gcam = GradCam.from_one_img(self.learn_inference,img1)
            gcam.plot(plot_hm=True,plot_gbp=False)
            st.image('gradcam.png', caption='Grad-CAM')
        else: 
            st.write(f'Click the button to classify') 

if __name__=='__main__':

    file_name='10731_model.pkl'

    predictor = Predict(file_name)
    
    
