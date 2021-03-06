from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import zipfile
import tqdm
import glob
import os
import sys
import shutil
from fastai.vision import *
import fastai

from fastprogress.fastprogress import force_console_behavior
#import fastprogress
#fastprogress.fastprogress.NO_BAR = True
master_bar, progress_bar = force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar


path2zip = './nude_sexy_safe_v1_x320/training/'
train_classes = ['sexy','nude','safe']
batch_size = 32
stage1_epochs = 7
stage2_epochs = 3

def get_parent_dir(f):
  return os.path.split(os.path.split(f)[0])[-1]
            
def train():

    print('number of train_classes', len(train_classes))
    np.random.seed(42)
    
    il = (ImageList.from_folder(path2zip)
          .filter_by_func(lambda f: get_parent_dir(f) in train_classes)
          .split_by_rand_pct(valid_pct=0.2, seed=42)
          .label_from_func(lambda f: get_parent_dir(f))
          .transform(tfms=get_transforms())
          )
    data = (ImageDataBunch.create_from_ll(il, bs=batch_size,
                                         size=224,
                                         num_workers=4)
            .normalize(imagenet_stats)
            )
    print(len(data.classes), data.c, len(data.train_ds), len(data.valid_ds))

    
    learn = (cnn_learner(data, models.resnet34,
                        metrics=accuracy)
             .mixup(alpha=0.2)
             )
    learn.loss_func = LabelSmoothingCrossEntropy()
    learn.fit_one_cycle(stage1_epochs, max_lr=1e-2)
    learn.save('stage-1')
    learn.unfreeze()
    learn.fit_one_cycle(stage2_epochs, max_lr=slice(1e-6,3e-6))
    learn.save('stage-2')
    learn.export('trained_model.pkl')
    
if __name__ == '__main__':
    train()
    
    
    
