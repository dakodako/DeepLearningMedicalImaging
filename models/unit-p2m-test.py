#%%
from keras.layers import ZeroPadding2D, BatchNormalization, Input, MaxPooling2D, AveragePooling2D, Conv2D, LeakyReLU, Flatten, Conv2DTranspose, Activation, add, Lambda, GaussianNoise, merge, concatenate, Dropout, InputSpec, Layer
from keras import initializers, regularizers, constraints
from keras import backend as K 
from keras.models import Model, load_model
from keras.layers.core import Dense, Flatten, Reshape
from keras.optimizers import Adam, adam 
from keras.activations import tanh 
from keras.regularizers import l2 
from keras.initializers import RandomNormal 
import nibabel as nib 
from PIL import Image 
#from tensorflow.contrib.kfac.python.ops import optimizer
from collections import OrderedDict 
from time import localtime, strftime 
#from scipy.misc import imsave, toimage 
import numpy as np 
import json 
import sys 
import time 
import datetime 
from glob import glob
import os
