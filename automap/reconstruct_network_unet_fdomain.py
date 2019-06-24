#%%
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.layers.core import Dropout, Lambda
from keras.layers.merge import concatenate
from keras.utils import plot_model
from time import localtime, strftime 
import datetime
from keras.callbacks import TensorBoard
#%%
import time
#from tensorflow.python.keras.callbacks import TensorBoard
#%%
import os
import numpy as np
#import scipy.misc
import numpy.random as rng 
#from PIL import Image, ImageDraw, ImageFont
#from sklearn.utils import shuffle
import nibabel as nib # python library for reading MR images
#from sklearn.model_selection import train_test_split
import random
import math
from glob import glob
from random import sample
#from matplotlib import pyplot as plt
#%%       
def unet2(input_img):
	#s = Lambda(lambda x: x/255)(input_img)
	c1 = Conv2D(32,(3,3),activation = 'relu', padding = 'same')(input_img)
	c1 = Dropout(0.1)(c1) # ????
	c1 = Conv2D(32,(3,3), activation = 'relu', padding = 'same')(c1)
	p1 = MaxPooling2D((2,2), strides = (2,2))(c1)
	c2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(p1)
	c2 = Dropout(0.1)(c2) # ????
	c2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(c2)
	p2 = MaxPooling2D((2,2), strides = (2,2))(c2)

	c3 = Conv2D(128,(3,3), activation = 'relu', padding = 'same')(p2)
	c3 = Dropout(0.1)(c3) # ????
	c3 = Conv2D(128,(3,3), activation = 'relu', padding = 'same')(c3)
	p3 = MaxPooling2D((2,2), strides = (2,2))(c3)

	c4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(p3)
	c4 = Dropout(0.1)(c4) # ????
	c4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(c4)
	p4 = MaxPooling2D((2,2), strides = (2,2))(c4)

	c5 = Conv2D(512, (3,3),activation = 'relu', padding = 'same')(p4)
	c5 = Dropout(0.1)(c5) # ????
	c5 = Conv2D(512, (3,3),activation = 'relu', padding = 'same')(c5)

	u6 = Conv2DTranspose(256,(2,2), strides = (2,2), padding = 'same')(c5)
	u6 = concatenate([u6, c4])
	c6 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(u6)
	c6 = Dropout(0.1)(c6)
	c6 = Conv2D(256,(3,3), activation = 'relu', padding = 'same')(c6)
	u7 = Conv2DTranspose(128, (3,3), strides = (2,2), padding = 'same')(c6)
	u7 = concatenate([u7,c3])
	c7 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(u7)
	c7 = Dropout(0.1)(c7)
	c7 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(c7)

	u8 = Conv2DTranspose(64, (2,2), strides = (2,2), padding = 'same')(c7)
	u8 = concatenate([u8, c2])
	c8 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(u8)
	c8 = Dropout(0.1)(c8)
	c8 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(c8)

	u9 = Conv2DTranspose(32, (2,2), strides = (2,2), padding = 'same')(c8)
	u9 = concatenate([u9, c1])
	c9 = Conv2D(32,(3,3), activation = 'relu', padding = 'same')(u9)
	c9 = Dropout(0.1)(c9)
	c9 = Conv2D(32,(3,3), activation = 'relu', padding = 'same')(c9)
	output = Conv2D(2,(1,1), activation = 'relu', padding = 'same')(c9) 
	return output

#%%
batch_size = 10
epochs = 100
inChannel = 2
x, y = 256, 256 
input_img = Input(shape = (x, y, inChannel))
#%%
reconstruct_network = Model(input_img, unet2(input_img))
reconstruct_network.compile(loss='mean_squared_error', optimizer = RMSprop(), metrics=['accuracy'])
reconstruct_network.summary()

#%%
def to_freq_space_2d(img):
    """ Performs FFT of an image
    :param img: input 2D image
    :return: Frequency-space data of the input image, third dimension (size: 2)
    contains real and imaginary part
    """
    
    img_f_original = np.fft.fft2(img)  # FFT
    img_fshift = np.fft.fftshift(img_f_original)  # FFT shift
    # img_f_flat = np.reshape(img_fshift, (np.product(img_f_original.shape),))
    # idx = sample(range(np.product(img_f_original.shape)), int(0.3 * np.product(img_f_original.shape)))
    # img_f_flat[idx] = 0
    # img_f_down= np.reshape(img_f_flat, img_f_original.shape)
    img_f_real = img_fshift.real
    img_f_imag = img_fshift.imag 
    img_f = np.dstack((img_f_real, img_f_imag))
    # img_f_down_real = img_f_down.real 
    # img_f_down_imag = img_f_down.imag
    # img_f_down = np.dstack((img_f_down_real, img_f_down_imag))

    img_f_flat_down = np.reshape(img_fshift, (np.product(img_f_original.shape), ))
    idx = sample(range(np.product(img_f_original.shape)), int(0.3*np.product(img_f_original.shape)))
    img_f_flat_down[idx] = 0
    img_f_down = np.reshape(img_f_flat_down, img_f_original.shape)
    img_f_down_real = img_f_down.real 
    img_f_down_imag = img_f_down.imag 
    img_f_down = np.dstack((img_f_down_real, img_f_down_imag))
    return img_f, img_f_down
def undersample(img_f):
    img_f_real = img_f[:,:,0]
    img_f_imag = img_f[:,:,1]
    img_f_real_flat = np.reshape(img_f_real, (np.product(img_f_real.shape),))
    idx = sample(range(np.product(img_f_real.shape)), int(0.3*np.product(img_f_real.shape)))
    img_f_real_flat[idx] = 0
    img_f_real_down = np.reshape(img_f_real_flat, img_f_real.shape)
    img_f_imag_flat = np.reshape(img_f_imag, (np.product(img_f_imag.shape),))
    img_f_imag_flat[idx] = 0
    img_f_imag_down = np.reshape(img_f_imag_flat, img_f_imag.shape)
    img_f_down = np.dstack((img_f_real_down, img_f_imag_down))
    return img_f_down
def read_image(path):
    img = nib.load(path)
    img = img.get_data()
    _,_,w = img.shape
    _w = int(w/2)
    img = img[:,:,_w:]
    img = np.squeeze(np.array(img))
    return img
def load_batch(path, batch_size):
    random.shuffle(path)
    n_batches = int(len(path)/batch_size)
    for i in range(n_batches):
        batch = path[i*batch_size:(i + 1)* batch_size]
        imgs_f, imgs_f_down = [], []
        for f in batch:
            img = read_image(f)
            img = (img - np.min(img))/(np.max(img) - np.min(img))
            img_f, img_f_down = to_freq_space_2d(img)
            img_f = (img_f - np.min(img_f))/(np.max(img_f) - np.min(img_f))
            img_f_down = (img_f_down - np.min(img_f_down))/(np.max(img_f_down) - np.min(img_f_down))
            img_f = img_f - 0.5 
            img_f_down = img_f_down - 0.5 
            #img_f_down = undersample(img_f)
            #img_f_down = undersample(img_f)
            #img_f_down = np.reshape(img_f_down, (img_f.shape[0]*img_f.shape[1], img_f.shape[2]))
            #img_f_down = img_f_down.view(dtype = np.complex128)
            #img_down = abs(np.fft.ifft2(img_f_down))
            #img_down = np.reshape(img_down, (img_f.shape[0], img_f.shape[1]))
            #img_down = np.rot90(img_down)
            #img_down = np.rot90(img_down)
            #img_down = np.rot90(img_down)
            imgs_f_down.append(img_f_down)
            imgs_f.append(img_f)
        imgs_f = np.asarray(imgs_f, dtype = float)
        imgs_f_down = np.asarray(imgs_f_down, dtype = float)
        imgs_f = np.reshape(imgs_f, (-1, imgs_f.shape[1], imgs_f.shape[2], imgs_f.shape[3]))
        imgs_f_down = np.reshape(imgs_f_down, (-1, imgs_f.shape[1],imgs_f.shape[2], imgs_f.shape[3]))
        yield imgs_f, imgs_f_down
#%%
#%%
print('+++++++++++++ Start Training +++++++++++++')

date_time = strftime("%Y%m%d-%H%M%S", localtime()) 
def saveModel(model, model_name, epoch, date_time):
        directory = os.path.join('saved_model', date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = 'saved_model/{}/{}_epoch_{}_weights.hdf5'.format(date_time, model_name, epoch)
        model.save_weights(model_path_w)
path = glob('/data/cephfs/punim0900/p2m/datasets/p2m8/train/*')
history_loss = []
for epoch in range(1, epochs + 1):
    for batch_i, (imgs, imgs_down) in enumerate(load_batch(path, batch_size)):
        loss = reconstruct_network.train_on_batch(imgs_down, imgs)
        print('Epoch', epoch, 'batch: ', batch_i, 'MSE loss', loss)
        history_loss.append(loss)
#%%
saveModel(reconstruct_network, 'reconstruct_network_unet_fdomain_normalised', epochs, date_time)
numpy_loss_history = np.array(history_loss)
directory = os.path.join('saved_model', date_time)
if not os.path.exists(directory):
    os.makedirs(directory)
np.save('saved_model/{}/loss_history.npy'.format(date_time), numpy_loss_history)

#%%
# path = glob('/Volumes/Samsung_T5/automap_datasets/brain/train/*')
# filepath = random.choice(path)
# print(filepath)
# #%%
# img = np.load(filepath)
# #%%
# img_f,img_f_down = to_freq_space_2d(img)
# #%%
# import matplotlib.pyplot as plt 

# #%%
# plt.imshow(10*np.log10(abs(img_f)), cmap = 'gray')
# #%%
# plt.imshow(10*np.log10(abs(img_f_down)), cmap = 'gray')
# #%%
# print(img_f.shape)
# plt.imshow(10*np.log10(np.abs(img_f)), cmap = 'gray')
# #%%
# plt.imshow(10*np.log10(img_f.imag), cmap = 'gray')
# #%%
# plt.imshow(np.abs((np.fft.ifft2(img_f))), cmap = 'gray')
# #%%
# img_fft = np.zeros(shape = (64,64), dtype = complex)
# img_fft.real = img_f_down[:,:,0]
# img_fft.imag = img_f_down[:,:,1]
# img_back = np.fft.ifft2(img_fft)
# plt.imshow(np.abs(img_back), cmap = 'gray')
# # img_f = np.fft.fft2(img)
# # img_f = np.fft.fftshift(img_f)
# # print(img_f.shape)

# # #%%
# # img_f_flat = np.reshape(img_f, (np.product(img_f.shape),))
# # idx = sample(range(np.product(img_f.shape)), int(0.1 * np.product(img_f.shape)))
# # img_f_flat[idx] = 0
# # img_f = np.reshape(img_f_flat, img_f.shape)
# # img_down = np.fft.ifft2(img_f)
# # #img_down = np.fft.fftshift(img_down)
# # img_down = abs(img_down)
# # #%%
# # import matplotlib.pyplot as plt
# # #%%
# # plt.imshow(img, cmap = 'gray')

# # #%%
# # plt.imshow(img_down, cmap = 'gray')

# # #%%
