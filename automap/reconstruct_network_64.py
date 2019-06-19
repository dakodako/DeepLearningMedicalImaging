#%%
import numpy as np 
import os 
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from glob import glob
import datetime 
from time import localtime, strftime
import random 
from random import sample 
import nibabel as nib 
#%%
print('++++++++++ contructing Neural Netork +++++++++ ')
n = 64
inChannel = 1 
m1 = 32 
m2 = 64 
batch_size = 10 
epochs = 200
input_img = Input(shape = (n,n,inChannel,))
conv1 = Conv2D(m1, (3, 3), activation = 'relu', padding = 'same')(input_img)
conv2 = Conv2D(m2, (3, 3), activation = 'relu', padding = 'same')(conv1)
deconv = Conv2DTranspose(1, (3,3), padding = 'same')(conv2)

automap = Model(input_img, deconv)
optimizer = RMSprop(lr = 0.00002) 
automap.compile(loss='mean_squared_error', optimizer = optimizer)

automap.summary()

#%%
def to_freq_space_2d(img):
    """ Performs FFT of an image
    :param img: input 2D image
    :return: Frequency-space data of the input image, third dimension (size: 2)
    contains real and imaginary part
    """
    
    img_f = np.fft.fft2(img)  # FFT
    img_fshift = np.fft.fftshift(img_f)  # FFT shift
    img_real = img_fshift.real  # Real part: (im_size1, im_size2)
    img_imag = img_fshift.imag  # Imaginary part: (im_size1, im_size2)
    img_real_imag = np.dstack((img_real, img_imag))  # (im_size1, im_size2, 2)

    return img_real_imag  
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
    img = np.load(path)
    return img
def load_batch(path, batch_size):
    random.shuffle(path)
    n_batches = int(len(path)/batch_size)
    for i in range(n_batches):
        batch = path[i*batch_size:(i + 1)* batch_size]
        imgs, imgs_down = [], []
        for f in batch:
            img = read_image(f)
            img = (img - np.min(img))/(np.max(img) - np.min(img))
            img_f = to_freq_space_2d(img)
            img_f_down = undersample(img_f)
            img_f_down = undersample(img_f)
            img_f_down = np.reshape(img_f_down, (img_f.shape[0]*img_f.shape[1], img_f.shape[2]))
            img_f_down = img_f_down.view(dtype = np.complex128)
            img_down = abs(np.fft.ifft2(img_f_down))
            img_down = np.reshape(img_down, (img_f.shape[0], img_f.shape[1]))
            img_down = np.rot90(img_down)
            img_down = np.rot90(img_down)
            img_down = np.rot90(img_down)
            imgs_down.append(img_down)
            imgs.append(img)
        imgs = np.asarray(imgs, dtype = float)
        imgs_down = np.asarray(imgs_down, dtype = float)
        imgs = np.reshape(imgs, (-1, imgs.shape[1], imgs.shape[2], 1))
        imgs_down = np.reshape(imgs_down, (-1, imgs.shape[1],imgs.shape[2], 1))
        yield imgs, imgs_down
#%%
print('+++++++++++++ Start Training +++++++++++++')

date_time = strftime("%Y%m%d-%H%M%S", localtime()) 
def saveModel(model, model_name, epoch, date_time):
        directory = os.path.join('saved_model', date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = 'saved_model/{}/{}_epoch_{}_weights.hdf5'.format(date_time, model_name, epoch)
        model.save_weights(model_path_w)
path = glob('/data/cephfs/punim0900/automap/datasets/brain/train/*')
history_loss = []
for epoch in range(1, epochs + 1):
    for batch_i, (imgs, imgs_down) in enumerate(load_batch(path, batch_size)):
        loss = automap.train_on_batch(imgs_down, imgs)
        print('Epoch', epoch, 'batch: ', batch_i, 'MSE loss', loss)
        history_loss.append(loss)
#%%
saveModel(automap, 'reconstruct_network', epochs, date_time)
numpy_loss_history = np.array(history_loss)
directory = os.path.join('saved_model', date_time)
if not os.path.exists(directory):
    os.makedirs(directory)
np.save('saved_model/{}/loss_history.npy'.format(date_time), numpy_loss_history)
#%%
# path = glob('/Volumes/Samsung_T5/p2m_datasets/p2m7/test/*')
# filepath = random.choice(path)
# img = read_image(filepath)
# print(img.shape)

#%%
# img = (img - np.min(img))/(np.max(img) - np.min(img))
# img_f = to_freq_space_2d(img)
# img_f_down = undersample(img_f)
# img_f_down = np.reshape(img_f_down, (img_f.shape[0]*img_f.shape[1], img_f.shape[2]))
# img_f_down = img_f_down.view(dtype = np.complex128)
# img_down = abs(np.fft.ifft2(img_f_down))
# img_down = np.reshape(img_down, (img_f.shape[0], img_f.shape[1]))
# img_down = np.rot90(img_down)
# img_down = np.rot90(img_down)
# img_down = np.rot90(img_down)
# print(img_down.shape)

#%%
#import matplotlib.pyplot as plt

#%%
# plt.imshow(img_down, cmap = 'gray')
#%%
#plt.imshow(np.rot90(img_down, axes=(1, 0)), cmap = 'gray')
#%%
# plt.imshow(img, cmap = 'gray')

#%%
#print(np.max(img_down))
#print(np.min(img_down))

#%%
#print(np.max(img))
#print(np.min(img))
