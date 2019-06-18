#%%
import numpy as np 
import os
#from matplotlib import pyplot as plt 
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from glob import glob
from PIL import Image
#import dataloader
import datetime
from time import localtime, strftime 
import matplotlib.pyplot as plt
from random import sample
import random
#%%
def search_max_min_batch(path, num_images, normalize):
    max_total = 0
    min_total = 10000
    max_total_f = 0
    min_total_f = 10000
    for i in range(num_images):
        img = read_image(path[i], img_size = 64)
        img_f = to_freq_space_2d(img)
        if normalize:
            img_f = img_f - np.mean(img_f)
            img = img - np.mean(img)
        if np.max(img_f) > max_total_f:
            max_total_f = np.max(img_f)
        if np.min(img_f) < min_total_f:
            min_total_f = np.min(img_f)
        if np.max(img) > max_total:
            max_total = np.max(img)
        if np.min(img) < min_total:
            min_total = np.min(img)
    return max_total_f, min_total_f, max_total, min_total
def load_batch(path, batch_size, num_images):
    random.shuffle(path)
    n_batches = int(num_images/batch_size)
    #max_f, min_f, max, min = search_max_min_batch(path, num_images = num_images, normalize = False)
    for i in range(n_batches):
        batch = path[i * batch_size:(i + 1)* batch_size]
        imgs, imgs_f = [], []
        for img in batch:
            #img = read_image(img, img_size = 64)
            img = read_brain_image(img)
            #img = img - np.mean(img)
            img = (img - np.min(img))/(np.max(img) - np.min(img))
            img_f = to_freq_space_2d(img)
            img_f = undersample(img_f)
            #img_f = misalign_k_space(img_f)
            #img_f = img_f - np.mean(img_f)
            #mg_f = (img_f - np.min(img_f))/(np.max(img_f) - np.min(img_f))
            imgs.append(img)
            imgs_f.append(img_f)
        imgs = np.asarray(imgs, dtype = float)
        imgs_f = np.asarray(imgs_f, dtype = float)
        imgs = np.reshape(imgs, (-1, imgs.shape[1], imgs.shape[2],1))
        imgs_f = np.reshape(imgs_f, (-1, imgs_f.shape[1]* imgs_f.shape[2] * imgs_f.shape[3]))
        yield imgs, imgs_f
def read_images(dir, n_imgs, normalize):
    imgs = []
    imgs_f = []
    max_f, min_f, max, min = search_max_min_batch(path, n_imgs, normalize = True)
    for i in range(n_imgs):
        img = read_image(path[i], img_size = 64)
        if normalize:
            img = (img - min)/(max - min)
        img = img - np.mean(img)
        #img = img/np.std(img)
        img_f = to_freq_space_2d(img)
        imgs.append(img)
        imgs_f.append(img_f)
        """
        img = np.array(Image.open(path[i]).convert('LA'))
        img_gray = np.squeeze(img[:,:,0])
        (w,h )= img_gray.shape
        if w < 256 or h < 256:
            img = Image.open(path[i]).convert('LA')
            img = img.resize((256,256))
            img = np.array(img)
            img = np.squeeze(img[:,:,0])
            imgs.append(img)
        else:
            m_w = int(w/2)
            m_h = int(h/2)
            img_cropped = np.zeros(shape = (256,256))
            img_cropped = img_gray[(m_w - 128):(m_w + 128), (m_h - 128):( m_h + 128)]
            imgs.append(img_cropped)
        """
    imgs = np.array(imgs)
    imgs_f = np.array(imgs_f)
    imgs_f = np.reshape(imgs_f, (-1, imgs_f.shape[1] * imgs_f.shape[2] * 2))
    return imgs, imgs_f
def misalign_k_space(k_space):
    rows, cols, _ = k_space.shape
    k_space_mis = np.zeros(shape = k_space.shape)
    for i in range(rows):
        if i % 2 == 0:
            k_space_mis[i, 1:cols - 1, :] = k_space[i, 0:cols -2, :]
        else:
            k_space_mis[i,:,:] = k_space[i, :, :]
    return k_space_mis
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
def read_brain_image(path):
    return np.load(path)
def read_image(path, img_size):
    img = Image.open(path).convert('LA')
    img = img.resize((img_size, img_size))
    #a = random.uniform(0,1)
    #if a < 0.5:
        #img = img.rotate(180)
    img = np.array(img)
    img_gray = np.squeeze(img[:,:,0])
    """
    img = np.array(Image.open(path).convert('LA'))
    img_gray = np.squeeze(img[:,:,0])

    (w,h) = img_gray.shape 
    if w < self.img_size or h < self.img_size:
        img = Image.open(path).convert('LA')
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img)
        img = np.squeeze(img[:,:,0])    
    else:
        m_w = int(w/2)
        m_h = int(h/2)
        img = np.zeros(shape = (self.img_size,self.img_size))
        img = img_gray[(m_w - int(self.img_size/2)):(m_w + int(self.img_size/2)), (m_h - int(self.img_size/2)):( m_h + int(self.img_size/2))]
    """
    return img_gray
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

#%%
def custom_loss(lambda_1):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true)) + lambda_1 * K.mean(K.abs(y_pred - y_true))
    return loss
#%%
n = 64
inChannel = 2
m1 = 32
m2 = 64
input_img = Input(shape = (n*n*inChannel,)) #2*n^2
batch_size = 1
epochs = 1
# simulating Fourier Transform: mapping from sensor domain to the image domain
fc1 = Dense(n*n, activation = 'tanh')(input_img)
fc2 = Dense(n*n, activation = 'tanh')(fc1)
fc2 = Reshape((n,n,1))(fc2)

# sparse autoencoder
conv1 = Conv2D(m1, (3, 3), activation = 'relu', padding = 'same')(fc2)
conv2 = Conv2D(m2, (3, 3), activation = 'relu', padding = 'same')(conv1)
deconv = Conv2DTranspose(1, (3, 3), padding = 'same')(conv2)


automap = Model(input_img, deconv)

   
optimizer = RMSprop(lr = 0.00002) 
automap.compile(loss='mean_squared_error', optimizer = optimizer)

automap.summary()
#%%
automap.load_weights('automap/saved_model/20190618-143638/automap_undersampled_epoch_200_weights.hdf5')
#%%
path = glob('/Volumes/Samsung_T5/automap_datasets/brain/test/*')
results = []
inputs = []
fft_results = []
for batch_i, (imgs, imgs_f) in enumerate(load_batch(path, batch_size = 1,  num_images = 28)):
    img_f_fft= np.reshape(imgs_f, (64*64, 2))
    img_f_fft = img_f_fft.view(dtype = np.complex128)
    img_fft = abs(np.fft.ifft2(img_f_fft))
    img_fft = np.reshape(img_fft,(64,64))
    fft_results.append(img_fft)
    result = automap.predict(imgs_f)
    print('loss: ', np.mean(np.square(result - imgs)))
    result = np.array(result)
    result = np.squeeze(result)
    imgs = np.squeeze(imgs)
    results.append(result)
    inputs.append(imgs)
    
#%%
#print(np.min(results[0,:,:]))
#%%
results = np.array(results)
print(results.shape)
inputs = np.array(inputs)
print(inputs.shape)
fft_results = np.array(fft_results)
print(fft_results.shape)
#%%
#plt.imshow(fft_results[0,:,:], cmap = 'gray')
#%%
fig, axs = plt.subplots(3,14, figsize=(40,5))
print(axs.shape)
for row in range(3):
    for col in range(14):
        if row == 0:
            axs[row, col].imshow(inputs[col , :, :], cmap = 'gray')
        if row == 1:
            axs[row, col].imshow(results[col , :, :], cmap = 'gray')
        if row == 2:
            axs[row, col].imshow(np.rot90(fft_results[col, :, :], k =3), cmap = 'gray')
    
        
#plt.savefig('brain_automap_results_undersample2.png')

#%%

#%%
plt.imshow(results[17,:,:], cmap = 'gray')

#%%
plt.imshow(inputs[17,:,:], cmap = 'gray')

#%%
loss_history = np.load('automap/saved_model/20190612-033652/loss_history.npy')

#%%
print(loss_history.shape)

#%%
plt.plot(range(len(loss_history)), loss_history)

#%%
