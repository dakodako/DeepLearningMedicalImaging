#%%
import numpy as np 
import os 
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras.layers.core import Dropout, Lambda
from keras.layers.merge import concatenate
from keras import regularizers
from keras import backend as K
from glob import glob
import datetime 
from time import localtime, strftime
import random 
from random import sample 
import nibabel as nib 
import matplotlib.pyplot as plt
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
	output = Conv2D(1,(1,1), activation = 'relu', padding = 'same')(c9) 
	return output

#%%
print('++++++++++ contructing Neural Netork +++++++++ ')
n = 256 
inChannel = 1 
m1 = 32 
m2 = 64 
batch_size = 10 
epochs = 200
input_img = Input(shape = (n,n,inChannel,))
conv1 = Conv2D(m1, (3, 3), activation = 'relu', padding = 'same')(input_img)
conv2 = Conv2D(m2, (3, 3), activation = 'relu', padding = 'same')(conv1)
deconv = Conv2DTranspose(1, (3,3), padding = 'same')(conv2)

automap = Model(input_img, unet2(input_img))
optimizer = RMSprop(lr = 0.00002) 
automap.compile(loss='mean_squared_error', optimizer = optimizer)
model_path = 'automap/saved_model/20190619-124204/reconstruct_network_unet_epoch_100_weights.hdf5'
automap.load_weights(model_path)
automap.summary()
#%%
def to_freq_space_2d(img):
    """ Performs FFT of an image
    :param img: input 2D image
    :return: Frequency-space data of the input image, third dimension (size: 2)
    contains real and imaginary part
    """
    
    img_f = np.fft.fft2(img)  # FFT
    #img_fshift = np.fft.fftshift(img_f)  # FFT shift
    img_f_flat = np.reshape(img_f, (np.product(img_f.shape),))
    idx = sample(range(np.product(img_f.shape)), int(0.3 * np.product(img_f.shape)))
    img_f_flat[idx] = 0
    img_f= np.reshape(img_f_flat, img_f.shape)
    #img_real = img_f.real  # Real part: (im_size1, im_size2)
    #img_imag = img_f.imag  # Imaginary part: (im_size1, im_size2)
    #img_real_imag = np.dstack((img_real, img_imag))  # (im_size1, im_size2, 2)

    return img_f 
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
def load_batch(path, batch_size, img_num):
    random.shuffle(path)
    n_batches = int(img_num/batch_size)
    for i in range(n_batches):
        batch = path[i*batch_size:(i + 1)* batch_size]
        imgs, imgs_down = [], []
        for f in batch:
            img = read_image(f)
            img = (img - np.min(img))/(np.max(img) - np.min(img))
            img_f_down = to_freq_space_2d(img)
            # img_f_down = undersample(img_f)
            # img_f_down = undersample(img_f)
            # img_f_down = np.reshape(img_f_down, (img_f.shape[0]*img_f.shape[1], img_f.shape[2]))
            # img_f_down = img_f_down.view(dtype = np.complex128)
            img_down = abs(np.fft.ifft2(img_f_down))
            # img_down = np.reshape(img_down, (img_f.shape[0], img_f.shape[1]))
            # img_down = np.rot90(img_down)
            # img_down = np.rot90(img_down)
            # img_down = np.rot90(img_down)
            imgs_down.append(img_down)
            imgs.append(img)
        imgs = np.asarray(imgs, dtype = float)
        imgs_down = np.asarray(imgs_down, dtype = float)
        imgs = np.reshape(imgs, (-1, imgs.shape[1], imgs.shape[2], 1))
        imgs_down = np.reshape(imgs_down, (-1, imgs.shape[1],imgs.shape[2], 1))
        yield imgs, imgs_down
#%%
path = glob('/Volumes/Samsung_T5/p2m_datasets/p2m7/test/*')
results = []
inputs = []
ground_truth = []

for batch_i, (imgs, imgs_down) in enumerate(load_batch(path, batch_size = 1, img_num = 32)):
    # plt.subplot(1,2,1)
    # plt.imshow(np.squeeze(imgs), cmap = 'gray')
    # plt.subplot(1,2,2)
    # plt.imshow(np.squeeze(imgs_down), cmap = 'gray')
    # #plt.imshow(np.squeeze(imgs), cmap = 'gray')
    # #plt.imshow(np.squeeze(imgs_down), cmap = 'gray')
    result = automap.predict(imgs_down)
    result = np.array(result)
    result = np.squeeze(result)
    results.append(result)
    inputs.append(imgs_down)
    ground_truth.append(imgs)




#%%
results = np.array(results)
inputs = np.squeeze(np.array(inputs))
ground_truth = np.squeeze(np.array(ground_truth))
print(results.shape)
print(inputs.shape)
print(ground_truth.shape)
#%%
fig, axs = plt.subplots(4,4, figsize=(5,5))
print(axs.shape)
for row in range(4):
    for col in range(4):
        if row == 0:
            axs[row, col].imshow(ground_truth[col + 4  , :, :], cmap = 'gray')
        if row == 1:
            axs[row, col].imshow(inputs[col +4  , :, :], cmap = 'gray')
        if row == 2:
            axs[row, col].imshow(results[col +4 ,:,:], cmap = 'gray')
        if row == 3:
            cur = axs[row, col].imshow(np.abs(results[col+4 ,:,:] - ground_truth[col+4 ,:,:]), cmap = 'inferno')
            cur.set_clim(0,0.5)
            fig.colorbar(cur, ax = axs[row, col])
#plt.savefig('reconstruct_unet5.png')
#%%
col = 5
plt.subplots(1,4, figsize = (40,10))
plt.subplot(1,4,3)
plt.imshow(ground_truth[col,:,:], cmap = 'gray')
plt.subplot(1,4,2)
plt.imshow(results[col,:,:], cmap = 'gray')
plt.subplot(1,4,1)
plt.imshow(inputs[col,:,:], cmap = 'gray')
plt.subplot(1,4,4)
cur = plt.imshow(abs(results[col,:,:] - ground_truth[col,:,:]), cmap = 'inferno')
cur.set_clim(0, 0.5)
plt.colorbar(cur)
plt.savefig('reconstruct_unet_5.png')
#plt.subplot(1,3,3)
#plt.imshow(results[0,:,:], cmap = 'gray')
# #%%
# plt.imshow(ground_truth[10,:,:], cmap = 'gray')

# #%%
# plt.imshow(inputs[10,:,:], cmap = 'gray')

# #%%
# plt.imshow(results[10,:,:], cmap = 'gray')
# #%%
# plt.imshow(np.abs(results[10,:,:] - ground_truth[10,:,:]), cmap = 'gray')
# #%%
# filepath = random.choice(path)
# print(filepath)

# #%%
# img = read_image(filepath)
# img = (img - np.min(img))/(np.max(img) - np.min(img))
# img_f_down = to_freq_space_2d(img)
# #%%
# print(img_f_down.shape)
# #%%
# img_down = np.abs(np.fft.ifft2(img_f_down))
# #%%
# plt.imshow(img, cmap = 'gray')
# #%%
# plt.imshow(img_down, cmap = 'gray')
# #%%

# img_down = np.expand_dims(img_down, 0)
# img_down = np.expand_dims(img_down, 3)
# result = automap.predict(img_down)
# #%%
# plt.imshow(np.squeeze(result), cmap = 'gray')

# #%%


#%%
