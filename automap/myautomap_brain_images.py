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
import datetime
import pandas as pd
from time import localtime, strftime 
import random 
from random import sample
import nibabel as nib 
#%%
#from matplotlib import pyplot as plt
def load_batch(path, batch_size):
    random.shuffle(path)
    n_batches = int(len(path)/batch_size)
    #max_f, min_f, max, min = search_max_min_batch(path, num_images = num_images, normalize = False)
    for i in range(n_batches):
        batch = path[i * batch_size:(i + 1)* batch_size]
        imgs, imgs_f = [], []
        for img in batch:
            img = read_image(img)
            #img = img - np.mean(img)
            img = (img - np.min(img))/(np.max(img) - np.min(img))
            img_f = to_freq_space_2d(img)
            #img_f = img_f - np.mean(img_f)
            #img_f = (img_f - np.min(img_f))/(np.max(img_f) - np.min(img_f))
            imgs.append(img)
            imgs_f.append(img_f)
        imgs = np.asarray(imgs, dtype = float)
        imgs_f = np.asarray(imgs_f, dtype = float)
        imgs = np.reshape(imgs, (-1, imgs.shape[1], imgs.shape[2],1))
        imgs_f = np.reshape(imgs_f, (-1, imgs_f.shape[1]* imgs_f.shape[2] * imgs_f.shape[3]))
        yield imgs, imgs_f

def read_image(path):
    img = np.load(path)
    return img
def misalign_k_space(k_space):
    rows, cols, chan = k_space.shape
    k_space_mis = np.zeros(shape = k_space.shape)
    for i in range(rows):
        if i % 2 == 0:
            k_space_mis[i, 1:cols - 1, :] = k_space[i, 0:cols -2, :]
        else:
            k_space_mis[i,:,:] = k_space[i, :, :]
    return k_space_mis

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
#print('+++++++++ loading images +++++++++')
#train_n_imgs = 3000	
#val_n_imgs = 10
#dl = dataloader.Dataloader(path = './datasets/large/train/*', batch_size = 10)
#y_train, x_train = dl.read_images(dir ='./datasets/large/train/*', n_imgs = train_n_imgs)
#y_train = np.expand_dims(y_train, 3)
#y_val, x_val = dl.read_images(dir = './datasets/large/val/*', n_imgs = val_n_imgs)
#y_val = np.expand_dims(y_val, 3)
def custom_loss(lambda_1):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true)) + lambda_1 * K.mean(K.abs(y_pred - y_true))
    return loss
#%%
print('++++++++++ contructing AUTOMAP +++++++++ ')
n = 64
inChannel = 2
m1 = 32
m2 = 64
input_img = Input(shape = (n*n*inChannel,)) #2*n^2
batch_size = 30
epochs = 200

# simulating Fourier Transform: mapping from sensor domain to the image domain
fc1 = Dense(n*n, activation = 'tanh')(input_img)
fc2 = Dense(n*n, activation = 'tanh')(fc1)
fc2 = Reshape((n,n,1))(fc2)

# sparse autoencoder
conv1 = Conv2D(m1, (3, 3), activation = 'relu', padding = 'same')(fc2)
conv2 = Conv2D(m2, (3, 3), activation = 'relu', padding = 'same', kernel_regularizer = regularizers.l1(0.0001))(conv1)
deconv = Conv2DTranspose(1, (3, 3), padding = 'same')(conv2)


automap = Model(input_img, deconv)

optimizer = RMSprop(lr = 0.00002) 
automap.compile(loss='mean_squared_error', optimizer = optimizer)

automap.summary()
print('+++++++++++++ Start Training +++++++++++++ ')
#automap_train = automap.fit(x = x_train, y = y_train, batch_size = 100, epochs = epochs, verbose = 1, validation_data=(x_val, y_val))
#hist_df = pd.DataFrame(automap_train.history) 


# or save to csv:
# model_path_w = 'saved_model/{}/{}_epoch_{}_weights.hdf5'.format(date_time, model_name, epoch)
 
date_time = strftime("%Y%m%d-%H%M%S", localtime()) 
'''
directory = os.path.join('saved_model',date_time)
if not os.path.exists(directory):
    os.makedirs(directory)
hist_csv_file = 'saved_model/{}/{}_history.csv'.format(date_time, 'automap')
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
'''
#%%
##train_dir = 'imagenet/train/*' # does not exist yet
#val_dir = 'imagenet/val/*' # does not exist yet
#train_X = read_imgs_in_freq(train_dir)
#train_ground = read_images(train_dir)
#valid_X = read_imgs_in_freq(val_dir)
#valid_ground = read_images(val_dir)
#automap_train = automap.fit(train_X, train_ground, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data=(valid_X, valid_ground))

def saveModel(model, model_name, epoch, date_time):
        directory = os.path.join('saved_model', date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = 'saved_model/{}/{}_epoch_{}_weights.hdf5'.format(date_time, model_name, epoch)
        model.save_weights(model_path_w)
#%%
path = glob('datasets/brain/train/*')
#max_f , min_f = search_max_min_batch(path, num_images = 10000, normalize = True)
history_loss = []
for epoch in range(1,epochs + 1):
    for batch_i, (imgs, imgs_f) in enumerate(load_batch(path, 10)):
        loss = automap.train_on_batch(imgs_f, imgs)
        print('Epoch', epoch, 'batch: ', batch_i, 'MSE loss', loss)
        history_loss.append(loss)

#%%

saveModel(automap, 'automap', epochs, date_time)
numpy_loss_history = np.array(history_loss)
directory = os.path.join('saved_model', date_time)
if not os.path.exists(directory):
    os.makedirs(directory)
np.save('saved_model/{}/loss_history.npy'.format(date_time), numpy_loss_history)
np.savetxt("saved_model/{}/loss_history.txt".format(date_time), numpy_loss_history, delimiter=",")
#%%
#path = glob('datasets/large/train/*')
#max, min = search_max_min_batch(path = path, num_images = 28)
#%%
#imgs, imgs_f = read_images(path, n_imgs = 28, normalize = True)
#plt.imshow(imgs[2,:,:], cmap = 'gray')
#%%
#img = Image.open(path[22]).convert('LA')
#img = np.array(img)
#plt.imshow(img[:,:,0], cmap = 'gray')
#%%
#print(max)

#%%
#print(min)
#print(imgs[2,32,32])

#%%
#print(path)
#%%
#for batch_i, (imgs, imgs_f) in enumerate(load_batch(path, 2, num_images = 28)):
    #print(imgs.shape)
    #print(imgs_f.shape)
#%%
#print(np.max(imgs_f))
#print(np.min(imgs_f))
#%%
#plt.imshow(np.squeeze(imgs[0,:,:,:]), cmap = 'gray')

#%%
#plt.imshow(np.reshape(imgs_f[0,:], (64,64, 2))[:,:,1], cmap = 'gray')
#%%
#img_f = to_freq_space_2d(np.squeeze(imgs[0,:,:,:]))
#%%
#plt.imshow(img_f[:,:,1], cmap = 'gray')

