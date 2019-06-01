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
#%%
def custom_loss(lambda_1):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true)) + lambda_1 * K.mean(K.abs(y_pred - y_true))
    return loss

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

   
automap.compile(loss=custom_loss(0.0001), optimizer = RMSprop())

automap.summary()
#%%
automap.load_weights('automap/saved_model/20190530-104832/automap_epoch_100_weights.hdf5')

#%%
class Dataloader():
    def __init__(self, path = './datasets/train/*'):
        ds = glob(path)
        self.img_path = []
        for n in range(len(ds)):
            if not os.path.isdir(ds[n]):
                self.img_path.append(ds[n])
        self.batch_size = 1
        self.n_batches = int(len(self.img_path)/self.batch_size)
        self.img_size = 64
    def read_image(self, path):
        img = Image.open(path).convert('LA')
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img)
        img_gray = np.squeeze(img[:,:,0])
        #(w,h) = img_gray.shape 

        
        """
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
    def read_images(self, dir):
        path = glob(dir)
        imgs = []

        for i in range(len(path)):
            img = self.read_image(path[i])
            imgs.append(img)
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
        return imgs
    def undersample(self, img_f, ratio):
        img_real = img_f[:,:,0]
        img_imag = img_f[:,:,1]
        img_real_flat = np.reshape(img_real, (np.product(img_real.shape), ))
        img_imag_flat = np.reshape(img_imag, (np.product(img_imag.shape),))
        idx = sample(range(np.product(img_real.shape)), int(ratio * np.product(img_real.shape)))
        img_real_flat[idx] = 0
        img_imag_flat[idx] = 0
        img_real_down = np.reshape(img_real_flat, img_real.shape)
        img_imag_down = np.reshape(img_imag_flat, img_imag.shape)
        img_f_down = np.zeros(shape = img_f.shape)
        img_f_down[:,:,0] = img_real_down
        img_f_down[:,:,1] = img_imag_down
        return img_f_down
    def to_freq_space_2d(self, img):
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
    def read_imgs_in_freq(self, dir):
        path = glob(dir)
        imgs_f = []
        for i in range(len(path)):
            img_cropped = self.read_image(path[i])
            '''
            img = np.array(Image.open(path[i]).convert('LA'))
            img_gray = np.squeeze(img[:,:,0])
            (w,h )= img_gray.shape
            if w < 256 or h < 256:
                img = Image.open(path[i]).convert('LA')
                img_cropped = img.resize((256,256))
                img_cropped = np.array(img_cropped)
                img_cropped = np.squeeze(img_cropped[:,:,0])
                
                #imgs.append(img)
            else:
                m_w = int(w/2)
                m_h = int(h/2)
                img_cropped = np.zeros(shape = (256,256))
                img_cropped = img_gray[(m_w - 128):(m_w + 128), (m_h - 128):( m_h + 128)]
            '''
            img_freq = self.to_freq_space_2d(img_cropped)
            imgs_f.append(img_freq)
        imgs_f = np.array(imgs_f)
        return imgs_f 
    def load_batch(self):
        for i in range(self.n_batches - 1):
            batch = self.img_path[i*self.batch_size:(i+1)*self.batch_size]
            imgs, imgs_f = [], []
            for img in batch:
                img = self.read_image(img)
                #print(img.shape)
                img_f = self.to_freq_space_2d(img)
                #print('img_f', img_f.shape)
                imgs.append(img)
                imgs_f.append(img_f)
            imgs = np.asarray(imgs, dtype=float)
            imgs_f = np.asarray(imgs_f, dtype = float)
            imgs = np.reshape(imgs, (-1, imgs.shape[1], imgs.shape[2], 1))
            imgs_f = np.reshape(imgs_f, (-1, imgs_f.shape[1]*imgs_f.shape[2]*imgs_f.shape[3]))
            yield imgs, imgs_f
#%%
dl = Dataloader(path = 'imagenet/*')
#%%
img = dl.read_image('imagenet/ILSVRC2011_val_00000019.JPEG')

#%%
print(img.shape)
#%%
plt.imshow(img, cmap = 'gray')
#%%
img_f = dl.to_freq_space_2d(img)
img_f = np.reshape(img_f, (-1, 64*64*2))
#%%
print(img_f.shape)
#%%
pred = automap.predict(img_f)
#%%
print(pred.shape)
plt.imshow(np.squeeze(pred), cmap = 'gray')
#%%
print(np.mean(np.square(np.squeeze(pred) - img)))