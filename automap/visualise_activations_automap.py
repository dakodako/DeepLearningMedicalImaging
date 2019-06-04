#%%
#import pydicom
from keras import models
from keras.models import load_model, Model, Sequential
#from pydicom.data import get_testdata_files
import numpy as np
import matplotlib.pyplot as plt
#from scipy.misc import imresize, imrotate
from skimage.transform import rotate, resize
import nibabel as nib
import glob
import sys
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
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
def load_data(path):
    random.shuffle(path)
    #max_f, min_f, max, min = search_max_min_batch(path, num_images = num_images, normalize = False)
    img = np.random.choice(path)
    img = read_image(img, img_size = 64)
    img = img - np.mean(img)
    #img = (img - np.min(img))/(np.max(img) - np.min(img))
    img_f = to_freq_space_2d(img)
    #img_f = img_f - np.mean(img_f)
    #img_f = (img_f - np.min(img_f))/(np.max(img_f) - np.min(img_f))
    img = np.asarray(img, dtype = float)
    img_f = np.asarray(img_f, dtype = float)
    img = np.reshape(img, (-1, img.shape[0], img.shape[1],1))
    img_f = np.reshape(img_f, (-1, img_f.shape[0]* img_f.shape[1] * img_f.shape[2]))
    return img, img_f
def load_batch(path, batch_size, num_images):
    random.shuffle(path)
    n_batches = int(num_images/batch_size)
    #max_f, min_f, max, min = search_max_min_batch(path, num_images = num_images, normalize = False)
    for i in range(n_batches):
        batch = path[i * batch_size:(i + 1)* batch_size]
        imgs, imgs_f = [], []
        for img in batch:
            img = read_image(img, img_size = 64)
            img = img - np.mean(img)
            #img = (img - np.min(img))/(np.max(img) - np.min(img))
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
automap.load_weights('automap/saved_model/20190603-113141/automap_epoch_100_weights.hdf5')
print(len(automap.layers))
#%%
weights = automap.get_weights()
#%%
print(len(weights))
print(len(weights[2]))
#%%
plt.imshow(np.reshape(weights[2], (4096,4096)), cmap = 'magma')
#%%
path = glob('/Volumes/Samsung_T5/automap_datasets/val/*')
print(len(path))
#%%
img, img_f = load_data(path)
print(img.shape)
print(img_f.shape)
#%%
plt.imshow(np.squeeze(img), cmap = 'gray')
#%%
result = automap.predict(img_f)
#%%
plt.imshow(np.squeeze(result), cmap = 'gray')
#%%
fc3 = automap.layers[3]
fc3_output = fc3.output
fc3_activations_model = Model(inputs = automap.input, outputs = fc3_output)
fc3_activations = fc3_activations_model.predict(img_f)
#model = load_model('autoencoder2_petra4.h5')
#model.summary()
#%%
print(fc3_activations.shape)
#fc3_activations = np.reshape(fc3_activations, (64,64))
#%%
plt.imshow(np.squeeze(fc3_activations))
#%%
filepath_test_X = 'PETRA2/*'
filepath_test_ground = 'MP2RAGE2/*'
test_X = open_images(filepath_test_X)
test_ground = open_images(filepath_test_ground)
print(test_X.shape)
#%%
test_img = test_X[30,:,:,:]
plt.imshow(test_img[:,:,0], cmap = 'gray')
#%%
test_img_tensor = np.expand_dims(test_img, axis = 0)
print(test_img_tensor.shape)
layer_outputs = [layer.output for layer in model.layers[:59]]
print(len(layer_outputs))

#%%
pred = model.predict(test_img_tensor)
plt.imshow(pred[0,:,:,0], cmap = 'gray')
#%%

im = plt.imshow(np.abs(pred[0,:,:,0]-test_ground[30,:,:,0]))

#%%
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
activations = activation_model.predict(test_img_tensor)
print(len(activations))

#%%
layer_names = []
for layer in model.layers[:59]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    print(size)
   
    n_cols = n_features // images_per_row
    print(n_cols)
    display_grid = np.zeros((size*n_cols, images_per_row * size))
    print(display_grid.shape)
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:,:, col*images_per_row + row]
            #channel_image = np.log(channel_image)
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col*size : (col+1)*size,
                        row*size:(row+1)*size] = channel_image
    if n_cols != 0:
        scale = 1./size
        fig = plt.figure(figsize=(scale*display_grid.shape[1], scale * display_grid.shape[0]))
        print(display_grid.shape)
        plt.title(layer_name)
        plt.grid(False)
        figname = 'activations/p2m/'+layer_name + '.png'
        plt.imshow(display_grid, aspect='auto')
        plt.show()
        fig.savefig(figname, dpi=fig.dpi)
