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
import dataloader
import datetime
import pandas as pd
from time import localtime, strftime 
#%%
print('+++++++++ loading images +++++++++')
train_n_imgs = 3000	
val_n_imgs = 10
dl = dataloader.Dataloader(path = './datasets/large/train/*', batch_size = 10)
y_train, x_train = dl.read_images(dir ='./datasets/large/train/*', n_imgs = train_n_imgs)
y_train = np.expand_dims(y_train, 3)
y_val, x_val = dl.read_images(dir = './datasets/large/val/*', n_imgs = val_n_imgs)
y_val = np.expand_dims(y_val, 3)
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
epochs = 100

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
automap_train = automap.fit(x = x_train, y = y_train, batch_size = 100, epochs = epochs, verbose = 1, validation_data=(x_val, y_val))
hist_df = pd.DataFrame(automap_train.history) 


# or save to csv:
# model_path_w = 'saved_model/{}/{}_epoch_{}_weights.hdf5'.format(date_time, model_name, epoch)
 
date_time = strftime("%Y%m%d-%H%M%S", localtime()) 
directory = os.path.join('saved_model',date_time)
if not os.path.exists(directory):
    os.makedirs(directory)
hist_csv_file = 'saved_model/{}/{}_history.csv'.format(date_time, 'automap')
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

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

'''
for epoch in range(1,epochs + 1):
    print('epoch: ', epoch)
    for batch_i, (imgs, imgs_F) in enumerate(dl.load_batch()):
        print('batch: ', batch_i)
        loss = automap.train_on_batch(imgs_F, imgs)
        print('MSE loss', loss)
'''
filename = date_time + 'automap.hdf5'
#automap.save_weights(filename)
saveModel(automap, 'automap', epochs, date_time)
