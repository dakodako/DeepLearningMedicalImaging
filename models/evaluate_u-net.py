#%%

from keras import models
from keras.models import load_model
import numpy as np 
from skimage.transform import rotate, resize
import nibabel as nib 
from glob import glob 
import sys 
import matplotlib.pyplot as plt

#%%
batch_size = 1
path = glob('datasets/p2m4/val/*')
print(path)
#%%
batch_images = np.random.choice(path, size = batch_size)
print(batch_images)
#%%
a = nib.load(batch_images[0])
a = a.get_fdata()
print(a.shape)
#%%
target = a[0,:,0:255]
X = a[:,:,255:511]
mi = np.min(X)
m = np.max(X)
X = (X - mi)/(m - mi)
mi = np.min(target)
m = np.max(target)
target = (target - mi)/(m - mi)
#plt.imshow(np.squeeze(X), cmap= 'gray')
X = np.expand_dims(X, 3)
print(X.shape)

#%%
model = load_model('models/u-net-p2m_l2_2.h5')
model.summary()

#%%

pred = model.predict(X)
#%%
print(pred.shape)
#%%
plt.imshow(np.squeeze(pred), cmap = 'gray')
#%%
plt.imshow(np.squeeze(target), cmap = 'gray')
#%%
loss = np.mean(np.square(pred- target))
print(loss)
