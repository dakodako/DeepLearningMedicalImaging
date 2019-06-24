#%%
import nibabel as nib 
from glob import glob 
from skimage.transform import resize
import numpy as np
import os
from matplotlib import pyplot as plt
#%%
path_name = '/Volumes/Samsung_T5/p2m_datasets/p2m7/train/*'
path = glob(path_name)
print(len(path))
for i in range(len(path)):
    print('num ', i)
    img = nib.load(path[i])
    img = img.get_data()
    _,_,w = img.shape
    _w = int(w/2)
    img = img[:,:,_w:]
    img = np.array(img)
    img = resize(img, (1, 128, 128))
    img = np.squeeze(img)
    filename = str(i) + '.npy'
    np.save(os.path.join('/Volumes/Samsung_T5/automap_datasets/brain_128/train/', filename), img)

#%%
#sample = np.load('automap/1000.npy')
#print(sample.shape)
#%%
#plt.imshow(sample, cmap = 'gray')