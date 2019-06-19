#%%
import numpy as np 
from matplotlib import pyplot as plt 
#%%
#ground_truths = np.load('models/UNIT/ground_truth.npy')
#outputs = np.load('models/UNIT/outputs.npy')
fakeAs = np.load('models/results/CycleGAN/fakeAs.npy')
fakeBs = np.load('models/results/CycleGAN/fakeBs.npy')
inputs = np.load('models/UNIT/inputs.npy')
ground_truths = np.load('models/UNIT/ground_truth.npy')
outputs = np.load('models/UNIT/outputs.npy')
#%%
fakeAs = np.load('models/CycleGAN/fakeAs.npy')
fakeBs = np.load('models/CycleGAN/fakeBs.npy')

#%%
print(fakeAs.shape)
print(fakeBs.shape)

#%%
print(ground_truths.shape)
print(outputs.shape)

#%%
plt.imshow(fakeAs[0,:,:], cmap = 'gray')

#%%
print(np.mean(np.abs(fakeAs[0,:,:] - ground_truths[0,:,:])))
print(np.mean(np.abs(fakeBs[0,:,:] - inputs[0,:,:])))
#%%
plt.imshow(ground_truths[0,:,:], cmap = 'gray')

#%%
