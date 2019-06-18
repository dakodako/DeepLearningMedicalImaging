#%%
import numpy as np 
import matplotlib.pyplot as plt
#%%
fa = np.load('temp_output_a.npy')
a = np.load('input_a.npy')
fb = np.load('temp_output_b.npy')
b = np.load('input_b.npy')
#%%
plt.imshow(np.squeeze(fa), cmap = 'gray')
#%%
plt.imshow(np.squeeze(a), cmap = 'gray')
#%%
plt.imshow(np.squeeze(fb), cmap = 'gray')
#%%
plt.imshow(np.squeeze(b), cmap = 'gray')