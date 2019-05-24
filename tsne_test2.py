#%%
from __future__ import print_function
import time 
import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import nibabel as nib 
from skimage.transform import resize, rotate
from glob import glob
#%%
from keras.datasets import mnist
#%%
class DataLoader():
    def __init__(self, dataset_name, img_res = (256,256)):
        self.img_res = img_res
        self.dataset_name = dataset_name
    def load_entire_batch(self):
        path = glob('datasets/p2m4/val/*')
        rndperm = np.random.permutation(len(path))
        imgs_A = []
        imgs_B = []
        imgs_A_label = []
        imgs_B_label = []
        for i in range(len(path)):
            fname = path[rndperm[i]]
            img = nib.load(fname)
            img = img.get_data()
            _,_,w = img.shape
            _w = int(w/2)
            img_A, img_B = img[:,:,:_w], img[:,:,_w:]
            img_A = np.squeeze(img_A)
            img_B = np.squeeze(img_B)
            imgs_A.append(img_A)
            imgs_A_label.append(1)
            imgs_B.append(img_B)
            imgs_B_label.append(0)
        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)
        imgs_A_label = np.array(imgs_A_label)
        imgs_B_label = np.array(imgs_B_label)
        return imgs_A, imgs_B, imgs_A_label, imgs_B_label
    def load_data(self, batch_size = 1, is_testing = False, is_jitter = False):
        def randomCrop(img , mask, width, height):
            assert img.shape[0] >= height
            assert img.shape[1] >= width
            assert img.shape[0] == mask.shape[0]
            assert img.shape[1] == mask.shape[1]
            x = np.random.randint(0, img.shape[1] - width)
            y = np.random.randint(0, img.shape[0] - height)
            img = img[y:y+height, x:x+width]
            mask = mask[y:y+height, x:x+width]
            return img, mask
        data_type = "train" if not is_testing else "val"
        #path = glob('/home/student.unimelb.edu.au/chid/Documents/MRI_data/MRI_data/Daris/%s/%s/*' %(self.dataset_name,data_type))
        #path = glob('/home/chid/p2m/datasets/%s/%s/*' % (self.dataset_name, data_type))
        #path = glob('/Users/chid/.keras/datasets/%s/%s/*' % (self.dataset_name, data_type))
        path = glob('datasets/%s/%s/*' % (self.dataset_name, data_type))
        batch_images = np.random.choice(path, size = batch_size)
        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = nib.load(img_path)
            img = img.get_data()
            _,_,w = img.shape
            _w = int(w/2)
            img_A, img_B = img[:,:,:_w], img[:,:,_w:]
            #img_A, img_B = img[:,:,_w:],img[:,:,:_w]
            img_A = np.squeeze(img_A)
            img_B = np.squeeze(img_B)
            #img_A = Image.fromarray(img_A, mode = 'F')
            #img_B = Image.fromarray(img_B, mode = 'F')
            #img_A = img_A.resize(size = (self.img_res[0], self.img_res[1]))
            #img_B = img_B.resize(size = (self.img_res[0], self.img_res[1]))
            #img_A = img_A.resize( (self.img_res[0],self.img_res[1]))
            #img_B = resize(img_B, (self.img_res[0],self.img_res[1]))
            if not is_testing and np.random.random() <0.5 and is_jitter:
                # 1. Resize an image to bigger height and width
                img_A = Image.fromarray(img_A, mode = 'F')
                img_B = Image.fromarray(img_B, mode = 'F')
                img_A = img_A.resize(shape = (img_A.shape[0] + 64, img_A.shape[1] + 64))
                img_B = img_B.resize(shape = (img_B.shape[0] + 64, img_B.shape[1] + 64))
                img_A = np.array(img_A)
                img_B = np.array(img_B)
                # 2. Randomly crop the image
                img_A, img_B = randomCrop(img_A, img_B, self.img_res[0], self.img_res[1])
                # 3. Randomly flip the image horizontally
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)
            m_A = np.max(img_A)
            mi_A = np.min(img_A)
            img_A = 2* (img_A - mi_A)/(m_A - mi_A) - 1
            m_B = np.max(img_B)
            mi_B = np.min(img_B)
            img_B = 2* (img_B - mi_B)/(m_B - mi_B) -1 
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = np.asarray(imgs_A, dtype=float)
        imgs_A = np.reshape(imgs_A, (-1,imgs_A.shape[1], imgs_A.shape[2],1))
        imgs_B = np.asarray(imgs_B, dtype = float)
        imgs_B = np.reshape(imgs_B, (-1,imgs_B.shape[1],imgs_B.shape[2],1))
        return imgs_A, imgs_B
#%%   
dataloader = DataLoader(dataset_name = 'p2m4')
a, b, a_label, b_label = dataloader.load_entire_batch()
#%%
a = np.reshape(a, (a.shape[0], a.shape[1] * a.shape[2]))
b = np.reshape(b, (b.shape[0], b.shape[1] * b.shape[2]))
X = np.concatenate((a,b))
print(X.shape)
#%%
y = np.concatenate((a_label, b_label))
print(y.shape)

#%%
feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
# X.shape[1] = 28 * 28 = 784 which means the 784 features in the handwritten digits
#%%
df = pd.DataFrame(X, columns = feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
#%%
X, y = None, None 
#%%
print('Size of the dataframe: {}'.format(df.shape))
#%%
# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

#%%
plt.gray()
fig = plt.figure( figsize=(16,7) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
plt.show()
#%%
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#%%
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df[feat_cols].values)
#%%
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
#%%
print(tsne_results.shape)
#%%
df_subset = None
#%%
#df.loc[rndperm,:]['tsne-2d-one'] = tsne_results[:,0]
#df.loc[rndperm,:]['tsne-2d-two'] = tsne_results[:,1]
#df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls",2),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

#%%
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls",2),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

#%%
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["pca-one"], 
    ys=df.loc[rndperm,:]["pca-two"], 
    zs=df.loc[rndperm,:]["pca-three"],
    c=df.loc[rndperm,:]["y"], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()
#%%
N = 288
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

