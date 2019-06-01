#%%
from glob import glob 
import nibabel as nib 
from PIL import Image
import numpy as np 
import os
import random
from random import sample
#from matplotlib import pyplot as plt
np.random.seed(42)
#%%
class Dataloader():
    def __init__(self, path = './datasets/train/*',  batch_size = 10):
        ds = glob(path)
        self.img_path = []
        for n in range(len(ds)):
            if not os.path.isdir(ds[n]):
                self.img_path.append(ds[n])
        random.shuffle(self.img_path)
        self.batch_size = batch_size
        self.n_batches = int(len(self.img_path)/self.batch_size)
        self.img_size = 64
        self.normalize = True
    def read_image(self, path):
        img = Image.open(path).convert('LA')
        img = img.resize((self.img_size, self.img_size))
        a = random.uniform(0,1)
        if a < 0.5:
            img = img.rotate(180)
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
    def read_images(self, dir, n_imgs):
        path = glob(dir)
        imgs = []
        imgs_f = []
        for i in range(n_imgs):
            img = self.read_image(path[i])
            img = img - np.mean(img)
            img_f = self.to_freq_space_2d(img)
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
        if self.normalize:
            imgs = (imgs - np.min(imgs))/(np.max(imgs) - np.min(imgs))
        imgs = np.array(imgs)
        imgs_f = np.array(imgs_f)
        imgs_f = np.reshape(imgs_f, (-1, imgs_f.shape[1] * imgs_f.shape[2] * 2))
        return imgs, imgs_f
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
        for i in range(self.img_size):
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
#dl = Dataloader('imagenet/*', batch_size=1)
#%%
#imgs, imgs_f = dl.read_images('imagenet/*')
#print(imgs.shape)
#print(imgs_f.shape)
#%%
#plt.imshow(imgs[27,:,:], cmap = 'gray')
#%%
#for batch_i, (imgs, imgs_f) in enumerate(dl.load_batch()):
    #print(imgs.shape)
    #print(imgs_f.shape)
#img = Image.open('imagenet/ILSVRC2011_val_00000009.JPEG')




