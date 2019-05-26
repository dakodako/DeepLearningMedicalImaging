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
from PIL import Image
#%%
from keras.models import load_model
from keras.layers import Layer, InputSpec
from keras.layers import Input,Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network
from keras import initializers, regularizers, constraints
import tensorflow as tf
#%%
class CycleGAN():
    def __init__(self, lr_D=2e-4, lr_G=2e-4, image_shape = (256, 256, 1),
                 date_time_string_addition='', image_folder='MR'):
        self.img_shape = image_shape
        self.channels = self.img_shape[-1]
        self.normalization = InstanceNormalization
        # Hyper parameters
        self.lambda_1 = 10.0  # Cyclic loss weight A_2_B
        self.lambda_2 = 10.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.learning_rate_D = lr_D
        self.learning_rate_G = lr_G
        self.generator_iterations = 1  # Number of generator training iterations in each training loop
        self.discriminator_iterations = 1  # Number of generator training iterations in each training loop
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = 1
        self.epochs = 100 # choose multiples of 25 since the models are save each 25th epoch
        self.save_interval = 1
        self.synthetic_pool_size = 50
        self.data_loader = DataLoader(dataset_name = 'p2m4', img_res = (256,256))
        # Linear decay of learning rate, for both discriminators and generators
        self.use_linear_decay = False
        self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start

        # Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
        self.use_identity_learning = False
        self.identity_mapping_modulus = 10  # Identity mapping will be done each time the iteration number is divisable with this number

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True

        # Multi scale discriminator - if True the generator have an extra encoding/decoding step to match discriminator information access
        self.use_multiscale_discriminator = False

        # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = False

        # Supervised learning part - for MR images - comparison
        self.use_supervised_learning = False
        self.supervised_weight = 10.0

        # Fetch data during training instead of pre caching all images - might be necessary for large datasets
        self.use_data_generator = False

        # Tweaks
        self.REAL_LABEL = 1.0  # Use e.g. 0.9 to avoid training the discriminators to zero loss

        # Used as storage folder name
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + date_time_string_addition

        # optimizer
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

        # ======= Discriminator model ==========
        if self.use_multiscale_discriminator:
            D_A = self.modelMultiScaleDiscriminator()
            D_B = self.modelMultiScaleDiscriminator()
            loss_weights_D = [0.5, 0.5] # 0.5 since we train on real and synthetic images
        else:
            D_A = self.modelDiscriminator()
            D_B = self.modelDiscriminator()
            loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images
        # D_A.summary()

        # Discriminator builds
        image_A = Input(shape=self.img_shape)
        image_B = Input(shape=self.img_shape)
        guess_A = D_A(image_A)
        guess_B = D_B(image_B)
        self.D_A = Model(inputs=image_A, outputs=guess_A, name='D_A_model')
        self.D_B = Model(inputs=image_B, outputs=guess_B, name='D_B_model')

        # self.D_A.summary()
        # self.D_B.summary()
        self.D_A.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)
        self.D_B.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)

        # Use containers to avoid falsy keras error about weight descripancies
        self.D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static_model')
        self.D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static_model')

        # ======= Generator model ==========
        # Do note update discriminator weights during generator training
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # Generators
        self.G_A2B = self.modelGenerator(name='G_A2B_model')
        self.G_B2A = self.modelGenerator(name='G_B2A_model')
        # self.G_A2B.summary()

        if self.use_identity_learning:
            self.G_A2B.compile(optimizer=self.opt_G, loss='MAE')
            self.G_B2A.compile(optimizer=self.opt_G, loss='MAE')

        # Generator builds
        real_A = Input(shape=self.img_shape, name='real_A')
        real_B = Input(shape=self.img_shape, name='real_B')
        synthetic_B = self.G_A2B(real_A)
        synthetic_A = self.G_B2A(real_B)
        dA_guess_synthetic = self.D_A_static(synthetic_A)
        dB_guess_synthetic = self.D_B_static(synthetic_B)
        reconstructed_A = self.G_B2A(synthetic_B)
        reconstructed_B = self.G_A2B(synthetic_A)

        model_outputs = [reconstructed_A, reconstructed_B]
        compile_losses = [self.cycle_loss, self.cycle_loss,
                          self.lse, self.lse]
        compile_weights = [self.lambda_1, self.lambda_2,
                           self.lambda_D, self.lambda_D]

        if self.use_multiscale_discriminator:
            for _ in range(2):
                compile_losses.append(self.lse)
                compile_weights.append(self.lambda_D)  # * 1e-3)  # Lower weight to regularize the model
            for i in range(2):
                model_outputs.append(dA_guess_synthetic[i])
                model_outputs.append(dB_guess_synthetic[i])
        else:
            model_outputs.append(dA_guess_synthetic)
            model_outputs.append(dB_guess_synthetic)

        if self.use_supervised_learning:
            model_outputs.append(synthetic_A)
            model_outputs.append(synthetic_B)
            compile_losses.append('MAE')
            compile_losses.append('MAE')
            compile_weights.append(self.supervised_weight)
            compile_weights.append(self.supervised_weight)

        self.G_model = Model(inputs=[real_A, real_B],
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=self.opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)
        # self.G_A2B.summary()

        
        
#===============================================================================
# Architecture functions

    def ck(self, x, k, use_normalization):
        x = Conv2D(filters=k, kernel_size=4, strides=2, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x0)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractionally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

#===============================================================================
# Models

    def modelMultiScaleDiscriminator(self, name=None):
        x1 = Input(shape=self.img_shape)
        x2 = AveragePooling2D(pool_size=(2, 2))(x1)
        #x4 = AveragePooling2D(pool_size=(2, 2))(x2)

        out_x1 = self.modelDiscriminator('D1')(x1)
        out_x2 = self.modelDiscriminator('D2')(x2)
        #out_x4 = self.modelDiscriminator('D4')(x4)

        return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

    def modelDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False)
        # Layer 2
        x = self.ck(x, 128, True)
        # Layer 3
        x = self.ck(x, 256, True)
        # Layer 4
        x = self.ck(x, 512, True)
        # Output layer
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        return Model(inputs=input_img, outputs=x, name=name)

    def modelGenerator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 32)
        # Layer 2
        x = self.dk(x, 64)
        # Layer 3
        x = self.dk(x, 128)

        if self.use_multiscale_discriminator:
            # Layer 3.5
            x = self.dk(x, 256)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk(x)

        if self.use_multiscale_discriminator:
            # Layer 12.5
            x = self.uk(x, 128)

        # Layer 13
        x = self.uk(x, 64)
        # Layer 14
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.channels, kernel_size=7, strides=1)(x)
        x = Activation('tanh')(x)  # They say they use Relu but really they do not
        return Model(inputs=input_img, outputs=x, name=name)


    #===============================================================================
    # Help functions

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss
    
#%%
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

#%%
class DataLoader():
    def __init__(self, dataset_name, img_res = (256,256)):
        self.img_res = img_res
        self.dataset_name = dataset_name
    def load_entire_batch(self):
        path = glob('datasets/p2m4/val/*')
        rndperm = np.random.permutation(len(path))
        print(len(path))
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
b, a, b_label, a_label = dataloader.load_entire_batch()
#%%
print(a.shape)
#%%
plt.imshow(np.squeeze(a[0,:,:]))
#%%
# === load models and evaluate === *
GAN = CycleGAN()
#%%
G_A2B = GAN.G_A2B
G_B2A = GAN.G_B2A
D_A = GAN.D_A
D_B = GAN.D_B
G_A2B.load_weights('models/saved_models/20190520-110332/G_A2B_model_weights_epoch_100.hdf5')
G_B2A.load_weights('models/saved_models/20190520-110332/G_B2A_model_weights_epoch_100.hdf5')
D_A.load_weights('models/saved_models/20190520-110332/D_A_model_weights_epoch_100.hdf5')
D_B.load_weights('models/saved_models/20190520-110332/D_B_model_weights_epoch_100.hdf5')
#%%
unet_model = load_model('models/u-net-p2m_l2_2.h5')
#%%
plt.imshow(np.squeeze(b[0,:,:]))
#%%
fake_As = []
fake_As_label = []
for n in range(b.shape[0]):
    # a is the mp2rage image b is the petra image
    s = np.expand_dims(b[n,:,:], 0)
    s = np.expand_dims(s, 3)
    fake_A = G_B2A.predict(s)
    fake_A = np.squeeze(fake_A)
    fake_As.append(fake_A)
    fake_As_label.append(2) #2 means fake_A
#%%
unet_As = []
unet_As_label = []
for n in range(b.shape[0]):
    s = np.expand_dims(b[n,:,:], 0)
    s = np.expand_dims(s, 3)
    s = 2*s - 1
    unet_A = unet_model.predict(s)
    unet_As.append(unet_A)
    unet_As_label.append(3) # 3 means unet_A
#%%
unet_As = np.array(np.squeeze(unet_As))
print(unet_As.shape)
unet_As_label = np.array(unet_As_label)
print(unet_As_label.shape)
#%%
fake_As = np.array(fake_As)
print(fake_As.shape)
#%%
fake_As_label = np.array(fake_As_label)
print(fake_As_label.shape)
#%%
unet_As = np.reshape(unet_As, (unet_As.shape[0], unet_As.shape[1]* unet_As.shape[2]))
#%%
fake_As = np.reshape(fake_As, (fake_As.shape[0], fake_As.shape[1]* fake_As.shape[2]))

a = np.reshape(a, (a.shape[0], a.shape[1] * a.shape[2]))
#%%
X = np.concatenate((a,fake_As, unet_As))
print(X.shape)
#%%
y = np.concatenate((a_label, fake_As_label, unet_As_label))
print(y.shape)

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
    palette=sns.color_palette("hls",3),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

#%%
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-three",
    hue="y",
    palette=sns.color_palette("hls",3),
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
    cmap='viridis'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
ax.legend()
plt.show()
#%%
N = 3*144
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

