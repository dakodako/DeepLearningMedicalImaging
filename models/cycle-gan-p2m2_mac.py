#%%
from keras.layers import Layer, InputSpec
from keras.layers import Input,Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network
from keras import initializers, regularizers, constraints
#%%
from collections import OrderedDict
#from scipy.misc import imsave, toimage  # has depricated
import numpy as np
import random
import datetime
import time
import json
import math
import csv
import sys
import os
#%%
from PIL import Image
from keras.utils import Sequence
#from skimage.transform import resize, rotate
from glob import glob 
import nibabel as nib
import matplotlib.pyplot as plt
#%%
import keras.backend as K
import tensorflow as tf
#%%


np.random.seed(seed=12345)
class InstanceNormalization(Layer):
    """Instance normalization layer.

    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.

    # Output shape
        Same shape as input.

    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
'''
def load_data(nr_of_channels, batch_size=1, nr_A_train_imgs=None, nr_B_train_imgs=None,
              nr_A_test_imgs=None, nr_B_test_imgs=None, subfolder='',
              generator=False, D_model=None, use_multiscale_discriminator=False, use_supervised_learning=False, REAL_LABEL=1.0):

    trainA_path = os.path.join('data', subfolder, 'trainA')
    trainB_path = os.path.join('data', subfolder, 'trainB')
    testA_path = os.path.join('data', subfolder, 'testA')
    testB_path = os.path.join('data', subfolder, 'testB')

    trainA_image_names = os.listdir(trainA_path)
    if nr_A_train_imgs != None:
        trainA_image_names = trainA_image_names[:nr_A_train_imgs]

    trainB_image_names = os.listdir(trainB_path)
    if nr_B_train_imgs != None:
        trainB_image_names = trainB_image_names[:nr_B_train_imgs]

    testA_image_names = os.listdir(testA_path)
    if nr_A_test_imgs != None:
        testA_image_names = testA_image_names[:nr_A_test_imgs]

    testB_image_names = os.listdir(testB_path)
    if nr_B_test_imgs != None:
        testB_image_names = testB_image_names[:nr_B_test_imgs]

    if generator:
        return data_sequence(trainA_path, trainB_path, trainA_image_names, trainB_image_names, batch_size=batch_size)  # D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL)
    else:
        trainA_images = create_image_array(trainA_image_names, trainA_path, nr_of_channels)
        trainB_images = create_image_array(trainB_image_names, trainB_path, nr_of_channels)
        testA_images = create_image_array(testA_image_names, testA_path, nr_of_channels)
        testB_images = create_image_array(testB_image_names, testB_path, nr_of_channels)
        return {"trainA_images": trainA_images, "trainB_images": trainB_images,
                "testA_images": testA_images, "testB_images": testB_images,
                "trainA_image_names": trainA_image_names,
                "trainB_image_names": trainB_image_names,
                "testA_image_names": testA_image_names,
                "testB_image_names": testB_image_names}


def create_image_array(image_list, image_path, nr_of_channels):
    image_array = []
    for image_name in image_list:
        if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image -> MR image
                image = np.array(Image.open(os.path.join(image_path, image_name)))
                image = image[:, :, np.newaxis]
            else:                   # RGB image -> 3 channels
                image = np.array(Image.open(os.path.join(image_path, image_name)))
            image = normalize_array(image)
            image_array.append(image)

    return np.array(image_array)
  
  
  # If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
def normalize_array(array):
    array = array / 127.5 - 1
    return array


class data_sequence(Sequence):

    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, batch_size=1):  # , D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL):
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        for image_name in image_list_A:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))

    def __len__(self):
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))

    def __getitem__(self, idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:
        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        real_images_A = create_image_array(batch_A, '', 3)
        real_images_B = create_image_array(batch_B, '', 3)

        return real_images_A, real_images_B  # input_data, target_data
'''
class DataLoader():
    def __init__(self, dataset_name, img_res = (128,128)):
        self.img_res = img_res
        self.dataset_name = dataset_name

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
    
        data_type = "train" if not is_testing else "test"
        #path = glob('/home/student.unimelb.edu.au/chid/Documents/MRI_data/MRI_data/Daris/%s/%s/*' %(self.dataset_name,data_type))
        #path = glob('/home/chid/p2m/datasets/%s/%s/*' % (self.dataset_name, data_type))
        path = glob('/Users/chid/.keras/datasets/%s/%s/*' % (self.dataset_name, data_type))
        #path = glob('/home/chid/p2m/datasets/%s/%s/*' % (self.dataset_name, data_type))
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
    
    def load_batch(self, batch_size = 1, is_testing = False, is_jitter = False):
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
        data_type = "train" if not is_testing else "test"
        path = glob('/home/chid/p2m/datasets/%s/%s/*' % (self.dataset_name, data_type))
        #path = glob('/Users/chid/.keras/datasets/%s/%s/*' % (self.dataset_name, data_type))
        #path = glob('/home/student.unimelb.edu.au/chid/Documents/MRI_data/MRI_data/Daris/%s/%s/*' % (self.dataset_name,data_type)) 
        self.n_batches = int(len(path) / batch_size)
        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = nib.load(img)
                img = img.get_data()
                _,_,w = img.shape
                _w = int(w/2)
                img_A, img_B = img[:,:,:_w], img[:,:,_w:]
                #img_A, img_B = img[:,:,_w:],img[:,:,:_w]
                img_A = np.squeeze(img_A)
                img_B = np.squeeze(img_B)
                #img_A = resize(img_A, (self.img_res[0],self.img_res[1]))
                #img_B = resize(img_B, (self.img_res[0],self.img_res[1]))
                #print(img_A.shape)
                #print(img_B.shape)
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
                img_B = 2* (img_B - mi_B)/(m_B - mi_B) - 1
                imgs_A.append(img_A)
                imgs_B.append(img_B)
            imgs_A = np.asarray(imgs_A, dtype=float)
            imgs_A = np.reshape(imgs_A, (-1,imgs_A.shape[1], imgs_A.shape[2],1))
            imgs_B = np.asarray(imgs_B, dtype = float)
            imgs_B = np.reshape(imgs_B, (-1,imgs_B.shape[1], imgs_B.shape[2],1))
            yield imgs_A, imgs_B
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
        self.data_loader = DataLoader(dataset_name = 'p2m', img_res = (256,256))
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

        # ======= Data ==========
        # Use 'None' to fetch all available images
        '''
        nr_A_train_imgs = None
        nr_B_train_imgs = None
        nr_A_test_imgs = None
        nr_B_test_imgs = None

        if self.use_data_generator:
            print('--- Using dataloader during training ---')
        else:
            print('--- Caching data ---')
        sys.stdout.flush()

        if self.use_data_generator:
            self.data_generator = load_data(
                nr_of_channels=self.batch_size, generator=True, subfolder=image_folder)

            # Only store test images
            nr_A_train_imgs = 0
            nr_B_train_imgs = 0

        data = load_data(nr_of_channels=self.channels,
                                   batch_size=self.batch_size,
                                   nr_A_train_imgs=nr_A_train_imgs,
                                   nr_B_train_imgs=nr_B_train_imgs,
                                   nr_A_test_imgs=nr_A_test_imgs,
                                   nr_B_test_imgs=nr_B_test_imgs,
                                   subfolder=image_folder)

        self.A_train = data["trainA_images"]
        self.B_train = data["trainB_images"]
        self.A_test = data["testA_images"]
        self.B_test = data["testB_images"]
        self.testA_image_names = data["testA_image_names"]
        self.testB_image_names = data["testB_image_names"]
        if not self.use_data_generator:
            print('Data has been loaded')
        '''
        # ======= Create designated run folder and store meta data ==========
        '''
        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.writeMetaDataToJSON()
        '''
        # ======= Avoid pre-allocating GPU memory ==========
        # uncommenting the following lines when using GPU
        # TensorFlow wizardry
        #config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        #config.gpu_options.allow_growth = True

        # Create a session with the above options specified.
        #K.tensorflow_backend.set_session(tf.Session(config=config))

        # ===== Tests ======
        # Simple Model
        '''
        self.G_A2B = self.modelSimple('simple_T1_2_T2_model')
        self.G_B2A = self.modelSimple('simple_T2_2_T1_model')
        self.G_A2B.compile(optimizer=Adam(), loss='MAE')
        self.G_B2A.compile(optimizer=Adam(), loss='MAE')
        self.trainSimpleModel()
        self.load_model_and_generate_synthetic_images()
        '''
        # ======= Initialize training ==========
        #sys.stdout.flush()
        #plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)
        '''
        self.train(epochs=self.epochs, batch_size=self.batch_size, save_interval=self.save_interval)
        '''
        #self.load_model_and_generate_synthetic_images()

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
# Test - simple model
    '''
    def modelSimple(self, name=None):
        inputImg = Input(shape=self.img_shape)
        #x = Conv2D(1, kernel_size=5, strides=1, padding='same')(inputImg)
        #x = Dense(self.channels)(x)
        x = Conv2D(256, kernel_size=1, strides=1, padding='same')(inputImg)
        x = Activation('relu')(x)
        x = Conv2D(self.channels, kernel_size=1, strides=1, padding='same')(x)

        return Model(input=inputImg, output=x, name=name)

    def trainSimpleModel(self):
        real_A = self.A_test[0]
        real_B = self.B_test[0]
        real_A = real_A[np.newaxis, :, :, :]
        real_B = real_B[np.newaxis, :, :, :]
        epochs = 200
        for epoch in range(epochs):
            print('Epoch {} started'.format(epoch))
            self.G_A2B.fit(x=self.A_train, y=self.B_train, epochs=1, batch_size=1)
            self.G_B2A.fit(x=self.B_train, y=self.A_train, epochs=1, batch_size=1)
            #loss = self.G_A2B.train_on_batch(x=real_A, y=real_B)
            #print('loss: ', loss)
            synthetic_image_A = self.G_B2A.predict(real_B, batch_size=1)
            synthetic_image_B = self.G_A2B.predict(real_A, batch_size=1)
            self.save_tmp_images(real_A, real_B, synthetic_image_A, synthetic_image_B)

        self.saveModel(self.G_A2B, 200)
        self.saveModel(self.G_B2A, 200)
    '''
    #===============================================================================
    # Training
    def train(self, epochs, batch_size=1, save_interval=1):
        def run_training_iteration( epoch_iterations):
            # ======= Discriminator training ==========
                # Generate batch of synthetic images
            synthetic_images_B = self.G_A2B.predict(real_images_A)
            synthetic_images_A = self.G_B2A.predict(real_images_B)
            synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
            synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)

            for _ in range(self.discriminator_iterations):
                DA_loss_real = self.D_A.train_on_batch(x=real_images_A, y=ones)
                DB_loss_real = self.D_B.train_on_batch(x=real_images_B, y=ones)
                DA_loss_synthetic = self.D_A.train_on_batch(x=synthetic_images_A, y=zeros)
                DB_loss_synthetic = self.D_B.train_on_batch(x=synthetic_images_B, y=zeros)
                if self.use_multiscale_discriminator:
                    DA_loss = sum(DA_loss_real) + sum(DA_loss_synthetic)
                    DB_loss = sum(DB_loss_real) + sum(DB_loss_synthetic)
                    print('DA_losses: ', np.add(DA_loss_real, DA_loss_synthetic))
                    print('DB_losses: ', np.add(DB_loss_real, DB_loss_synthetic))
                else:
                    DA_loss = DA_loss_real + DA_loss_synthetic
                    DB_loss = DB_loss_real + DB_loss_synthetic
                D_loss = DA_loss + DB_loss

                if self.discriminator_iterations > 1:
                    print('D_loss:', D_loss)
                    sys.stdout.flush()

            # ======= Generator training ==========
            target_data = [real_images_A, real_images_B]  # Compare reconstructed images to real images
            if self.use_multiscale_discriminator:
                for i in range(2):
                    target_data.append(ones[i])
                    target_data.append(ones[i])
            else:
                target_data.append(ones)
                target_data.append(ones)

            if self.use_supervised_learning:
                target_data.append(real_images_A)
                target_data.append(real_images_B)

            for _ in range(self.generator_iterations):
                G_loss = self.G_model.train_on_batch(
                    x=[real_images_A, real_images_B], y=target_data)
                if self.generator_iterations > 1:
                    print('G_loss:', G_loss)
                    sys.stdout.flush()

            gA_d_loss_synthetic = G_loss[1]
            gB_d_loss_synthetic = G_loss[2]
            reconstruction_loss_A = G_loss[3]
            reconstruction_loss_B = G_loss[4]

            # Identity training
            '''
            if self.use_identity_learning and loop_index % self.identity_mapping_modulus == 0:
                G_A2B_identity_loss = self.G_A2B.train_on_batch(
                    x=real_images_B, y=real_images_B)
                G_B2A_identity_loss = self.G_B2A.train_on_batch(
                    x=real_images_A, y=real_images_A)
                print('G_A2B_identity_loss:', G_A2B_identity_loss)
                print('G_B2A_identity_loss:', G_B2A_identity_loss)
            '''
            # Update learning rates
            '''
            if self.use_linear_decay and epoch > self.decay_epoch:
                self.update_lr(self.D_A, decay_D)
                self.update_lr(self.D_B, decay_D)
                self.update_lr(self.G_model, decay_G)
            '''
            # Store training data
            DA_losses.append(DA_loss)
            DB_losses.append(DB_loss)
            gA_d_losses_synthetic.append(gA_d_loss_synthetic)
            gB_d_losses_synthetic.append(gB_d_loss_synthetic)
            gA_losses_reconstructed.append(reconstruction_loss_A)
            gB_losses_reconstructed.append(reconstruction_loss_B)

            GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
            GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
            D_losses.append(D_loss)
            GA_losses.append(GA_loss)
            GB_losses.append(GB_loss)
            G_losses.append(G_loss)
            reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
            reconstruction_losses.append(reconstruction_loss)

            print('\n')
            print('Epoch----------------', epoch, '/', epochs)
            #print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
            print('D_loss: ', D_loss)
            print('G_loss: ', G_loss[0])
            print('reconstruction_loss: ', reconstruction_loss)
            print('DA_loss:', DA_loss)
            print('DB_loss:', DB_loss)

            #if loop_index % 20 == 0:
                # Save temporary images continously
                #self.save_tmp_images(real_images_A, real_images_B, synthetic_images_A, synthetic_images_B)
                #self.print_ETA(start_time, epoch, epoch_iterations, loop_index)

        # ======================================================================
        # Begin training
        # ======================================================================
        training_history = OrderedDict()

        DA_losses = []
        DB_losses = []
        gA_d_losses_synthetic = []
        gB_d_losses_synthetic = []
        gA_losses_reconstructed = []
        gB_losses_reconstructed = []

        GA_losses = []
        GB_losses = []
        reconstruction_losses = []
        D_losses = []
        G_losses = []

        # Image pools used to update the discriminators
        synthetic_pool_A = ImagePool(self.synthetic_pool_size)
        synthetic_pool_B = ImagePool(self.synthetic_pool_size)

        # self.saveImages('(init)')

        # labels
        if self.use_multiscale_discriminator:
            label_shape1 = (batch_size,) + self.D_A.output_shape[0][1:]
            label_shape2 = (batch_size,) + self.D_A.output_shape[1][1:]
            #label_shape4 = (batch_size,) + self.D_A.output_shape[2][1:]
            ones1 = np.ones(shape=label_shape1) * self.REAL_LABEL
            ones2 = np.ones(shape=label_shape2) * self.REAL_LABEL
            #ones4 = np.ones(shape=label_shape4) * self.REAL_LABEL
            ones = [ones1, ones2]  # , ones4]
            zeros1 = ones1 * 0
            zeros2 = ones2 * 0
            #zeros4 = ones4 * 0
            zeros = [zeros1, zeros2]  # , zeros4]
        else:
            label_shape = (batch_size,) + self.D_A.output_shape[1:]
            ones = np.ones(shape=label_shape) * self.REAL_LABEL
            zeros = ones * 0

        # Linear decay
        '''
        if self.use_linear_decay:
            decay_D, decay_G = self.get_lr_linear_decay_rate()
        '''
        # Start stopwatch for ETAs
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            for batch_i, (real_images_A, real_images_B) in enumerate(self.data_loader.load_batch(batch_size)):
                run_training_iteration(  epoch_iterations = 1)
            if epoch % save_interval == 0:
                print('\n', '\n', '-------------------------Saving images for epoch', epoch, '-------------------------', '\n', '\n')
                self.saveImages(epoch, real_images_A, real_images_B)
        
            '''
            if self.use_data_generator:
                loop_index = 1
                for images in self.data_generator:
                    real_images_A = images[0]
                    real_images_B = images[1]
                    if len(real_images_A.shape) == 3:
                        real_images_A = real_images_A[:, :, :, np.newaxis]
                        real_images_B = real_images_B[:, :, :, np.newaxis]

                    # Run all training steps
                    run_training_iteration(loop_index, self.data_generator.__len__())

                    # Store models
                    if loop_index % 20000 == 0:
                        self.saveModel(self.D_A, loop_index)
                        self.saveModel(self.D_B, loop_index)
                        self.saveModel(self.G_A2B, loop_index)
                        self.saveModel(self.G_B2A, loop_index)

                    # Break if loop has ended
                    if loop_index >= self.data_generator.__len__():
                        break

                    loop_index += 1

            else:  # Train with all data in cache
                A_train = self.A_train
                B_train = self.B_train
                random_order_A = np.random.randint(len(A_train), size=len(A_train))
                random_order_B = np.random.randint(len(B_train), size=len(B_train))
                epoch_iterations = max(len(random_order_A), len(random_order_B))
                min_nr_imgs = min(len(random_order_A), len(random_order_B))

                # If we want supervised learning the same images form
                # the two domains are needed during each training iteration
                if self.use_supervised_learning:
                    random_order_B = random_order_A
                for loop_index in range(0, epoch_iterations, batch_size):
                    if loop_index + batch_size >= min_nr_imgs:
                        # If all images soon are used for one domain,
                        # randomly pick from this domain
                        if len(A_train) <= len(B_train):
                            indexes_A = np.random.randint(len(A_train), size=batch_size)
                            indexes_B = random_order_B[loop_index:
                                                        loop_index + batch_size]
                        else:
                            indexes_B = np.random.randint(len(B_train), size=batch_size)
                            indexes_A = random_order_A[loop_index:
                                                        loop_index + batch_size]
                    else:
                        indexes_A = random_order_A[loop_index:
                                                    loop_index + batch_size]
                        indexes_B = random_order_B[loop_index:
                                                    loop_index + batch_size]

                    sys.stdout.flush()
                    real_images_A = A_train[indexes_A]
                    real_images_B = B_train[indexes_B]

                    # Run all training steps
                    run_training_iteration(loop_index, epoch_iterations)
            '''
            #================== within epoch loop end ==========================
            '''
            if epoch % save_interval == 0:
                print('\n', '\n', '-------------------------Saving images for epoch', epoch, '-------------------------', '\n', '\n')
                self.saveImages(epoch, real_images_A, real_images_B)

            if epoch % 20 == 0:
                # self.saveModel(self.G_model)
                self.saveModel(self.D_A, epoch)
                self.saveModel(self.D_B, epoch)
                self.saveModel(self.G_A2B, epoch)
                self.saveModel(self.G_B2A, epoch)
            
            training_history = {
                'DA_losses': DA_losses,
                'DB_losses': DB_losses,
                'gA_d_losses_synthetic': gA_d_losses_synthetic,
                'gB_d_losses_synthetic': gB_d_losses_synthetic,
                'gA_losses_reconstructed': gA_losses_reconstructed,
                'gB_losses_reconstructed': gB_losses_reconstructed,
                'D_losses': D_losses,
                'G_losses': G_losses,
                'reconstruction_losses': reconstruction_losses}
            self.writeLossDataToFile(training_history)

            # Flush out prints each loop iteration
            sys.stdout.flush()
            '''
        self.saveModel(self.D_A, epochs)
        self.saveModel(self.D_B, epochs)
        self.saveModel(self.G_A2B, epochs)
        self.saveModel(self.G_B2A, epochs) 
    #===============================================================================
    # Help functions

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss
    
    def truncateAndSave(self, real_, real, synthetic, reconstructed, path_name):
        if len(real.shape) > 3:
            real = real[0]
            synthetic = synthetic[0]
            reconstructed = reconstructed[0]

        synthetic = synthetic.clip(min=0)
        reconstructed = reconstructed.clip(min=0)

        # Append and save
        if real_ is not None:
            if len(real_.shape) > 4:
                real_ = real_[0]
            image = np.hstack((real_[0], real, synthetic, reconstructed))
        else:
            image = np.hstack((real, synthetic, reconstructed))

        if self.channels == 1:
            image = image[:, :, 0]

        #toimage(image, cmin=0, cmax=1).save(path_name)
        Image.fromarray(image).save(path_name)
    
    def saveImages(self, epoch, real_image_A, real_image_B, num_saved_images=1):
        directory = os.path.join('images', self.date_time)
        if not os.path.exists(os.path.join(directory, 'A')):
            os.makedirs(os.path.join(directory, 'A'))
            os.makedirs(os.path.join(directory, 'B'))
            os.makedirs(os.path.join(directory, 'Atest'))
            os.makedirs(os.path.join(directory, 'Btest'))

        testString = ''

        real_image_Ab = None
        real_image_Ba = None
        for i in range(num_saved_images + 1):
            if i == num_saved_images:
                A_test, B_test = self.data_loader.load_data(batch_size= 1, is_testing=True)
                real_image_A = A_test
                real_image_B = B_test
                #real_image_A = np.expand_dims(real_image_A, axis=0)
                #real_image_B = np.expand_dims(real_image_B, axis=0)
                testString = 'test'
                if self.channels == 1:  # Use the paired data for MR images
                    real_image_Ab = B_test
                    real_image_Ba = A_test
                    #real_image_Ab = np.expand_dims(real_image_Ab, axis=0)
                    #real_image_Ba = np.expand_dims(real_image_Ba, axis=0)
            else:
                #real_image_A = self.A_train[rand_A_idx[i]]
                #real_image_B = self.B_train[rand_B_idx[i]]
                if len(real_image_A.shape) < 4:
                    real_image_A = np.expand_dims(real_image_A, axis=0)
                    real_image_B = np.expand_dims(real_image_B, axis=0)
                if self.channels == 1:  # Use the paired data for MR images
                    real_image_Ab = real_image_B  # self.B_train[rand_A_idx[i]]
                    real_image_Ba = real_image_A  # self.A_train[rand_B_idx[i]]
                    real_image_Ab = np.expand_dims(real_image_Ab, axis=0)
                    real_image_Ba = np.expand_dims(real_image_Ba, axis=0)

            synthetic_image_B = self.G_A2B.predict(real_image_A)
            synthetic_image_A = self.G_B2A.predict(real_image_B)
            reconstructed_image_A = np.squeeze(self.G_B2A.predict(synthetic_image_B))
            reconstructed_image_B = np.squeeze(self.G_A2B.predict(synthetic_image_A))
            synthetic_image_A = np.squeeze(synthetic_image_A)
            synthetic_image_B = np.squeeze(synthetic_image_B)
            real_image_A = np.squeeze(real_image_A)
            real_image_B = np.squeeze(real_image_B)
            #gen_imgs_A = np.concatenate([real_image_A, synthetic_image_B, reconstructed_image_A])
            #gen_imgs_B = np.concatenate([real_image_B, synthetic_image_A, reconstructed_image_B])
            #gen_imgs_A = 0.5 * gen_imgs_A + 0.5
            #gen_imgs_B = 0.5 * gen_imgs_B + 0.5
            synthetic_image_A = 0.5 * synthetic_image_A + 0.5
            reconstructed_image_A = 0.5 * synthetic_image_A + 0.5
            synthetic_image_B = 0.5 * synthetic_image_B + 0.5
            reconstructed_image_B = 0.5 * reconstructed_image_B + 0.5
            titles = ['Original', 'Translated', 'Reconstructed']
            r, c = 1, 3
            fig, axs = plt.subplots(r,c)
            cnt = 0
            #for i in range(r):
            axs[0].imshow(real_image_A, cmap = 'gray')
            axs[0].set_title('Original')
            axs[0].axis('off')
            axs[1].imshow(synthetic_image_B, cmap = 'gray')
            axs[1].set_title('Translated')
            axs[1].axis('off')
            axs[2].imshow(reconstructed_image_A, cmap = 'gray')
            axs[2].set_title('Reconstructed')
            axs[2].axis('off')
            # for j in range(c):
            #     print(gen_imgs_A[cnt].shape)
            #     axs[j].imshow(gen_imgs_A[cnt])
            #     axs[j].set_title(titles[j])
            #     axs[j].axis('off')
            #     cnt += 1
            fig.savefig('images/{}/{}/epoch{}_sample{}.png'.format(self.date_time, 'A' + testString, epoch, i))
            plt.close()
            cnt = 0
            fig, axs = plt.subplots(r,c)
            axs[0].imshow(real_image_B, cmap = 'gray')
            axs[0].set_title('Original')
            axs[0].axis('off')
            axs[1].imshow(synthetic_image_A, cmap = 'gray')
            axs[1].set_title('Translated')
            axs[1].axis('off')
            axs[2].imshow(reconstructed_image_B, cmap = 'gray')
            axs[2].set_title('Reconstructed')
            axs[2].axis('off')
            #for i in range(r):
            # for j in range(c):
            #     axs[j].imshow(gen_imgs_B[cnt])
            #     axs[j].set_title(titles[j])
            #     axs[j].axis('off')
            #     cnt += 1
            fig.savefig('images/{}/{}/epoch{}_sample{}.png'.format(self.date_time, 'B' + testString, epoch, i))
            plt.close()
            #self.truncateAndSave(real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A,'images/{}/{}/epoch{}_sample{}.png'.format(self.date_time, 'A' + testString, epoch, i))
            #self.truncateAndSave(real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B,'images/{}/{}/epoch{}_sample{}.png'.format(self.date_time, 'B' + testString, epoch, i))
    '''
    def save_tmp_images(self, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B):
        try:
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            real_images = np.vstack((real_image_A[0], real_image_B[0]))
            synthetic_images = np.vstack((synthetic_image_B[0], synthetic_image_A[0]))
            reconstructed_images = np.vstack((reconstructed_image_A[0], reconstructed_image_B[0]))

            self.truncateAndSave(None, real_images, synthetic_images, reconstructed_images,
                                 'images/{}/{}.png'.format(
                                     self.date_time, 'tmp'))
        except: # Ignore if file is open
            pass

    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        if self.use_data_generator:
            max_nr_images = len(self.data_generator)
        else:
            max_nr_images = max(len(self.A_train), len(self.B_train))

        updates_per_epoch_D = 2 * max_nr_images + self.discriminator_iterations - 1
        updates_per_epoch_G = max_nr_images + self.generator_iterations - 1
        if self.use_identity_learning:
            updates_per_epoch_G *= (1 + 1 / self.identity_mapping_modulus)
        denominator_D = (self.epochs - self.decay_epoch) * updates_per_epoch_D
        denominator_G = (self.epochs - self.decay_epoch) * updates_per_epoch_G
        decay_D = self.learning_rate_D / denominator_D
        decay_G = self.learning_rate_G / denominator_G

        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)

    def print_ETA(self, start_time, epoch, epoch_iterations, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * epoch_iterations + loop_index) / self.batch_size
        iterations_total = self.epochs * epoch_iterations / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Time passed', passed_time_string, ': ETA in', eta_string)

'''
#===============================================================================
# Save and load

    def saveModel(self, model, epoch):
        # Create folder to save model architecture and weights
        directory = os.path.join('saved_models', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = 'saved_models/{}/{}_weights_epoch_{}.hdf5'.format(self.date_time, model.name, epoch)
        model.save_weights(model_path_w)
        model_path_m = 'saved_models/{}/{}_model_epoch_{}.json'.format(self.date_time, model.name, epoch)
        model.save_weights(model_path_m)
        json_string = model.to_json()
        with open(model_path_m, 'w') as outfile:
            json.dump(json_string, outfile)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def writeLossDataToFile(self, history):
        keys = sorted(history.keys())
        with open('images/{}/loss_output.csv'.format(self.date_time), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))
'''
    def writeMetaDataToJSON(self):

        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save meta_data
        data = {}
        data['meta_data'] = []
        data['meta_data'].append({
            'img shape: height,width,channels': self.img_shape,
            'batch size': self.batch_size,
            'save interval': self.save_interval,
            'normalization function': str(self.normalization),
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'lambda_d': self.lambda_D,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'use linear decay on learning rates': self.use_linear_decay,
            'use multiscale discriminator': self.use_multiscale_discriminator,
            'epoch where learning rate linear decay is initialized (if use_linear_decay)': self.decay_epoch,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': self.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'REAL_LABEL': self.REAL_LABEL,
            'number of A train examples': len(self.A_train),
            'number of B train examples': len(self.B_train),
            'number of A test examples': len(self.A_test),
            'number of B test examples': len(self.B_test),
        })

        with open('images/{}/meta_data.json'.format(self.date_time), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

    def load_model_and_weights(self, model):
        path_to_model = os.path.join('generate_images', 'models', '{}.json'.format(model.name))
        path_to_weights = os.path.join('generate_images', 'models', '{}.hdf5'.format(model.name))
        #model = model_from_json(path_to_model)
        model.load_weights(path_to_weights)

    def load_model_and_generate_synthetic_images(self):
        response = input('Are you sure you want to generate synthetic images instead of training? (y/n): ')[0].lower()
        if response == 'y':
            self.load_model_and_weights(self.G_A2B)
            self.load_model_and_weights(self.G_B2A)
            synthetic_images_B = self.G_A2B.predict(self.A_test)
            synthetic_images_A = self.G_B2A.predict(self.B_test)

            def save_image(image, name, domain):
                if self.channels == 1:
                    image = image[:, :, 0]
                image = image.clip(min=0)
                toimage(image, cmin=0, cmax=1).save(os.path.join(
                    'generate_images', 'synthetic_images', domain, name))

            # Test A images
            for i in range(len(synthetic_images_A)):
                # Get the name from the image it was conditioned on
                name = self.testB_image_names[i].strip('.png') + '_synthetic.png'
                synt_A = synthetic_images_A[i]
                save_image(synt_A, name, 'A')

            # Test B images
            for i in range(len(synthetic_images_B)):
                # Get the name from the image it was conditioned on
                name = self.testA_image_names[i].strip('.png') + '_synthetic.png'
                synt_B = synthetic_images_B[i]
                save_image(synt_B, name, 'B')

            print('{} synthetic images have been generated and placed in ./generate_images/synthetic_images'
                  .format(len(self.A_test) + len(self.B_test)))


# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb

'''
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images

#%%
if __name__ == '__main__':
    GAN = CycleGAN()
    GAN.train(epochs = 100, batch_size=1, save_interval=1)

#%%
GAN = CycleGAN()
#%%
a, b = GAN.data_loader.load_data(batch_size=1, is_testing = False)
#GAN.G_model.summary()
#%%
#print(a.shape)
#%%
#A, B = test_model.data_loader.load_data(1)
#%%
#A, B = GAN.data_loader.load_data(1)
#print(A.shape)
#print(np.mean(A))
#%%
#ones = np.ones(shape = (1,16,16,1))
#print(ones.shape)
#%%
#GAN.D_A.train_on_batch(x = A, y = ones)
#%%
#GAN.D_A.summary()