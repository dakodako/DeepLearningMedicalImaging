#%%
from keras.layers import ZeroPadding2D, BatchNormalization, Input, MaxPooling2D, AveragePooling2D, Conv2D, LeakyReLU, Flatten, Conv2DTranspose, Activation, add, Lambda, GaussianNoise, merge, concatenate, Dropout, InputSpec, Layer
from keras import initializers, regularizers, constraints
from keras import backend as K 
from keras.models import Model, load_model
from keras.layers.core import Dense, Flatten, Reshape
from keras.optimizers import Adam, adam 
from keras.activations import tanh 
from keras.regularizers import l2 
from keras.initializers import RandomNormal 
import nibabel as nib 
from PIL import Image 
#from tensorflow.contrib.kfac.python.ops import optimizer
from collections import OrderedDict 
from time import localtime, strftime 
#from scipy.misc import imsave, toimage 
import numpy as np 
import json 
import sys 
import time 
import datetime 
from glob import glob
import os
#%%
import matplotlib.pyplot as plt
#%%
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

class unit():
    def __init__(self, lr = 1e-4, img_res = (256,256)):
        self.channels = 1
        self.img_shape = (img_res[0], img_res[1], self.channels)
        self.latent_dim = (int(self.img_shape[0] / 4), int(self.img_shape[1] / 4), 256)
        weight_decay = 0.0001/2 
        self.learning_rate = lr 
        self.beta_1 = 0.5 
        self.beta_2 = 0.999 
        self.lambda_0 = 10 
        self.lambda_1 = 0.1 
        self.lambda_2 = 100
        self.lambda_3 = self.lambda_1 
        self.lambda_4 = self.lambda_2 
        self.dataloader = DataLoader(dataset_name = 'p2m4', img_res = (256,256))
        opt = Adam(self.learning_rate, self.beta_1, self.beta_2)
        self.date_time = strftime("%Y%m%d-%H%M%S", localtime()) 
        # Discriminator
        self.discriminatorA = self.modelMultiDiscriminator("discriminatorA")
        self.discriminatorB = self.modelMultiDiscriminator("discriminatorB")

        for layer in self.discriminatorA.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= l2(weight_decay)
                layer.bias_regularizer = l2(weight_decay)
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02)
                layer.bias_initializer = RandomNormal(mean=0.0, stddev=0.02)

        for layer in self.discriminatorB.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= l2(weight_decay)
                layer.bias_regularizer = l2(weight_decay)
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02)
                layer.bias_initializer = RandomNormal(mean=0.0, stddev=0.02)
        self.discriminatorA.compile(optimizer=opt,
                                    loss=['binary_crossentropy',
                                          'binary_crossentropy',
                                          'binary_crossentropy'],
                                    loss_weights=[self.lambda_0,
                                                  self.lambda_0,
                                                  self.lambda_0])
        self.discriminatorB.compile(optimizer=opt,
                                    loss=['binary_crossentropy',
                                          'binary_crossentropy',
                                          'binary_crossentropy'],
                                    loss_weights=[self.lambda_0,
                                                  self.lambda_0,
                                                  self.lambda_0])
        self.encoderA = self.modelEncoder("encoderA")
        self.encoderB = self.modelEncoder("encoderB")
        self.encoderShared = self.modelSharedEncoder("encoderShared")
        self.decoderShared = self.modelSharedDecoder("decoderShared")
        self.generatorA = self.modelGenerator("generatorA")
        self.generatorB = self.modelGenerator("generatorB")
        imgA = Input(shape=(self.img_shape))
        imgB = Input(shape=(self.img_shape))
        encodedImageA = self.encoderA(imgA)
        encodedImageB = self.encoderB(imgB)

        sharedA = self.encoderShared(encodedImageA)
        sharedB = self.encoderShared(encodedImageB)

        outSharedA = self.decoderShared(sharedA)
        outSharedB = self.decoderShared(sharedB)
        # Input Generator
        outAa = self.generatorA(outSharedA)
        outBa = self.generatorA(outSharedB)

        outAb = self.generatorB(outSharedA)
        outBb = self.generatorB(outSharedB)

        guess_outBa = self.discriminatorA(outBa)
        guess_outAb = self.discriminatorB(outAb)

        # Cycle
        cycle_encodedImageA = self.encoderA(outBa)
        cycle_encodedImageB = self.encoderB(outAb)

        cycle_sharedA = self.encoderShared(cycle_encodedImageA)
        cycle_sharedB = self.encoderShared(cycle_encodedImageB)

        cycle_outSharedA = self.decoderShared(cycle_sharedA)
        cycle_outSharedB = self.decoderShared(cycle_sharedB)

        cycle_Ab_Ba = self.generatorA(cycle_outSharedB)
        cycle_Ba_Ab = self.generatorB(cycle_outSharedA)

        # Train only generators
        self.discriminatorA.trainable = False
        self.discriminatorB.trainable = False

        self.encoderGeneratorModel = Model(inputs=[imgA, imgB],
                              outputs=[sharedA, sharedB,
                                       cycle_sharedA, cycle_sharedB,
                                       outAa, outBb,
                                       cycle_Ab_Ba, cycle_Ba_Ab,
                                       guess_outBa[0], guess_outAb[0],
                                       guess_outBa[1], guess_outAb[1],
                                       guess_outBa[2], guess_outAb[2]])

        for layer in self.encoderGeneratorModel.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= l2(weight_decay)
                layer.bias_regularizer = l2(weight_decay)
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02)
                layer.bias_initializer = RandomNormal(mean=0.0, stddev=0.02)

        self.encoderGeneratorModel.compile(optimizer=opt,
                              loss=[self.vae_loss_CoGAN, self.vae_loss_CoGAN,
                                    self.vae_loss_CoGAN, self.vae_loss_CoGAN,
                                    'mae', 'mae',
                                    'mae', 'mae',
                                    'binary_crossentropy', 'binary_crossentropy',
                                    'binary_crossentropy', 'binary_crossentropy',
                                    'binary_crossentropy', 'binary_crossentropy'],
                              loss_weights=[self.lambda_1, self.lambda_1,
                                            self.lambda_3, self.lambda_3,
                                            self.lambda_2, self.lambda_2,
                                            self.lambda_4, self.lambda_4,
                                            self.lambda_0, self.lambda_0,
                                            self.lambda_0, self.lambda_0,
                                            self.lambda_0, self.lambda_0])
    def resblk(self, x0, k):
        # first layer
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding="same")(x0)
        x = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-05, center=True)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding="same")(x)
        x = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-05, center=True)(x, training=True)
        x = Dropout(0.5)(x, training=True)
        # merge
        x = add([x, x0])

        return x
    def vae_loss_CoGAN(self, y_true, y_pred):
        y_pred_2 = K.square(y_pred)
        encoding_loss = K.mean(y_pred_2)
        return encoding_loss
    def modelMultiDiscriminator(self, name):
        x1 = Input(shape=self.img_shape)
        x2 = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x1)
        x4 = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x2)

        x1_out = self.modelDiscriminator(x1)
        x2_out = self.modelDiscriminator(x2)
        x4_out = self.modelDiscriminator(x4)

        return Model(inputs=x1, outputs=[x1_out, x2_out, x4_out], name=name)

    def modelDiscriminator(self, x):
        # Layer 1
        x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 2
        x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 3
        x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 4
        x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 5
        x = Conv2D(1, kernel_size=1, strides=1)(x)
        prediction = Activation('sigmoid')(x)

        return prediction
    def modelEncoder(self, name):
        inputImg = Input(shape=self.img_shape)
        # Layer 1
        x = ZeroPadding2D(padding=(3, 3))(inputImg)
        x = Conv2D(64, kernel_size=7, strides=1, padding='valid')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 2
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(128, kernel_size=3, strides=2, padding='valid')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 3
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(256, kernel_size=3, strides=2, padding='valid')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 4: 2 res block
        x = self.resblk(x, 256)
        # Layer 5: 3 res block
        x = self.resblk(x, 256)
        # Layer 6: 3 res block
        z = self.resblk(x, 256)

        return Model(inputs=inputImg, outputs=z, name=name)
    def modelSharedEncoder(self, name):
        input = Input(shape=self.latent_dim)

        x = self.resblk(input, 256)
        z = GaussianNoise(stddev=1)(x, training=True)

        return Model(inputs=input, outputs=z, name=name)
    def modelSharedDecoder(self, name):
        input = Input(shape=self.latent_dim)

        x = self.resblk(input, 256)

        return Model(inputs=input, outputs=x, name=name)
    def modelGenerator(self, name):
        inputImg = Input(shape=self.latent_dim)
        # Layer 1: 1 res block
        x = self.resblk(inputImg, 256)
        # Layer 2: 2 res block
        x = self.resblk(x, 256)
        # Layer 3: 3 res block
        x = self.resblk(x, 256)
        # Layer 4:
        x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 5:
        x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 6
        x = Conv2DTranspose(self.channels, kernel_size=1, strides=1, padding='valid')(x)
        z = Activation("tanh")(x)

        return Model(inputs=inputImg, outputs=z, name=name)
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
            m_A = np.max(img_A)
            mi_A = np.min(img_A)
            img_A = (img_A - mi_A)/(m_A - mi_A) 
            m_B = np.max(img_B)
            mi_B = np.min(img_B)
            img_B = (img_B - mi_B)/(m_B - mi_B)
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
GAN = unit()
#%%
encoderA = GAN.encoderA
encoderB = GAN.encoderB
encoderShared = GAN.encoderShared
decoderShared = GAN.decoderShared
generatorA = GAN.generatorA
generatorB = GAN.generatorB
#%%
encoderA.load_weights('models/saved_models/20190526-175240/encoderA_epoch_100_weights.hdf5')
encoderB.load_weights('models/saved_models/20190526-175240/encoderB_epoch_100_weights.hdf5')
encoderShared.load_weights('models/saved_models/20190526-175240/encoderShared_epoch_100_weights.hdf5')
decoderShared.load_weights('models/saved_models/20190526-175240/decoderShared_epoch_100_weights.hdf5')
generatorA.load_weights('models/saved_models/20190526-175240/generatorA_epoch_100_weights.hdf5')
generatorB.load_weights('models/saved_models/20190526-175240/generatorB_epoch_100_weights.hdf5')
#%%
dataloader = DataLoader(dataset_name = 'p2m4')
a,b = dataloader.load_data(1, is_testing= True)
plt.imshow(np.squeeze(a))
# a is petra # b is mp2rage
#%%
encodedB = encoderB.predict(a)
encodedShared = encoderShared.predict(encodedB)
decodedShared = decoderShared.predict(encodedShared)
fake_B = generatorA.predict(decodedShared)
#%%
plt.imshow(np.squeeze(fake_B))
#%%
plt.imshow(np.squeeze(b))
#%%
u_net_model = load_model('models/u-net-p2m_l2_2.h5')
#%%
a_unet = (a + 1)/2.0
u_net_pred = u_net_model.predict(a_unet)
#%%
plt.imshow(np.squeeze(u_net_pred))