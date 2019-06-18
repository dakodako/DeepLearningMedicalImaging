#%%
import tensorflow as tf 
import keras
from keras import backend as K 
from keras import initializers, regularizers, constraints
from keras.layers import Layer, InputSpec, Input, Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, Lambda, GaussianNoise, merge, concatenate, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Flatten, Reshape
from keras.optimizers import Adam, adam
from keras.models import Model, load_model 
from keras.activations import tanh 
from keras.regularizers import l2 
from keras.initializers import RandomNormal 
from keras.backend import mean 
from keras.utils import plot_model, Sequence
if keras.__version__ >= '2.2.4':
    from keras.engine.topology import Network
else:
    from keras.engine.topology import Container
#%%
import numpy as np 
import random 
import datetime  
import time  
import json 
import math 
import sys 
import csv  
import os 
from glob import glob 
from time import localtime, strftime

#%%
import nibabel as nib 
from PIL import Image 
from collections import OrderedDict 
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
    def load_data(self, batch_size = 1, is_testing = False):
        data_type = 'train' if not is_testing else 'test' 
        path = glob('datasets/%s/%s/*' % (self.dataset_name, data_type))
        batch_images = np.random.choice(path, size = batch_size)
        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = nib.load(img_path)
            img = img.get_data()
            _,_,w = img.shape 
            _w = int(w/2)
            # img_B is petra and img_A is mp2
            img_B, img_A = img[:,:,:_w], img[:,:,_w:]
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
    def load_batch(self, batch_size = 1, is_testing = False):
        data_type = "train" if not is_testing else "test"
        path = glob('datasets/%s/%s/*' % (self.dataset_name, data_type))
        self.n_batches = int(len(path)/batch_size)
        for i in range(self.n_batches - 1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = nib.load(img)
                img = img.get_data()
                _,_,w = img.shape
                _w = int(w/2)
                img_B, img_A = img[:,:,:_w], img[:,:,_w:]
                #img_A, img_B = img[:,:,_w:],img[:,:,:_w]
                img_A = np.squeeze(img_A)
                img_B = np.squeeze(img_B)
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
class UNIT():
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
        self.dataloader = DataLoader(dataset_name = 'p2m7', img_res = (256,256))
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

    def evaluate(self, paths,batch_size = 1):
        if (len(paths) < 6):
            print("Not enough model weights")
        self.encoderA.load_weights(paths[0])
        self.encoderB.load_weights(paths[1])
        self.encoderShared.load_weights(paths[2])
        self.decoderShared.load_weights(paths[3])
        self.generatorA.load_weights(paths[4])
        self.generatorB.load_weights(paths[5])
        label_shape1 = (batch_size,) + self.discriminatorA.output_shape[0][1:]
        label_shape2 = (batch_size,) + self.discriminatorA.output_shape[1][1:]
        label_shape3 = (batch_size,) + self.discriminatorA.output_shape[2][1:]
        real_labels1 = np.ones(label_shape1)
        real_labels2 = np.ones(label_shape2)
        real_labels3 = np.ones(label_shape3)
        synthetic_labels1 = np.zeros(label_shape1)
        synthetic_labels2 = np.zeros(label_shape2)
        synthetic_labels3 = np.zeros(label_shape3)
        dummy = np.zeros(shape = ((batch_size,) + self.latent_dim))
        real_labels = [real_labels1, real_labels2, real_labels3]
        synthetic_labels = [synthetic_labels1, synthetic_labels2, synthetic_labels3]
        inputs = []
        ground_truth = []
        outputs = []
        for batch_i, (mp2_outputs, petra_inputs) in enumerate(self.dataloader.load_batch(batch_size= 1, is_testing=True)):
            inputs.append(np.squeeze(petra_inputs))
            encodedImageB = self.encoderB.predict(petra_inputs)
            encodedImageA = self.encoderA.predict(mp2_outputs)
            sharedA = self.encoderShared.predict(encodedImageA)
            sharedB = self.encoderShared.predict(encodedImageB)
            outSharedA = self.decoderShared.predict(sharedA)
            outSharedB = self.decoderShared.predict(sharedB)
            outAa = self.generatorA.predict(outSharedA)
            outBa = self.generatorA.predict(outSharedB) # what we want
            outAb = self.generatorB.predict(outSharedA)
            outBb = self.generatorB.predict(outSharedB)
            outputs.append(np.squeeze(outBa))
            ground_truth.append(np.squeeze(mp2_outputs))

            dA_loss_real = self.discriminatorA.evaluate(mp2_outputs, real_labels)
            dA_loss_fake = self.discriminatorA.evaluate(outBa, synthetic_labels)
            dA_loss = np.add(dA_loss_real, dA_loss_fake)
            dB_loss_real = self.discriminatorB.evaluate(petra_inputs, real_labels)
            dB_loss_fake = self.discriminatorB.evaluate(outAb, synthetic_labels)
            dB_loss = np.add(dB_loss_fake, dB_loss_real)
            g_loss = self.encoderGeneratorModel.evaluate([mp2_outputs, petra_inputs], [dummy, dummy,dummy, dummy, mp2_outputs, petra_inputs, mp2_outputs, petra_inputs, real_labels1, real_labels1,real_labels2, real_labels2, real_labels3, real_labels3])
            print('batch ', batch_i)
            print('Discriminator TOTAL loss: ', dA_loss[0] + dB_loss[0])
            print('Discriminator A loss total: ', dA_loss[0])
            print('Discriminator B loss total: ', dB_loss[0])
            print('Genarator loss total: ', g_loss[0])
            print('----------------Discriminator loss----')
            print('dA_loss_real: ', dA_loss_real[0])
            print('dA_loss_fake: ', dA_loss_fake[0])
            print('dB_loss_real: ', dB_loss_real[0])
            print('dB_loss_fake: ', dB_loss_fake[0])
            print('----------------Generator loss--------')
            print('Shared A: ', g_loss[1])
            print('Shared B: ', g_loss[2])
            print('Cycle shared A: ', g_loss[3])
            print('Cycle shared B: ', g_loss[4])
            print('OutAa MAE: ', g_loss[5])
            print('OutBb MAE: ', g_loss[6])
            print('Cycle_Ab_Ba MAE: ', g_loss[7])
            print('Cycle_Ba_Ab MAE: ', g_loss[8])
            print('guess_outBa: ', g_loss[9])
            print('guess_outAb: ', g_loss[10])
            print('guess_outBa: ', g_loss[11])
            print('guess_outAb: ', g_loss[12])
            print('guess_outBa: ', g_loss[13])
            print('guess_outAb: ', g_loss[14])
        return inputs, ground_truth, outputs
    def train(self, epochs = 100, batch_size = 1, save_interval = 1):
        def run_training_iteration(loop_index, imgA, imgB):
            encodedImageA = self.encoderA.predict(imgA)
            encodedImageB = self.encoderB.predict(imgB)

            sharedA = self.encoderShared.predict(encodedImageA)
            sharedB = self.encoderShared.predict(encodedImageB)

            outSharedA = self.decoderShared.predict(sharedA)
            outSharedB = self.decoderShared.predict(sharedB)

            outAa = self.generatorA.predict(outSharedA)
            outBa = self.generatorA.predict(outSharedB)

            outAb = self.generatorB.predict(outSharedA)
            outBb = self.generatorB.predict(outSharedB)
            # Train discriminator
            dA_loss_real = self.discriminatorA.train_on_batch(imgA, real_labels)
            dA_loss_fake = self.discriminatorA.train_on_batch(outBa, synthetic_labels)
            dA_loss = np.add(dA_loss_real, dA_loss_fake)

            dB_loss_real = self.discriminatorB.train_on_batch(imgB, real_labels)
            dB_loss_fake = self.discriminatorB.train_on_batch(outAb, synthetic_labels)
            dB_loss = np.add(dB_loss_real, dB_loss_fake)
            # Train generator
            g_loss = self.encoderGeneratorModel.train_on_batch([imgA, imgB],
                                                  [dummy, dummy,
                                                   dummy, dummy,
                                                   imgA, imgB,
                                                   imgA, imgB,
                                                   real_labels1, real_labels1,
                                                   real_labels2, real_labels2,
                                                   real_labels3, real_labels3])
            print('----------------Epoch-------640x480---------', epoch, '/', epochs - 1)
            #print('----------------Loop index-----------', loop_index, '/', epoch_iterations - 1)
            print('Discriminator TOTAL loss: ', dA_loss[0] + dB_loss[0])
            print('Discriminator A loss total: ', dA_loss[0])
            print('Discriminator B loss total: ', dB_loss[0])
            print('Genarator loss total: ', g_loss[0])
            print('----------------Discriminator loss----')
            print('dA_loss_real: ', dA_loss_real[0])
            print('dA_loss_fake: ', dA_loss_fake[0])
            print('dB_loss_real: ', dB_loss_real[0])
            print('dB_loss_fake: ', dB_loss_fake[0])
            print('----------------Generator loss--------')
            print('Shared A: ', g_loss[1])
            print('Shared B: ', g_loss[2])
            print('Cycle shared A: ', g_loss[3])
            print('Cycle shared B: ', g_loss[4])
            print('OutAa MAE: ', g_loss[5])
            print('OutBb MAE: ', g_loss[6])
            print('Cycle_Ab_Ba MAE: ', g_loss[7])
            print('Cycle_Ba_Ab MAE: ', g_loss[8])
            print('guess_outBa: ', g_loss[9])
            print('guess_outAb: ', g_loss[10])
            print('guess_outBa: ', g_loss[11])
            print('guess_outAb: ', g_loss[12])
            print('guess_outBa: ', g_loss[13])
            print('guess_outAb: ', g_loss[14])
            D_loss.append(dA_loss[0] + dB_loss[0])
            DA_loss.append(dA_loss[0])
            DB_loss.append(dB_loss[0])
            G_loss.append(g_loss[0])
            sA_loss.append(g_loss[1])
            sB_loss.append(g_loss[2])
            csA_loss.append(g_loss[3])
            csB_loss.append(g_loss[4])
            out_Aa_mae.append(g_loss[5])
            out_Bb_mae.append(g_loss[6])
            c_Ab_Ba_mae.append(g_loss[7])
            c_Ba_Ab_mae.append(g_loss[8])
            guess_outBa.append(g_loss[9])
            guess_outAb.append(g_loss[10])
            dA_loss_r.append(dA_loss_real[0])
            dB_loss_r.append(dB_loss_real[0])
            dA_loss_f.append(dA_loss_fake[0])
            dB_loss_f.append(dB_loss_fake[0])
        #A_train = self.A_train 
        #B_train = self.B_train 
        self.epochs = epochs 
        self.batch_size = batch_size 
        D_loss = []
        DA_loss = []
        DB_loss = []
        G_loss = []
        sA_loss = []
        sB_loss = []
        csA_loss = []
        csB_loss = []
        out_Aa_mae = []
        out_Bb_mae = []
        c_Ab_Ba_mae = []
        c_Ba_Ab_mae = []
        dA_loss_f = []
        dB_loss_f = []
        dA_loss_r = []
        dB_loss_r = []
        guess_outBa = []
        guess_outAb = []
        dummy = np.zeros(shape = ((self.batch_size,) + self.latent_dim))
        label_shape1 = (batch_size,) + self.discriminatorA.output_shape[0][1:]
        label_shape2 = (batch_size,) + self.discriminatorA.output_shape[1][1:]
        label_shape3 = (batch_size,) + self.discriminatorA.output_shape[2][1:]

        real_labels1 = np.ones(label_shape1)
        real_labels2 = np.ones(label_shape2)
        real_labels3 = np.ones(label_shape3)
        synthetic_labels1 = np.zeros(label_shape1)
        synthetic_labels2 = np.zeros(label_shape2)
        synthetic_labels3 = np.zeros(label_shape3)

        real_labels = [real_labels1, real_labels2, real_labels3]
        synthetic_labels = [synthetic_labels1, synthetic_labels2, synthetic_labels3]
        for epoch in range(epochs):
            for batch_i, (real_images_A, real_images_B) in enumerate(self.dataloader.load_batch(batch_size = 1)):
                run_training_iteration(batch_i, real_images_A, real_images_B)
        self.saveModel(self.discriminatorA, 'discriminatorA', epochs)
        self.saveModel(self.discriminatorB, 'discriminatorB', epochs)
        self.saveModel(self.generatorA, 'generatorA', epochs)
        self.saveModel(self.generatorB, 'generatorB', epochs)
        self.saveModel(self.encoderA, 'encoderA', epochs)
        self.saveModel(self.encoderB, 'encoderB', epochs)
        self.saveModel(self.decoderShared, 'decoderShared', epochs)
        self.saveModel(self.encoderShared, 'encoderShared', epochs)
        np.savetxt('saved_model/{}/D_loss.txt'.format(self.date_time), D_loss)
        np.savetxt('saved_model/{}/DA_loss.txt'.format(self.date_time), DA_loss)
        np.savetxt('saved_model/{}/DB_loss.txt'.format(self.date_time), DB_loss)
        np.savetxt('saved_model/{}/G_loss.txt'.format(self.date_time), G_loss)
        np.savetxt('saved_model/{}/sA_loss.txt'.format(self.date_time), sA_loss)
        np.savetxt('saved_model/{}/sB_loss.txt'.format(self.date_time), sB_loss)
        np.savetxt('saved_model/{}/csA_loss.txt'.format(self.date_time), csA_loss)
        np.savetxt('saved_model/{}/csB_loss.txt'.format(self.date_time), csB_loss)
        np.savetxt('saved_model/{}/out_Aa_mae.txt'.format(self.date_time), out_Aa_mae)
        np.savetxt('saved_model/{}/out_Bb_mae.txt'.format(self.date_time), out_Bb_mae)
        np.savetxt('saved_model/{}/c_Ab_Ba_mae.txt'.format(self.date_time), c_Ab_Ba_mae)
        np.savetxt('saved_model/{}/c_Ba_Ab_mae.txt'.format(self.date_time), c_Ba_Ab_mae)
        np.savetxt('saved_model/{}/dA_loss_f.txt'.format(self.date_time), dA_loss_f)
        np.savetxt('saved_model/{}/dB_loss_f.txt'.format(self.date_time), dB_loss_f)
        np.savetxt('saved_model/{}/dA_loss_r.txt'.format(self.date_time), dA_loss_r)
        np.savetxt('saved_model/{}/dB_loss_r.txt'.format(self.date_time), dB_loss_r)
        np.savetxt('saved_model/{}/guess_outBa.txt'.format(self.date_time), guess_outBa)
        np.savetxt('saved_model/{}/guess_outAb.txt'.format(self.date_time), guess_outAb)
    def saveModel(self, model, model_name, epoch):
        directory = os.path.join('saved_model', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = 'saved_model/{}/{}_epoch_{}_weights.hdf5'.format(self.date_time, model_name, epoch)
        model.save_weights(model_path_w)
        model_path_m = 'saved_model/{}/{}_epoch_{}_model.json'.format(self.date_time, model_name, epoch)
        model.save_weights(model_path_m)

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
        self.epochs = 200 # choose multiples of 25 since the models are save each 25th epoch
        self.save_interval = 1
        self.synthetic_pool_size = 50
        self.data_loader = DataLoader(dataset_name = 'p2m7', img_res = (256,256))
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
        if keras.__version__ >= '2.2.4':
            self.D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static_model')
            self.D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static_model')
        else:
            self.D_A_static = Container(inputs=image_A, outputs=guess_A, name='D_A_static_model')
            self.D_B_static = Container(inputs=image_B, outputs=guess_B, name='D_B_static_model')

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
    def evaluate(self, path, batch_size = 1):
        print('loading model weights')
        self.G_A2B.load_weights(path[0])
        self.G_B2A.load_weights(path[1])
        self.D_A.load_weights(path[2])
        self.D_B.load_weights(path[3])
        synthetic_pool_A = ImagePool(self.synthetic_pool_size)
        synthetic_pool_B = ImagePool(self.synthetic_pool_size)
        fakeAs = []
        fakeBs = []
        print('start evaluating')
        for batch_i, (real_images_A, real_images_B) in enumerate(self.data_loader.load_batch(batch_size = 1, is_testing = True)):
            print('batch_i:', batch_i)
            label_shape = (batch_size,) + self.D_A.output_shape[1:]
            ones = np.ones(shape=label_shape) * self.REAL_LABEL
            zeros = ones * 0
            synthetic_images_B = self.G_A2B.predict(real_images_A)
            fakeBs.append(np.squeeze(synthetic_images_B))
            synthetic_images_A = self.G_B2A.predict(real_images_B)
            fakeAs.append(np.squeeze(synthetic_images_A))
            synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
            synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)
            DA_loss_real = self.D_A.evaluate(x=real_images_A, y=ones)
            DB_loss_real = self.D_B.evaluate(x=real_images_B, y=ones)
            DA_loss_synthetic = self.D_A.evaluate(x=synthetic_images_A, y=zeros)
            DB_loss_synthetic = self.D_B.evaluate(x=synthetic_images_B, y=zeros)
            DA_loss = DA_loss_real + DA_loss_synthetic
            DB_loss = DB_loss_real + DB_loss_synthetic
            D_loss = DA_loss + DB_loss
            print('D_loss:', D_loss)
            target_data = [real_images_A, real_images_B]
            target_data.append(ones)
            target_data.append(ones)
            G_loss = self.G_model.evaluate(x=[real_images_A, real_images_B], y=target_data)
            print('G_loss:', G_loss)
            gA_d_loss_synthetic = G_loss[1]
            gB_d_loss_synthetic = G_loss[2]
            reconstruction_loss_A = G_loss[3]
            reconstruction_loss_B = G_loss[4]
            reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
            GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
            GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
            print('\n')
            #print('Epoch----------------', epoch, '/', epochs)
            #print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
            print('D_loss: ', D_loss)
            print('G_loss: ', G_loss[0])
            print('reconstruction_loss: ', reconstruction_loss)
            print('DA_loss:', DA_loss)
            print('DB_loss:', DB_loss)
        return fakeAs, fakeBs
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
        for epoch in range(1, epochs + 1):
            for batch_i, (real_images_A, real_images_B) in enumerate(self.data_loader.load_batch(batch_size)):
                run_training_iteration(  epoch_iterations = 1)
            #if epoch % save_interval == 0:
                #print('\n', '\n', '-------------------------Saving images for epoch', epoch, '-------------------------', '\n', '\n')
                #self.saveImages(epoch, real_images_A, real_images_B)
        
        
        self.saveModel(self.D_A, epochs)
        self.saveModel(self.D_B, epochs)
        self.saveModel(self.G_A2B, epochs)
        self.saveModel(self.G_B2A, epochs) 
    
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

