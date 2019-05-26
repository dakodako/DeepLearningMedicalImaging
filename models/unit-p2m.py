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
class DataLoader():
    def __init__(self, dataset_name, img_res = (256,256)):
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
            img_B, img_A = img[:,:,:_w], img[:,:,_w:]
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
        data_type = "val" if not is_testing else "test"
        path = glob('datasets/%s/%s/*' % (self.dataset_name, data_type))
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
                img_B, img_A = img[:,:,:_w], img[:,:,_w:]
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
        #A_train = self.A_train 
        #B_train = self.B_train 
        self.epochs = epochs 
        self.batch_size = batch_size 
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
        for epoch in range(epochs + 1):
            print('epoch',epoch)
            for batch_i, (real_images_A, real_images_B) in enumerate(self.dataloader.load_batch(batch_size = 1)):
                print(real_images_A.shape)
                print(real_images_B.shape)
                run_training_iteration(batch_i, real_images_A, real_images_B)
        self.saveModel(self.discriminatorA, 'discriminatorA', epochs)
        self.saveModel(self.discriminatorB, 'discriminatorB', epochs)
        self.saveModel(self.generatorA, 'generatorA', epochs)
        self.saveModel(self.generatorB, 'generatorB', epochs)
        self.saveModel(self.encoderA, 'encoderA', epochs)
        self.saveModel(self.encoderB, 'encoderB', epochs)
        self.saveModel(self.decoderShared, 'decoderShared', epochs)
        self.saveModel(self.encoderShared, 'encoderShared', epochs)

    def saveModel(self, model, model_name, epoch):
        directory = os.path.join('saved_model', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = 'saved_model/{}/{}_epoch_{}_weights.hdf5'.format(self.date_time, model_name, epoch)
        model.save_weights(model_path_w)
        model_path_m = 'saved_model/{}/{}_epoch_{}_model.json'.format(self.date_time, model_name, epoch)
        model.save_weights(model_path_m)
    
        

            

    
#%%
UNIT = unit()
#%%
UNIT.discriminatorA.summary()
#%%
UNIT.encoderA.summary()
#%%
UNIT.encoderShared.summary()
#%%
UNIT.decoderShared.summary()
#%%
UNIT.generatorA.summary()

#%%
UNIT.train(epochs = 1)