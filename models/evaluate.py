#%%
from keras.layers import ZeroPadding2D, BatchNormalization, Input, MaxPooling2D, AveragePooling2D, Conv2D, LeakyReLU, Flatten, Conv2DTranspose, Activation, add, Lambda, GaussianNoise, merge, concatenate, Dropout, InputSpec, Layer, Activation, ZeroPadding2D, UpSampling2D, Flatten
import tensorflow as tf
from keras import initializers, regularizers, constraints
from keras import backend as K 
from keras.models import Model, load_model
from keras.layers.core import Dense, Flatten, Reshape
from keras.optimizers import Adam, adam 
from keras.activations import tanh 
from keras.regularizers import l2 
from keras.initializers import RandomNormal 
import nibabel as nib 
from keras.engine.topology import Network
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
import csv
import random
import matplotlib.pyplot as plt
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
        self.D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static_model')
        self.D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static_model')

        # ======= Generator model ==========
        # Do note update discriminator weights during generator training
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # Generators
        self.G_A2B = self.modelGenerator(name='G_A2B_model')
        self.G_B2A = self.modelGenerator(name='G_B2A_model')
        self.G_A2B.load_weights('models/saved_models/20190524-131743/G_A2B_model_weights_epoch_200.hdf5')
        self.G_B2A.load_weights('models/saved_models/20190524-131743/G_B2A_model_weights_epoch_200.hdf5')
        self.D_A.load_weights('models/saved_models/20190524-131743/D_A_model_weights_epoch_200.hdf5')
        self.D_B.load_weights('models/saved_models/20190524-131743/D_B_model_weights_epoch_200.hdf5')
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
    def evaluate(self, batch_size = 1):
        synthetic_pool_A = ImagePool(self.synthetic_pool_size)
        synthetic_pool_B = ImagePool(self.synthetic_pool_size)
        fakeAs = []
        fakeBs = []
        for batch_i, (real_images_B, real_images_A) in enumerate(self.data_loader.load_batch(batch_size = 1, is_testing = True)):
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
        data_type = "train" if not is_testing else "val"
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
C_GAN = CycleGAN()
#%%
GAN = unit() 
#%%
UNET = load_model('models/u-net-p2m_l2.h5')
#%%
encoderA = GAN.encoderA
encoderB = GAN.encoderB
encoderShared = GAN.encoderShared
decoderShared = GAN.decoderShared
generatorA = GAN.generatorA
generatorB = GAN.generatorB
#%%
encoderA.load_weights('models/saved_models/20190603-232225/encoderA_epoch_100_weights.hdf5')
encoderB.load_weights('models/saved_models/20190603-232225/encoderB_epoch_100_weights.hdf5')
encoderShared.load_weights('models/saved_models/20190603-232225/encoderShared_epoch_100_weights.hdf5')
decoderShared.load_weights('models/saved_models/20190603-232225/decoderShared_epoch_100_weights.hdf5')
generatorA.load_weights('models/saved_models/20190603-232225/generatorA_epoch_100_weights.hdf5')
generatorB.load_weights('models/saved_models/20190603-232225/generatorB_epoch_100_weights.hdf5')
#%%
dataloader = DataLoader(dataset_name = 'p2m8')
#%%
inputs = []
cycle_fake_As = []
unit_fake_As = []
unet_fake_As = []
#%%
for batch_i, (real_image_A, real_image_B) in enumerate(dataloader.load_batch(is_testing= True)):
    #print(real_image_A.shape)
    #print(real_image_B.shape)
    inputs.append(np.squeeze(real_image_B))
    encodedB = encoderB.predict(real_image_A)
    encodedShared = encoderShared.predict(encodedB)
    decodedShared = decoderShared.predict(encodedShared)
    fake_B = generatorA.predict(decodedShared)
    unit_fake_As.append(fake_B)
    cycle_fake_A = C_GAN.G_B2A.predict(real_image_A)
    cycle_fake_As.append(cycle_fake_A)
    real_image_B = (real_image_B + 1)/2
    unet_fake_As.append(UNET.predict(real_image_B))
#%%
print(np.min(real_image_B))
#%%
inputs = np.array(inputs)
#cycle_fake_As = np.array(cycle_fake_As)
unet_fake_As = np.array(unet_fake_As)
print(unet_fake_As.shape)
plt.imshow(np.squeeze(unet_fake_As[7,:,:,:,:]), cmap = 'gray')
#%%
np.save('cycle_fake_mp2rages_p2m4.npy', np.squeeze(cycle_fake_As))
#%%
unit_fake_As = np.array(unit_fake_As)
print(unit_fake_As.shape)
plt.imshow(np.squeeze(unit_fake_As[1,:,:,:,:]), cmap = 'gray')
#%%
np.save('unit_fake_mp2rages_p2m4.npy', np.squeeze(unit_fake_As))
#%%
np.save('unet_fake_mp2rages_p2m8.npy', np.squeeze(unet_fake_As))
#%%
unit_fake_As = np.squeeze(unit_fake_As)
cycle_fake_As = np.squeeze(cycle_fake_As)
unet_fake_As = np.squeeze(unet_fake_As)
fig, axs = plt.subplots(7,8, figsize=(40,40))
for row in range(7):
    for col in range(8):
        if row == 0:
            cur = axs[row, col].imshow(inputs[col, :, :], cmap = 'gray')
            cur.set_clim(-1,1)
            fig.colorbar(cur, ax = axs[row, col])
        if row == 1:
            cur = axs[row, col].imshow(unet_fake_As[col, :, :], cmap = 'gray')
            cur.set_clim(0,1)
            fig.colorbar(cur, ax = axs[row, col])
        if row == 2:
            cur = axs[row, col].imshow(cycle_fake_As[col, :, :], cmap = 'gray')
            cur.set_clim(-1,1)
            fig.colorbar(cur, ax = axs[row, col])
        if row == 3:
            cur = axs[row, col].imshow(unit_fake_As[col, :, :], cmap = 'gray')
            cur.set_clim(-1,1)
            fig.colorbar(cur, ax = axs[row, col])
        if row == 4:
            cur = axs[row, col].imshow(np.abs(inputs[col,:,:] - cycle_fake_As[col,:,:]))
            cur.set_clim(0,2)
            fig.colorbar(cur, ax = axs[row, col])
        if row == 5:
            cur = axs[row, col].imshow(np.abs(inputs[col,:,:] - unit_fake_As[col,:,:]))
            cur.set_clim(0,2)
            fig.colorbar(cur, ax = axs[row, col])
        if row == 6:
            inputs_unet = (inputs + 1)/2
            cur = axs[row, col].imshow(np.abs(inputs_unet[col, :, :] - unet_fake_As[col,:,:]))
            cur.set_clim(0,1)
            fig.colorbar(cur, ax = axs[row, col])
        
plt.savefig('results_plus_unet.png')
#%%
fig.colorbar()