#%%
import tensorflow as tf 
#from keras.engine.topology import Layer
#from keras.engine import InputSpec
from keras.layers import Input, Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
#from keras_contrib.layers.normalization.instancenormalization import InputSpec
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import Model
from keras.layers.core import Dense
from keras.engine.topology import Network
from keras import initializers, regularizers, constraints
from keras.layers import Layer, InputSpec
from keras import backend as K
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
class ReflectionPadding2D(Layer):
    def __init__(self, padding = (1,1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

class CycleGAN():
    def __init__(self, lr_D = 2e-4, lr_G =2e-4, img_shape = (256,256,1), use_patchgan = True):
        self.normalization = InstanceNormalization
        self.shape = img_shape
        self.nchannels = self.shape[-1]
        self.use_patchgan = use_patchgan
        self.learning_rate_D = lr_D
        self.learning_rate_G = lr_G 
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = 1
        self.epochs = 200
        self.lambda_1 = 10.0 # cyclic loss weight A_2_B
        self.lambda_2 = 10.0 # cyclic loss weight B_2_A
        self.lambda_D = 1.0
        # optimizer
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)
        # Build discriminator
        D_A = self.build_discriminator()
        D_B = self.build_discriminator()
        loss_weights_D = [0.5, 0.5]

        image_A = Input(shape = self.shape)
        image_B = Input(shape = self.shape)
        guess_A = D_A(image_A)
        guess_B = D_B(image_B)
        self.D_A = Model(inputs = image_A, outputs = guess_A, name = 'D_A_model')
        self.D_B = Model(inputs = image_B, outputs = guess_B, name = 'D_B_model')
        self.D_A.compile(optimizer = self.opt_D, loss = self.lse, loss_weights = loss_weights_D)
        self.D_B.compile(optimizer=self.opt_D, loss = self.lse, loss_weights = loss_weights_D)
        self.D_A_static = Network(inputs = image_A, outputs = guess_A, name = 'D_A_static_model')
        self.D_B_static = Network(inputs = image_B, outputs = guess_B, name = 'D_B_static_model')

        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # build generators
        self.G_A2B = self.build_generator(name = 'G_A2B_model')
        self.G_B2A = self.build_generator(name = 'G_B2A_model')

        real_A = Input(shape = self.shape, name = 'real_A')
        real_B = Input(shape = self.shape, name = 'real_B')
        fake_A = self.G_B2A(real_B)
        fake_B = self.G_A2B(real_A)
        dA_guess_fake = self.D_A_static(fake_A)
        dB_guess_fake = self.D_B_static(fake_B)
        
        reconstructed_A = self.G_B2A(fake_B)
        reconstructed_B = self.G_A2B(fake_A)

        model_outputs = [reconstructed_A, reconstructed_B]
        compile_loss = [self.cycle_loss, self.cycle_loss, self.lse, self.lse]
        compile_weights = [self.lambda_1, self.lambda_2, self.lambda_D, self.lambda_D]
        model_outputs.append(dA_guess_fake)
        model_outputs.append(dB_guess_fake)
    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred-y_true))
        return loss
    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss
    
    def c7s1_k(self, x, k):
        """
        Let c7s1-k denote a 7 × 7 Convolution-InstanceNorm- ReLU layer with k filters and stride 1
        """
        x = Conv2D(filters = k, kernel_size = 7, strides = 1, padding = 'valid')(x)
        x = self.normalization(axis = 3, center = True, epsilon = 1e-5)(x, training = True)
        x = Activation('relu')(x)
        return x
    def dk(self, x, k):
        """
        dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2
        """
        x = Conv2D(filters = k, kernel_size = 3, strides = 2, padding = 'same')(x)
        x = self.normalization(axis = 3, center = True, epsilon=1e-5)(x, training = True)
        x = Activation('relu')(x)
        return x

    def rk(self, x0):
        """
        Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layer.
        """
        k = int(x0.shape[-1])
        x = Conv2D(filters = k, kernel_size = 3, strides = 1, padding = 'same')(x0)
        x = self.normalization(axis=3, center = True, epsilon = 1e-5)(x, training = True)
        x = Activation('relu')(x)
        x = Conv2D(filters = k, kernel_size = 3, strides = 1, padding = 'same')(x)
        x = self.normalization(axis = 3, center = True, epsilon = 1e-5)(x, training = True)
        x = add([x, x0])
        return x

    def uk(self, x, k):
        """
        uk denotes a 3 × 3 fractional-strided-Convolution- InstanceNorm-ReLU layer with k filters and stride 1/2
        """

        x = Conv2DTranspose(filters = k, kernel_size = 3, strides = 2, padding = 'same')(x)
        x = self.normalization(axis = 3, center = True, epsilon=1e-5)(x, training = True)
        x = Activation('relu')(x)
        return x 
    def ck(self, x, k, use_normalization):
        """
        Ck denote a 4x4 convolution-InstanceNorm-LekyReLU layer with k filters and stride 2
        """
        x = Conv2D(filters = k, kernel_size = 4, strides = 2, padding = 'same')(x)
        if use_normalization:
            x = self.normalization(axis = 3, center = True, epsilon=1e-5)(x, training = True)
        x = LeakyReLU(alpha = 0.2)(x)
        return x

    def build_generator(self, name = None):
        input_img = Input(shape = self.shape)
        """
        c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
        """
        x = ReflectionPadding2D((3,3))(input_img)
        x = self.c7s1_k(x, 64)
        x = self.dk(x, 128)
        x = self.dk(x, 256)
        x = self.rk(x)
        x = self.rk(x)
        x = self.rk(x)
        x = self.rk(x)
        x = self.rk(x)
        x = self.rk(x)
        x = self.uk(x, 128)
        x = self.uk(x, 64)
        x = ReflectionPadding2D((3,3))(x)
        x = self.c7s1_k(x, self.nchannels) # use tanh for activation instead NEEDS to be CHANGED LATER!!!!
        return Model(inputs = input_img, outputs = x, name = name)
    def build_discriminator(self, name = None):
        """
        C64-C128-C256-C512
        """
        input_img = Input(shape = self.shape)
        x = self.ck(input_img, 64, False)
        x = self.ck(x, 128, True)
        x = self.ck(x, 128, True)
        x = self.ck(x, 256, True)
        x = self.ck(x, 512, True)
        if self.use_patchgan:
            x = Conv2D(filters = 1, kernel_size = 4, strides = 1, padding = 'same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        return Model(inputs = input_img, outputs = x, name = name)
#%%
test_model = CycleGAN((256,256, 1), use_patchgan= False)
#%%
test_generator = test_model.build_generator()
#%%
test_generator.summary()
#%%
test_discriminator = test_model.build_discriminator()
#%%
test_discriminator.summary()
#%%
test_input = Input(shape = (256,256,1))
test_x = ReflectionPadding2D((4,4))(test_input)
test_model = Model(inputs = test_input, outputs = test_x)
test_model.summary()