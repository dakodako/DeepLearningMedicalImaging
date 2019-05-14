#%%
import tensorflow as tf 
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.layers import Layer, Input, Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D
from keras_contrib.layers.normalization.instancenormalization import InputSpec
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import Model
#%%
class ReflectionPadding2D(Layer):
    def __init__(self, padding = (1,1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
class CycleGAN():
    def __init__(self, img_shape):
        self.normalization = InstanceNormalization
        self.shape = img_shape
        self.nchannels = self.shape[-1]
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
        return x

    def uk(self, x, k):
        """
        uk denotes a 3 × 3 fractional-strided-Convolution- InstanceNorm-ReLU layer with k filters and stride 1/2
        """

        x = Conv2DTranspose(filters = k, kernel_size = 3, strides = 2, padding = 'same')(x)
        x = self.normalization(axis = 3, center = True, epsilon=1e-5)(x, training = True)
        x = Activation('relu')(x)
        return x 
    
    def build_generator(self):
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
        x = self.c7s1_k(x, self.nchannels) # use tanh for activation instead NEEDS to be CHANGED LATER!!!!
        return Model(inputs = input_img, outputs = x)

#%%
test_model = CycleGAN((256,256, 1))
#%%
test_generator = test_model.build_generator()
#%%
test_generator.summary()