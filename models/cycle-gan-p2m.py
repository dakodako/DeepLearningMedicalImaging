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
from keras.optimizers import Adam
#%%
from skimage.io import imread
from skimage.transform import rotate, resize
#%%
import numpy as np
from glob import glob 
import nibabel as nib 
import datetime
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
        #path = glob('/Users/didichi/.keras/datasets/%s/%s/*' % (self.dataset_name, data_type))
        path = glob('/Users/chid/.keras/datasets/p2m/train/*')
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
            img_A = resize(img_A, (self.img_res[0],self.img_res[1]))
            img_B = resize(img_B, (self.img_res[0],self.img_res[1]))
            if not is_testing and np.random.random() <0.5 and is_jitter:
                # 1. Resize an image to bigger height and width
                img_A = resize(img_A, (img_A.shape[0] + 64, img_A.shape[1] + 64))
                img_B = resize(img_B, (img_B.shape[0] + 64, img_B.shape[1] + 64))
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
        path = glob('/Users/chid/.keras/datasets/%s/%s/*' % (self.dataset_name, data_type))
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
                img_A = resize(img_A, (self.img_res[0],self.img_res[1]))
                img_B = resize(img_B, (self.img_res[0],self.img_res[1]))
                #print(img_A.shape)
                #print(img_B.shape)
                if not is_testing and np.random.random() <0.5 and is_jitter:
                    # 1. Resize an image to bigger height and width
                    img_A = resize(img_A, (img_A.shape[0] + 64, img_A.shape[1] + 64))
                    img_B = resize(img_B, (img_B.shape[0] + 64, img_B.shape[1] + 64))
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
    def __init__(self, lr_D = 2e-4, lr_G =2e-4, img_shape = (256,256,1), dataset_name = 'p2m'):
        self.normalization = InstanceNormalization
        self.shape = img_shape
        self.dataset_name = dataset_name
        self.image_rows = img_shape[0]
        self.nchannels = self.shape[-1]
        self.use_patchgan = True
        self.use_supervised_learning = False
        self.use_linear_decay = False # need to be implemented
        self.decay_epoch = 101
        self.learning_rate_D = lr_D
        self.learning_rate_G = lr_G 
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = 1
        self.epochs = 1
        self.lambda_1 = 10.0 # cyclic loss weight A_2_B
        self.lambda_2 = 10.0 # cyclic loss weight B_2_A
        self.lambda_D = 1.0
        self.supervised_weight = 10.0
        patch = int(self.image_rows / 2**5)
        self.disc_patch = (patch, patch, 1)
        self.data_loader = DataLoader(dataset_name = self.dataset_name, img_res=self.shape)
        # optimizer
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)
        # Build discriminator
        D_A = self.build_discriminator()
        D_B = self.build_discriminator()
        loss_weights_D = [0.5]

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
        if self.use_supervised_learning:
            model_outputs.append(fake_A)
            model_outputs.append(fake_B)
            compile_loss.append('MAE')
            compile_loss.append('MAE')
            compile_weights.append(self.supervised_weight)
            compile_weights.append(self.supervised_weight)
        print(real_A.shape)
        self.G_model = Model(inputs = [real_A, real_B], outputs = model_outputs, name = 'G_model')
        self.G_model.compile(optimizer = self.opt_G, loss = compile_loss, loss_weights=compile_weights)
        self.G_model.summary()
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
    def build_discriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.shape)
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
    def build_discriminator2(self, name = None):
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
    #def get_lr_linear_decay_rate(self):
        #updates_per_epoch_D = 2
    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) -decay
        if new_lr < 0:
            new_lr = 0
        K.set_value(model.optimizer.lr, new_lr)
    def train(self, epochs, batch_size = 1, sample_interval = 50):
        start_time = datetime.datetime.now()
        DA_losses = []
        DB_losses = []
        gA_d_losses_fake = []
        gB_d_losses_fake = []
        gA_losses_reconstructed = []
        gB_losses_reconstructed = []
        GA_losses = []
        GB_losses = []
        reconstruction_losses = []
        D_losses = []
        G_losses = []
        #valid = np.ones((batch_size, ) + self.disc_patch)
        #fake = np.zeros((batch_size, ) + self.disc_patch)
        label_shape = (batch_size, ) + self.D_A.output_shape[1:]
        valid = np.ones(shape = label_shape)* 1.0
        fake = valid * 0 
        for epoch in range(1,epochs+1):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                # ====== discriminators training ====== #
                print(imgs_A.shape)
                fake_B = self.G_A2B.predict(imgs_A)
                fake_A = self.G_B2A.predict(imgs_B)
                print(imgs_A.shape)
                print(valid.shape)
                dA_loss_real = self.D_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.D_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_fake, dA_loss_real)

                dB_loss_real = self.D_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.D_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5*np.add(dA_loss_fake, dB_loss_real)

                d_loss = 0.5*np.add(dA_loss, dB_loss)

                # ====== generator training ====== #

                target_data = [imgs_A, imgs_B]
                target_data.append(valid)
                target_data.append(valid)

                if self.use_supervised_learning:
                    target_data.append(imgs_A)
                    target_data.append(imgs_B)
                G_loss = self.G_model.train_on_batch(x = [imgs_A, imgs_B], y = target_data)
                
                gA_d_loss_fake = G_loss[1]
                gB_d_loss_fake = G_loss[2]
                reconstruction_loss_A = G_loss[3]
                reconstruction_loss_B = G_loss[4]
                # Update learning rates
                #if self.use_linear_decay and epoch > self.decay_epoch:
                    #self.update_lr(self.D_A, )
                DA_losses.append(dA_loss)
                DB_losses.append(dB_loss)
                gA_d_losses_fake.append(gA_d_loss_fake)
                gB_d_losses_fake.append(gB_d_loss_fake)
                gA_losses_reconstructed.append(reconstruction_loss_A)
                gB_losses_reconstructed.append(reconstruction_loss_B)
                GA_loss = gA_d_loss_fake + reconstruction_loss_A
                GB_loss = gB_d_loss_fake + reconstruction_loss_B
                D_losses.append(d_loss)
                GA_losses.append(GA_loss)
                GB_losses.append(GB_loss)
                G_losses.append(G_loss)
                reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
                reconstruction_losses.append(reconstruction_loss)

                print('\n')
                print('Epoch---------------', epoch, '/', epochs)
                print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
                print('D_loss: ', d_loss)
                print('G_loss: ', G_loss[0])
                print('reconstruction_loss: ', reconstruction_loss)
                print('DA_loss:', DA_loss)
                print('DB_loss:', DB_loss)
#%%
test_model = CycleGAN((256,256, 1))
#%%
test_generator = test_model.build_generator()
#%%
test_generator.summary()
#%%
test_discriminator = test_model.build_discriminator()
#%%
test_discriminator.summary()
#%%
test_model.G_model.summary()
#%%
A, B = test_model.data_loader.load_data(1)
#%%
print(A.shape)
print(np.mean(A))
#%%
test_model.train(1)
#%%
ones = np.ones(shape = (1,16,16,1))
print(ones.shape)
#%%
test_model.D_A.train_on_batch(A, ones)
#%%
print(test_model.D_A.output_shape)
#%%
test_model.D_A.summary()
#%%
test_discriminator = test_model.build_discriminator()
test_discriminator.summary()
#%%
test_model.D_B.train_on_batch(x = B, y = ones* 0)