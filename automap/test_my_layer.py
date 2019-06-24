#%%
from keras import backend as K 
from keras.layers import Layer, Input, Dense
from keras.layers.core import Lambda 
from keras.models import Model
import numpy as np 
#%%
def hadamard_product_sum(tensors):
    out1 = tensors[0] * tensors[1]
    out2 = K.sum(out1, axis=-1)
    return [out1, out2]

def hadamard_product_sum_output_shape(input_shapes):
    shape1 = list(input_shapes[0])
    shape2 = list(input_shapes[1])
    assert shape1 == shape2  # else hadamard product isn't possible
    return [tuple(shape1), tuple(shape2[:-1])]
input_1 = Input(shape = (64, 64,))
input_2 = Input(shape = (64, 64,))

x1 = Dense(32)(input_1)
x2 = Dense(32)(input_2)
layer = Lambda(hadamard_product_sum, hadamard_product_sum_output_shape)
x_hadamard, x_sum = layer([x1, x2])
m = Model([input_1, input_2], [x_hadamard, x_sum])
m.summary()
#%%
