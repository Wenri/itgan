
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Reshape, LeakyReLU, PReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Add
from subpixel import SubpixelConv2D
from utils import sample_normal
from tensorflow.keras.regularizers import l2
from wn import WeightNorm


# build a residual block
def res_block(inputs, filters):
    x = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same', activation=None, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None)(x)
    x = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same', activation=None, use_bias=False)(x)
    x = BatchNormalization()(x)
    return Add()([x, inputs])


# build an upscale block
# PixelShuffler is replaced by an UpSampling2D layer (nearest upsampling)
def up_block(x):
    x = Conv2D(256, kernel_size=(3,3), strides=(1,1) , padding='same', activation=None, use_bias=False)(x)
    x = SubpixelConv2D(input_shape=x.get_shape(), scale=2)(x)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None)(x)
    return x


def build_model(input_shape):
    # This returns a tensor input to the model

    w = input_shape[0] // (2 ** 3)
    d = input_shape[2]

    inputs = Input(shape=(256,))

    x = Dense(w * w * 256)(inputs)
    x = BatchNormalization()(x)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None)(x)
    x = Reshape((w, w, 256))(x)

    x = up_block(x)
    
    for i in range(16):
        x = res_block(x, 64)

    x = up_block(x)
    x = up_block(x)

    # final conv layer : activated with tanh -> pixels in [-1, 1]
    outputs = Conv2D(3, kernel_size=(9, 9), strides=(1, 1), activation='tanh', use_bias=False, padding='same')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


class Decoder(Model):
    def __init__(self, input_shape):
        super(Decoder, self).__init__(name='Decoder')
        self.model = build_model(input_shape=input_shape)

    def call(self, inputs, training=None, mask=None):
        avg, log_var = inputs
        z = sample_normal(avg, log_var)
        return self.model(z, training=training)
