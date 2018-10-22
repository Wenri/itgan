
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers
from wn import WeightNorm


class Decoder(Model):
    def __init__(self, input_shape):
        super(Decoder, self).__init__(name='Decoder')
        w = input_shape[0] // (2**3)
        d = input_shape[2]
        self.model = Sequential([
            Dense(w * w * 256),
            BatchNormalization(),
            Activation('relu'),
            Reshape((w, w, 256)),

            Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),

            Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),

            Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),

            Conv2DTranspose(d, kernel_size=(5, 5), strides=(1, 1)),
            BatchNormalization(),
            Activation('tanh'),
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)
