
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers
from wn import WeightNorm


class Encoder(Model):
    def __init__(self, weight_decay, num_classes):
        super(Encoder, self).__init__(name='Encoder')
        self.model = Sequential([
            WeightNorm(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))),
            Activation('relu'),
            BatchNormalization(),
            WeightNorm(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))),
            Activation('relu'),
            BatchNormalization(),
            WeightNorm(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))),
            Activation('relu'),
            BatchNormalization(),
            Dropout(0.5),

            MaxPooling2D(pool_size=(2, 2)),

            WeightNorm(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))),
            Activation('relu'),
            BatchNormalization(),
            WeightNorm(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))),
            Activation('relu'),
            BatchNormalization(),
            WeightNorm(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))),
            Activation('relu'),
            BatchNormalization(),
            Dropout(0.5),

            MaxPooling2D(pool_size=(2, 2)),

            WeightNorm(Conv2D(512, (3, 3), padding='valid',kernel_regularizer=regularizers.l2(weight_decay))),
            Activation('relu'),
            BatchNormalization(),
            WeightNorm(Conv2D(256, (1, 1), padding='same',kernel_regularizer=regularizers.l2(weight_decay))),
            Activation('relu'),
            BatchNormalization(),
            WeightNorm(Conv2D(128, (1, 1), padding='same',kernel_regularizer=regularizers.l2(weight_decay))),
            Activation('relu'),
            BatchNormalization(),

            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
        ])
        self.metric_out = WeightNorm(Dense(128))
        self.cls_out = WeightNorm(Dense(num_classes))

    def call(self, inputs, training=None, mask=None):
        feature = self.model(inputs, training=training)
        metric_out = self.metric_out(feature)
        return metric_out, self.cls_out(metric_out)
