
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers
from wn import WeightNorm
from resnet import resnet_v2
from wrn import build_model as wrn

class Encoder(Model):
    def __init__(self, weight_decay, num_classes):
        super(Encoder, self).__init__(name='Encoder')
        # n = 12
        # depth = n * 9 + 2
        # self.model = resnet_v2((32, 32, 3), depth)
        self.model = wrn((32, 32, 3), dropout=0.2)
        self.metrics_out = GlobalMaxPooling2D()
        self.cls_out = Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        feature_map = self.model(inputs, training=training)
        metric_out = self.metrics_out(feature_map)
        return metric_out, self.cls_out(metric_out)
