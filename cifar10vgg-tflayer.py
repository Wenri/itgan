
from __future__ import print_function

from tensorflow.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K
from tensorflow import keras
from wn import WeightNorm
import tensorflow as tf
import numpy as np

def dump_variable(variables):
    for var in variables:
        print(var.name)

def gaussian_noise_layer(input_layer, std=0.15):
    with tf.name_scope('GaussianNoiseLayer'):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

class Encoder(object):
    def __init__(self, input_shape, z_dims, metric_dims, num_attrs, weight_decay):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.z_dims = z_dims
        self.metric_dims = metric_dims
        self.num_attrs = num_attrs
        self.weight_decay = weight_decay
        self.name = 'encoder'
        self.model = self.kerasmodel()

    def _conv(self, inputs, filters, name = None, w = 5, s = 1, training=True, padding='same'):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(inputs, filters, (w, w), (s, s), padding,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            # x = tf.layers.batch_normalization(x, training=training)
            x = tf.nn.leaky_relu(x, 0.1)
        return x

    def _convwn(self, inputs, filters, name = None, w = 5, s = 1, training=True, padding='same'):
        with tf.variable_scope(name):
            x = WeightNorm(tf.layers.Conv2D(filters, (w, w), (s, s), padding, use_bias=False), training=training)(inputs)
            # x = tf.layers.batch_normalization(x, training=training)
            x = tf.nn.leaky_relu(x, 0.1)
        return x
    
    def kerasmodel(self):
        weight_decay = self.weight_decay
        lrelu = lambda x: tf.nn.leaky_relu(x, 0.1)

        with tf.variable_scope(self.name, reuse=self.reuse):
            model = (
#                gaussian_noise_layer,

                Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)),
                lrelu,
                BatchNormalization(),
                Dropout(0.3),
                WeightNorm(Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))),
                lrelu,
                MaxPooling2D(pool_size=(2, 2), strides=2),

                WeightNorm(Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))),
                lrelu,
                Dropout(0.4),
                WeightNorm(Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))),
                lrelu,
                MaxPooling2D(pool_size=(2, 2), strides=2),

                WeightNorm(Conv2D(256, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))),
                lrelu,
                Dropout(0.4),
                WeightNorm(Conv2D(256, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))),
                lrelu,
                Dropout(0.4),
                WeightNorm(Conv2D(256, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))),
                lrelu,
                MaxPooling2D(pool_size=(2, 2), strides=2),

                WeightNorm(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))),
                lrelu,
                Dropout(0.4),
                WeightNorm(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))),
                lrelu,
                Dropout(0.4),
                WeightNorm(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))),
                lrelu,
                MaxPooling2D(pool_size=(2, 2), strides=2),

                WeightNorm(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))),
                lrelu,
                Dropout(0.4),
                WeightNorm(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))),
                lrelu,
                Dropout(0.4),
                WeightNorm(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))),
                lrelu,
                MaxPooling2D(pool_size=(2, 2), strides=2),
                Dropout(0.5),

                Flatten(),

                Dense(512, kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)),
                lrelu,
                BatchNormalization(),
                Dropout(0.5),
                Dense(self.num_attrs)
            )

        return model

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            x = inputs
            for l in self.model:
                try:
                    x = l(x, training=training)
                except:
                    x = l(x)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
        self.reuse = True

        return x

class cifar10vgg:
    def __init__(self,train=True):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = (32,32,3)

        self.model = self.build_model()
        dump_variable(self.model.variables)

        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar10vgg.h5')

    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def build_model(self):
        self.x_r = tf.placeholder(tf.float32, shape=(None,) + self.x_shape)
        self.c_r = tf.placeholder(tf.float32, shape=(None, self.num_classes))
        model = Encoder(self.x_shape, 128, 256, self.num_classes, self.weight_decay)
        self.y_pred = model(self.x_r)

        self.test_input = tf.placeholder(tf.float32, shape=(None,) + self.x_shape)
        self.c_test_pred = model(self.test_input, training=False)

        return model

    def predict(self,x,sess,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)

        num_data = len(x)
        test_pred = np.zeros((num_data, self.num_classes))
        for b in range(0, num_data, batch_size):
            bsize = min(batch_size, num_data - b)
            test_pred[b:b+bsize] = sess.run(
                    self.c_test_pred,
                    feed_dict={
                        self.test_input: x[b:b+bsize]
                    }
            )
        return test_pred

    def train(self, model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.001
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        num_data=len(x_train)

        current_epoch = tf.Variable(0, name='current_epoch', dtype=tf.int32)
        current_batch = tf.Variable(0, name='current_batch', dtype=tf.int32)

        update_epoch = current_epoch.assign(current_epoch + 1)
        update_batch = current_batch.assign(tf.mod(tf.minimum(current_batch + batch_size, num_data), num_data))

        perm = np.random.permutation(num_data)

        learning_rate = learning_rate / tf.to_float(2 ** (current_epoch // lr_drop))

        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

        with tf.name_scope('L_CPC'):
            L_CPC = tf.losses.softmax_cross_entropy(self.c_r, self.y_pred) #, weights=c_weights)
        reg_term = tf.losses.get_regularization_loss(model.name)
        losses = L_CPC + reg_term

        with tf.control_dependencies(model.update_ops):
            trainer = opt.minimize(losses, var_list=model.variables)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=True,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        x_test_std = datagen.standardize(x_test)

        sess = K.get_session()
        with sess.as_default():

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            total_losses = 0.0
            correct = 0
            batches = 0
            batch_samples = 0
            total_samples = 0

            for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
                batches = batches + 1
                batch_samples += len(x_batch)
                _, train_losses, train_pred = sess.run(
                        (trainer, losses, self.y_pred),
                        feed_dict={
                            self.x_r: x_batch, self.c_r: y_batch
                        }
                )
                correct += np.count_nonzero(np.argmax(train_pred, axis=1) == np.argmax(y_batch, axis=1))
                total_losses += train_losses
                if(batch_samples >= num_data):
                    total_samples += batch_samples
                    test_pred = np.argmax(self.predict(x_test_std, sess, False), axis=1)
                    test_gt = np.argmax(y_test, axis=1)
                    accuracy = np.mean(test_pred == test_gt)
                    print("epoch = %.2f, loss = %f, train_accuracy = %.4f, test_accuracy = %.4f" % 
                        (total_samples / num_data, total_losses * batch_size / batch_samples, correct / batch_samples, accuracy))
                    total_losses = 0.0
                    correct = 0
                    batch_samples = 0

        return model

if __name__ == '__main__':


    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = cifar10vgg()

    sess = K.get_session()
    predicted_x = model.predict(x_test, sess)
    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)



