
from __future__ import print_function


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow as tf
import numpy as np

from encoder import Encoder
from triplet_loss import batch_all_triplet_loss

def dump_variable(variables):
    for var in variables:
        print(var.name)


def gaussian_noise_layer(input_layer, std=0.15):
    with tf.name_scope('GaussianNoiseLayer'):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise


def loss_metric(c_m, z_m, mask = None):
    with tf.name_scope('L_GT'):
        if mask is not None:
            c_m = tf.boolean_mask(tf.Print(c_m, [tf.argmax(c_m, axis=1), mask], "SemiLabel"), mask)
            z_m = tf.boolean_mask(z_m, mask)
        n_labels = tf.count_nonzero(tf.reduce_sum(c_m, axis=0) > 0.1)
        two_labels = tf.count_nonzero(tf.reduce_sum(c_m, axis=0) > 1.1)
        cond = tf.logical_and(two_labels>0, n_labels>1)

        def triplet_loss():
            loss, frac = batch_all_triplet_loss(
                labels=tf.argmax(c_m, axis=1), embeddings=z_m, margin=0.2
            )
            return loss

        return tf.cond(cond, triplet_loss, lambda: tf.constant(0, dtype=tf.float32))

class cifar10vgg:
    def __init__(self, train=True):
        self.num_classes = 10
        self.weight_decay = 0.0005

        self.model = self.build_model()

        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar10vgg.h5')

    def normalize(self, X_train, X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self, x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def build_model(self):
        model = Encoder(self.weight_decay, self.num_classes)
        return model

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)

        num_data = len(x)
        test_pred = np.zeros((num_data, self.num_classes))
        for b in range(0, num_data, batch_size):
            bsize = min(batch_size, num_data - b)
            _, test_pred[b:b+bsize] = self.model(tf.convert_to_tensor(x[b:b+bsize]), training=False)
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

        global_step = tf.train.get_or_create_global_step()

        def _adj_learning_rate():
            return learning_rate / tf.to_float(2 ** (global_step * batch_size / num_data // lr_drop))

        opt = tf.train.AdamOptimizer(learning_rate=_adj_learning_rate, beta1=0.9)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        x_test = datagen.standardize(x_test)

        tf.global_variables_initializer()
        tf.local_variables_initializer()

        total_losses = 0.0
        correct = 0
        batches = 0
        batch_samples = 0
        total_samples = 0

        def loss(targets, metric_out, cls_out):
            cls_loss = tf.losses.softmax_cross_entropy(targets, cls_out)
            metric_loss = loss_metric(targets, metric_out)
            reg_term = tf.losses.get_regularization_loss(model.name)
            return cls_loss + metric_loss + reg_term

        def grad(inputs, targets):
            with tf.GradientTape() as tape:
                metric_out, cls_out = model(inputs, training=True)
                loss_value = loss(targets, metric_out, cls_out)
                return loss_value, cls_out, tape.gradient(loss_value, model.trainable_variables)

        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
            batches = batches + 1
            batch_samples += len(x_batch)

            train_losses, train_pred, grads = grad(tf.convert_to_tensor(x_batch), tf.convert_to_tensor(y_batch))
            trainer = opt.apply_gradients(zip(grads, model.trainable_variables), global_step=global_step)

            correct += np.count_nonzero(np.argmax(train_pred, axis=1) == np.argmax(y_batch, axis=1))
            total_losses += train_losses
            if batch_samples >= num_data:
                total_samples += batch_samples
                test_pred = np.argmax(self.predict(x_test, False), axis=1)
                test_gt = np.argmax(y_test, axis=1)
                accuracy = np.mean(test_pred == test_gt)
                print("epoch = %.2f, loss = %f, train_accuracy = %.4f, test_accuracy = %.4f" % 
                    (total_samples / num_data, total_losses * batch_size / batch_samples, correct / batch_samples, accuracy))
                total_losses = 0.0
                correct = 0
                batch_samples = 0

        return model

if __name__ == '__main__':


    tf.enable_eager_execution()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = cifar10vgg()

    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1) != np.argmax(y_test,1)

    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)



