import tensorflow as tf
from PIL import Image
import numpy as np

def image_cast(img):
    return tf.cast(img * 127.5 + 127.5, tf.uint8)


def kl_loss(avg, log_var):
    with tf.name_scope('KLLoss'):
        return tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + log_var - tf.square(avg) - tf.exp(log_var), axis=-1))


def lrelu(x, alpha=0.1):
    with tf.name_scope('LeakyReLU'):
        return tf.maximum(x, alpha * x)


def binary_accuracy(y_true, y_pred):
    with tf.name_scope('BinaryAccuracy'):
        return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.round(tf.sigmoid(y_pred))), dtype=tf.float32))


def gaussian_noise_layer(input_layer, std):
    with tf.name_scope('GaussianNoiseLayer'):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise


def sample_normal(avg, log_var):
    with tf.name_scope('SampleNormal'):
        epsilon = tf.random_normal(tf.shape(avg))
        return tf.add(avg, tf.multiply(tf.exp(0.5 * log_var), epsilon))


def vgg_conv_unit(x, filters, layers, training=True):
    # Convolution
    for i in range(layers):
        x = tf.layers.conv2d(x, filters, (3, 3), (1, 1), 'same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=training)
        x = lrelu(x)

    # Downsample
    x = tf.layers.conv2d(x, filters, (2, 2), (2, 2), 'same',
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=training)
    x = lrelu(x)

    return x


def vgg_deconv_unit(x, filters, layers, training=True):
    # Upsample
    x = tf.layers.conv2d_transpose(x, filters, (2, 2), (2, 2), 'same',
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=training)
    x = lrelu(x)

    # Convolution
    for i in range(layers):
        x = tf.layers.conv2d(x, filters, (3, 3), (1, 1), 'same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=training)
        x = lrelu(x)

    return x


def time_format(t):
    m, s = divmod(t, 60)
    m = int(m)
    s = int(s)
    if m == 0:
        return '%d sec' % s
    else:
        return '%d min %d sec' % (m, s)


def dump_variable(variables):
    for var in variables:
        print(var.name)


def save_images(imgs, filename):
        """
        Save images generated from random sample numbers
        """
        imgs = np.clip(imgs, 0.0, 1.0)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))

        _, height, width, dims = imgs.shape

        margin = min(width, height) // 10
        figure = np.ones(((margin + height) * 10 + margin, (margin + width) * 10 + margin, dims), np.float32)

        for i in range(100):
            row = i // 10
            col = i % 10

            y = margin + (margin + height) * row
            x = margin + (margin + width) * col
            figure[y:y+height, x:x+width, :] = imgs[i, :, :, :]

        figure = Image.fromarray((figure * 255.0).astype(np.uint8))
        figure.save(filename)
