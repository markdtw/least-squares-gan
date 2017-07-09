import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

class LSGAN():
    # This is actually DCGAN with a slight modification of the generator
    def __init__(self):
        pass

    def generator(self, z):
        with tf.variable_scope('gen'):
            with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose],
                    normalizer_fn=slim.batch_norm, weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                net = slim.fully_connected(z, 7 * 7 * 256, activation_fn=None)
                net = tf.reshape(net, [-1, 7, 7, 256])
                net = slim.conv2d_transpose(net, 256, 3, stride=2)
                net = slim.conv2d_transpose(net, 256, 3, stride=1)
                net = slim.conv2d_transpose(net, 256, 3, stride=2)
                net = slim.conv2d_transpose(net, 256, 3, stride=1)
                net = slim.conv2d_transpose(net, 128, 3, stride=2)
                net = slim.conv2d_transpose(net,  64, 3, stride=2)
                net = slim.conv2d_transpose(net,   3, 3, stride=1, activation_fn=tf.nn.tanh, normalizer_fn=None)
        return net

    def discriminator(self, images, reuse=False):
        with tf.variable_scope('dis', reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=None,
                    normalizer_fn=slim.batch_norm, weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                net = lrelu(slim.conv2d(images, 64, 5, stride=2, normalizer_fn=None))
                net = lrelu(slim.conv2d(net, 128, 5, stride=2))
                net = lrelu(slim.conv2d(net, 256, 5, stride=2))
                net = lrelu(slim.conv2d(net, 512, 5, stride=2)) # shape = (batch, 7, 7, 512)
                net = tf.reshape(net, [-1, (112 / 2**4)**2 * 512])
                net = slim.fully_connected(net, 1, normalizer_fn=None)
        return net
