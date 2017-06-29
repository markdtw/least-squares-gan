from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import glob

import scipy.misc
import numpy as np
import tensorflow as tf

from tqdm import tqdm

def preprocessLSUN(filename_list, crop_size):
    os.mkdir('LSUN/conference_room_train_images_center_cropped')

    for index, i in enumerate(tqdm(filename_list)):
        image = scipy.misc.imread(i)
        # do central crop
        x = image.shape[1] // 2 - crop_size // 2
        y = image.shape[0] // 2 - crop_size // 2
        crop_image = image[y:y+crop_size, x:x+crop_size]
        scipy.misc.imsave('LSUN/conference_room_train_images_center_cropped/'+str(index)+'.jpg', crop_image)


class Queue_loader():

    def __init__(self, dataset, batch_size, new_size=112, num_threads=2):
        if dataset == 'LSUN':
            if not os.path.exists('LSUN/conference_room_train_images_center_cropped'):
                filename_list = sorted(glob.glob('LSUN/conference_room_train_images/*'))
                preprocessLSUN(filename_list, 224)
            else:
                filename_list = sorted(glob.glob('LSUN/conference_room_train_images_center_cropped/*'))
            img_shape = [224, 224, 3]
        elif dataset == 'CelebA':
            filename_list = sorted(glob.glob('CelebA/splits/train/*'))
            img_shape = [218, 178, 3]
        else:
            raise SystemExit('dataset not compatible, exiting...')

        queue = self.readFromFile(filename_list, batch_size, img_shape)

        if dataset == 'LSUN':
            queue = tf.image.resize_images(queue, [new_size, new_size])
        else:
            queue = tf.image.crop_to_bounding_box(queue, (img_shape[0]-178)//2, (img_shape[1]-178)//2, 178, 178)
            queue = tf.image.resize_images(queue, [new_size, new_size])

        self.images = tf.cast(queue, tf.float32) / 127.5 - 1. # [-1,. 1.]
        self.iters = len(filename_list) // batch_size

        self.z_ph = tf.placeholder(tf.float32, [None, 1024])

    def readFromFile(self, filename_list, batch_size, img_shape, num_threads=4, min_after_dequeue=10000):

        filename_queue = tf.train.string_input_producer(filename_list, shuffle=False)
        reader = tf.WholeFileReader()
        _, serialized_example = reader.read(filename_queue)

        image = tf.image.decode_jpeg(serialized_example, channels=3)
        image.set_shape(img_shape)
        
        images = tf.train.shuffle_batch(
            [image], batch_size=batch_size, num_threads=num_threads,
            capacity=min_after_dequeue + (num_threads + 1) * batch_size,
            min_after_dequeue=min_after_dequeue,
        )
        
        return images
