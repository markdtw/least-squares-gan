from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import argparse

import scipy.misc
import numpy as np
import tensorflow as tf

from model import LSGAN
from utils import Queue_loader

def train(args):
    loader = Queue_loader(args.dataset, args.bsize)
    model = LSGAN()

    # first, generate fake images through G
    generated_op = model.generator(loader.z_ph)
    # second, get real and fake output from D
    d_fake_output_op = model.discriminator(generated_op)
    d_real_output_op = model.discriminator(loader.images, reuse=True)
    # third, the least-square loss
    d_loss_op = 0.5 * tf.reduce_mean(tf.square(d_real_output_op - 1)) + 0.5 * tf.reduce_mean(tf.square(d_fake_output_op))
    g_loss_op = 0.5 * tf.reduce_mean(tf.square(d_fake_output_op - 1))
    
    # fourth, get the trainable variables for each network
    vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dis')
    vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen')
    
    # now we shall define train ops
    d_train_op = tf.train.AdamOptimizer(args.lr, 0.5).minimize(d_loss_op, var_list=vars_d)
    g_train_op = tf.train.AdamOptimizer(args.lr, 0.5).minimize(g_loss_op, var_list=vars_g)

    d_msum_op = tf.summary.merge([tf.summary.scalar('d real output', tf.reduce_mean(d_real_output_op)),
                                  tf.summary.scalar('d loss', d_loss_op)])
    g_msum_op = tf.summary.merge([tf.summary.scalar('d fake output', tf.reduce_mean(d_fake_output_op)),
                                  tf.summary.scalar('g loss', g_loss_op)])
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    writer = tf.summary.FileWriter('log', sess.graph)
    sess.run(tf.global_variables_initializer())

    if args.modelpath is not None:
        print ('From model: {}'.format(args.modelpath))
        saver.restore(sess, args.modelpath)

    print ('Start training')
    print ('batch size: %d, ep: %d, iter: %d, initial lr: %.4f' % (args.bsize, args.ep, loader.iters, args.lr))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ep = 1
    step = 1
    while True:
        feed_dict = {loader.z_ph: np.random.uniform(-1., 1., (args.bsize, 1024)).astype(np.float32)}
        # update d network
        d_fake_output, d_real_output, d_loss, _, d_msum, images = sess.run([
            tf.reduce_mean(d_fake_output_op), tf.reduce_mean(d_real_output_op),
            d_loss_op, d_train_op, d_msum_op, loader.images], feed_dict=feed_dict)
        
        writer.add_summary(d_msum, (ep-1) * loader.iters + step)
        # update g network
        generated, g_loss, _, g_msum = sess.run([
            generated_op, g_loss_op, g_train_op, g_msum_op], feed_dict=feed_dict)

        writer.add_summary(g_msum, (ep-1) * loader.iters + step)
        if step % 40 == 0:
            print ('epoch: %2d, step: %3d, d_fake: %.3f, d_real: %.3f, d_loss: %.3f, g_loss: %.3f' %
                    (ep, step, d_fake_output, d_real_output, d_loss, g_loss))

        if step % loader.iters == 0:
            print ('epoch: %2d, step: %3d, d_fake: %.3f, d_real: %.3f, d_loss: %.3f, g_loss: %.3f, epoch %2d done' %
                    (ep, step, d_fake_output, d_real_output, d_loss, g_loss, ep))
            
            feed_dict = {loader.z_ph: np.random.uniform(-1., 1., (40, 1024)).astype(np.float32)}
            generated = sess.run(generated_op, feed_dict=feed_dict)
            assert generated.shape == (40, 112, 112, 3)
            generated = (generated + 1) * 127.5 # scale from [-1., 1.] to [0., 255.]
            generated = np.clip(generated, 0., 255.).astype(np.uint8)

            background = np.ones((10 + (112 + 10) * 5, 10 + (112 + 10) * 8, 3)).astype(np.uint8) * 255
            for i in xrange(5):
                for j in xrange(8):
                    background[i*122+10: i*122+10 + 112, j*122+10: j*122+10 + 112, :] = generated[i*8 + j]
                
            scipy.misc.imsave(os.path.join('log', 'generated-ep-' + str(ep) + '.jpg'), background)

            checkpoint_path = os.path.join('ckpt', 'lsgan-' + args.dataset)
            saver.save(sess, checkpoint_path, global_step=ep)
            ep += 1
            step = 1
        else:
            step += 1

        if ep == args.ep:
            print ('\nDone training, epoch limit: %d reached.' % (args.ep))
            break
    
    coord.request_stop()
    coord.join(threads)
    sess.close()
    print ('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='set this to train.')
    parser.add_argument('--lr', metavar='', type=float, default=2e-4, help='learning rate.')
    parser.add_argument('--ep', metavar='', type=int, default=200, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int, default=128, help='batch size.')
    parser.add_argument('--modelpath', metavar='', type=str, default=None, help='trained tensorflow model path.')
    parser.add_argument('--dataset', metavar='', type=str, default='LSUN', help='LSUN or CelebA, default LSUN.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))
    if args.train:
        train(args)
    if not args.train:
        parser.print_help()
