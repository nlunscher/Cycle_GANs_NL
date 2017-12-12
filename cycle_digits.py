
import tensorflow as tf
import numpy as np
import cv2

import random
import datetime
import sys, os

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial)

def conv2d(x, W, stride = 1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def conv2d_transpose(x, W, out_shape, stride = 1):
    return tf.nn.conv2d_transpose(x, W, strides=[1, stride, stride, 1],
                                    padding='SAME', output_shape=out_shape)

def make_directory(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

class cycle_digit():

    def __init__(self, sess):
        self.sess = sess
        self.data_folder = 'Data_digits/fast_data/'
        self.mnist_data = self.data_folder + 'MNIST_train'
        self.svhn_data = self.data_folder + "SVHN_train"
        self.im_per_digit = 5000
        self.batch_size = 1

        self.learning_rate = 1e-4 * 2

        self.save_folder = 'cd_saves'
        make_directory(self.save_folder)


    def load_pair(self):
        digit = 8 #random.choice([0,1,2,3,4,5,6,7,8,9])
        m_im_num = random.randint(0, self.im_per_digit)
        s_im_num = random.randint(0, self.im_per_digit)

        mnist = cv2.imread(self.mnist_data + '/' + str(digit) + '/' + str(m_im_num).zfill(7) + '.png')
        svhn = cv2.imread(self.svhn_data + '/' + str(digit) + '/' + str(s_im_num).zfill(7) + '.png')

        mnist = cv2.resize(mnist, (32,32), interpolation = cv2.INTER_LINEAR)

        mnist = np.array(mnist, dtype=np.float32) / 255.
        svhn = np.array(svhn, dtype=np.float32) / 255.

        return mnist, svhn

    def load_batch(self):
        mnists = []
        svhns = []

        for i in range(self.batch_size):
            m, s = self.load_pair()
            mnists.append(m)
            svhns.append(s)

        return np.asarray(mnists), np.asarray(svhns)

    def create_net(self):
        self.m_im = tf.placeholder(tf.float32, name='mnist_image',shape=[self.batch_size] + [32,32,3])
        self.s_im = tf.placeholder(tf.float32, name='svhn_image',shape= [self.batch_size] + [32,32,3])

        self.is_train = tf.placeholder(tf.bool, name = 'is_train')

        m2s_W_conv1 = weight_variable([7, 7, 3, 32])
        m2s_W_conv2 = weight_variable([5, 5, 32, 64])
        
        m2s_W_conv3_1 = weight_variable([3, 3, 64, 64])
        m2s_W_conv3_2 = weight_variable([3, 3, 64, 64])

        m2s_W_conv4_1 = weight_variable([3, 3, 64, 64])
        m2s_W_conv4_2 = weight_variable([3, 3, 64, 64])
        
        m2s_W_dconv1 = weight_variable([5, 5, 32, 64])
        m2s_b_dconv1 = bias_variable([32])
        m2s_W_dconv2 = weight_variable([7, 7, 3, 32])
        m2s_b_dconv2 = bias_variable([3])

        self.m2s_vars = [m2s_W_conv1, m2s_W_conv2, 
                        m2s_W_conv3_1, m2s_W_conv3_2, m2s_W_conv4_1, m2s_W_conv4_2, 
                        m2s_W_dconv1, m2s_b_dconv1, m2s_W_dconv2, m2s_b_dconv2]

        def m2s(m_image, is_train):
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(m_image, m2s_W_conv1, 2), training=is_train), name="m2s_conv1") #16
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, m2s_W_conv2, 2), training=is_train), name="m2s_conv2") #8

            h_c3_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, m2s_W_conv3_1, 1), training=is_train), name="m2s_h_c3_1")
            h_c3_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c3_1, m2s_W_conv3_2, 1), training=is_train) + h_conv2, name="m2s_h_c3_2")

            h_c4_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c3_2, m2s_W_conv4_1, 1), training=is_train), name="m2s_h_c4_1")
            h_c4_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c4_1, m2s_W_conv4_2, 1), training=is_train) + h_c3_2, name="m2s_h_c4_2")

            osize_dconv1 = h_conv1.get_shape().as_list()
            osize_dconv1[0] = self.batch_size
            h_dconv1 = tf.nn.relu(conv2d_transpose(h_c4_2, m2s_W_dconv1, osize_dconv1,2) + m2s_b_dconv1, name="m2s_dconv1") #16
            osize_dconv2 = m_image.get_shape().as_list()
            osize_dconv2[0] = self.batch_size
            h_dconv2 = tf.nn.sigmoid(conv2d_transpose(h_dconv1, m2s_W_dconv2, osize_dconv2,2) + m2s_b_dconv2, name="m2s_dconv2") #32

            return h_dconv2

        sD_W_conv1 = weight_variable([7, 7, 3, 32])
        sD_W_conv2 = weight_variable([5, 5, 32, 64])
        sD_W_conv3 = weight_variable([3, 3, 64, 64])
        sD_W_f1 = weight_variable([1024, 256])
        sD_b_f1 = bias_variable([256])
        sD_W_f2 = weight_variable([256, 1])
        sD_b_f2 = bias_variable([1])

        self.sD_vars = [sD_W_conv1, sD_W_conv2, sD_W_conv3, sD_W_f1, sD_b_f1, sD_W_f2, sD_b_f2]

        def sD(s_image, is_train):
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(s_image, sD_W_conv1, 2), training=is_train), name="sD_conv1") #16
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, sD_W_conv2, 2), training=is_train), name="sD_conv2") #8
            h_conv3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, sD_W_conv3, 2), training=is_train), name="sD_conv3") #4
            h3_flat = tf.reshape(h_conv3, [-1, 4*4*64])
            h4 = tf.nn.sigmoid(tf.matmul(h3_flat, sD_W_f1) + sD_b_f1, name="sD_f1")
            h5 = tf.matmul(h4, sD_W_f2) + sD_b_f2
            h5_sig = tf.nn.sigmoid(h5, name="sD_f2")

            return h5, h5_sig

        s2m_W_conv1 = weight_variable([7, 7, 3, 32])
        s2m_W_conv2 = weight_variable([5, 5, 32, 64])

        s2m_W_conv3_1 = weight_variable([3, 3, 64, 64])
        s2m_W_conv3_2 = weight_variable([3, 3, 64, 64])

        s2m_W_conv4_1 = weight_variable([3, 3, 64, 64])
        s2m_W_conv4_2 = weight_variable([3, 3, 64, 64])

        s2m_W_dconv1 = weight_variable([5, 5, 32, 64])
        s2m_b_dconv1 = bias_variable([32])
        s2m_W_dconv2 = weight_variable([7, 7, 3, 32])
        s2m_b_dconv2 = bias_variable([3])

        self.s2m_vars = [s2m_W_conv1, s2m_W_conv2,
                        s2m_W_conv3_1, s2m_W_conv3_2, s2m_W_conv4_1, s2m_W_conv4_2,
                        s2m_W_dconv1, s2m_b_dconv1, s2m_W_dconv2, s2m_b_dconv2]

        def s2m(s_image, is_train):
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(s_image, s2m_W_conv1, 2), training=is_train), name="s2m_conv1") #16
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, s2m_W_conv2, 2), training=is_train), name="s2m_conv2") #8

            h_c3_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, s2m_W_conv3_1, 1), training=is_train), name="s2m_h_c3_1")
            h_c3_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c3_1, s2m_W_conv3_2, 1), training=is_train) + h_conv2, name="s2m_h_c3_2")

            h_c4_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c3_2, s2m_W_conv4_1, 1), training=is_train), name="s2m_h_c4_1")
            h_c4_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c4_1, s2m_W_conv4_2, 1), training=is_train) + h_c3_2, name="s2m_h_c4_2")

            osize_dconv1 = h_conv1.get_shape().as_list()
            osize_dconv1[0] = self.batch_size
            h_dconv1 = tf.nn.relu(conv2d_transpose(h_c4_2, s2m_W_dconv1, osize_dconv1,2) + s2m_b_dconv1, name="s2m_dconv1") #16
            osize_dconv2 = s_image.get_shape().as_list()
            osize_dconv2[0] = self.batch_size
            h_dconv2 = tf.nn.sigmoid(conv2d_transpose(h_dconv1, s2m_W_dconv2, osize_dconv2,2) + s2m_b_dconv2, name="s2m_dconv2") #32

            return h_dconv2

        mD_W_conv1 = weight_variable([7, 7, 3, 32])
        mD_W_conv2 = weight_variable([5, 5, 32, 64])
        mD_W_conv3 = weight_variable([3, 3, 64, 64])
        mD_W_f1 = weight_variable([1024, 256])
        mD_b_f1 = bias_variable([256])
        mD_W_f2 = weight_variable([256, 1])
        mD_b_f2 = bias_variable([1])

        self.mD_vars = [mD_W_conv1, mD_W_conv2, mD_W_conv3, mD_W_f1, mD_b_f1, mD_W_f2, mD_b_f2]

        def mD(m_image, is_train):
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(m_image, mD_W_conv1, 2), training=is_train), name="mD_conv1") #16
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, mD_W_conv2, 2), training=is_train), name="mD_conv2") #8
            h_conv3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, mD_W_conv3, 2), training=is_train), name="mD_conv3") #4
            h3_flat = tf.reshape(h_conv3, [-1, 4*4*64])
            h4 = tf.nn.sigmoid(tf.matmul(h3_flat, mD_W_f1) + mD_b_f1, name="mD_f1")
            h5 = tf.matmul(h4, mD_W_f2) + mD_b_f2
            h5_sig = tf.nn.sigmoid(h5, name="mD_f2")

            return h5, h5_sig
        
        # G
        self.m2s_im = m2s(self.m_im, self.is_train)
        self.s2m_im = s2m(self.s_im, self.is_train)

        self.m2s2m_im = s2m(self.m2s_im, self.is_train)
        self.s2m2s_im = m2s(self.s2m_im, self.is_train)

        m_G_fake, self.m_G_fake_sig = mD(self.s2m_im, self.is_train)
        s_G_fake, self.s_G_fake_sig = sD(self.m2s_im, self.is_train)

        self.m2s_Ad = tf.reduce_mean(tf.square(self.s_G_fake_sig - tf.ones_like(self.s_G_fake_sig)))
        self.s2m_Ad = tf.reduce_mean(tf.square(self.m_G_fake_sig - tf.ones_like(self.m_G_fake_sig)))

        self.cycle_loss = 5 * (tf.reduce_mean(tf.reduce_sum(tf.abs(self.m2s2m_im - self.m_im), 3)) +
                                tf.reduce_mean(tf.reduce_sum(tf.abs(self.s2m2s_im - self.s_im), 3)))

        self.G_loss = self.m2s_Ad + self.s2m_Ad + self.cycle_loss

        # D
        # s_D_fake, self.s_D_fake_sig = sD(self.m2s_im, self.is_train)
        s_D_fake, self.s_D_fake_sig = s_G_fake, self.s_G_fake_sig
        s_D_real, self.s_D_real_sig = sD(self.s_im, self.is_train)

        # m_D_fake, self.m_D_fake_sig = mD(self.s2m_im, self.is_train)
        m_D_fake, self.m_D_fake_sig = m_G_fake, self.m_G_fake_sig
        m_D_real, self.m_D_real_sig = mD(self.m_im, self.is_train)

        self.sD_loss_real = tf.reduce_mean(tf.square(self.s_D_real_sig - tf.ones_like(self.s_D_real_sig)))
        self.sD_loss_fake = tf.reduce_mean(tf.square(self.s_D_fake_sig - tf.zeros_like(self.s_D_fake_sig)))
        self.sD_loss = self.sD_loss_real + self.sD_loss_fake

        self.mD_loss_real = tf.reduce_mean(tf.square(self.m_D_real_sig - tf.ones_like(self.m_D_real_sig)))
        self.mD_loss_fake = tf.reduce_mean(tf.square(self.m_D_fake_sig - tf.zeros_like(self.m_D_fake_sig)))
        self.mD_loss = self.mD_loss_real + self.mD_loss_fake

        self.D_loss = (self.sD_loss + self.mD_loss) / 2


    def train(self, iterations):

        cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Images", 256*3, 256*2)

        self.create_net()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            G_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.G_loss, var_list=(self.m2s_vars + self.s2m_vars))
            D_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.D_loss, var_list=(self.sD_vars + self.mD_vars))

        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord = self.coord)

        print 'Training for', iterations
        for i in range(iterations):
            m_reals, s_reals = self.load_batch()

            _, _, m, m2s, m2s2m, s, s2m, s2m2s, \
                        m2sDS, s2mDS, \
                        m2sD, s2mD, cL, \
                        mDS, mDFS, sDS, sDFS, \
                        mD, mFD, sD, sFD = self.sess.run([G_train_step, D_train_step, 
                                                self.m_im, self.m2s_im, self.m2s2m_im, self.s_im, self.s2m_im, self.s2m2s_im,
                                                self.s_G_fake_sig, self.m_G_fake_sig, 
                                                self.m2s_Ad, self.s2m_Ad, self.cycle_loss,
                                                self.m_D_real_sig, self.m_D_fake_sig, self.s_D_real_sig, self.s_D_fake_sig,
                                                self.mD_loss_real, self.mD_loss_fake, self.sD_loss_real, self.sD_loss_fake
                                                ], 
                                                    feed_dict={self.m_im:m_reals, self.s_im:s_reals,
                                                                self.is_train:True})

            if (i % 100 == 0 or i == iterations-1):
                print i, " =======================================================================", datetime.datetime.now() - start_time
                print "        ", "Max m:", np.max(m[0]), "Max m2s:", np.max(m2s), "Max m2s2m:", np.max(m2s2m)
                print "        ", "Max s:", np.max(s[0]), "Max s2m:", np.max(s2m), "Max s2m2s:", np.max(s2m2s)
                print "        ", "m2sDS:", m2sDS[0], "s2mDS:", s2mDS[0]
                print "        ", "m2sD:", m2sD, "s2mD:", s2mD, "cL:", cL
                print "        ", "mDS:", mDS[0], "mDFS:", mDFS[0], "sDS:", sDS[0], "sDFS:", sDFS[0]
                print "        ", "mD:", mD, "mFD:", mFD, "sD:", sD, "sFD:", sFD

                x = np.concatenate([np.concatenate([m[0], m2s[0], m2s2m[0]], axis = 1),
                                    np.concatenate([s[0], s2m[0], s2m2s[0]], axis = 1)], axis = 0)
                cv2.imshow('Images', x)
                cv2.waitKey(50)

                if (i % 1000 == 0 or i == iterations-1):
                    cv2.imwrite(self.save_folder + "/" + str(i).zfill(7) + ".png", (x * 256).astype(np.uint8))


np.set_printoptions(precision=4)

start_time = datetime.datetime.now()
print "STARTING"
print "================================================================"

with tf.Session() as sess:
    c = cycle_digit(sess)
    c.train(1000000)

print "================================================================"
print "END:", datetime.datetime.now() - start_time