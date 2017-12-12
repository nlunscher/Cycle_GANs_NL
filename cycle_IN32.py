
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

class cycle_IN():

    def __init__(self, sess):
        self.sess = sess
        self.data_folder = 'Data_ImageNet/fast_data/IN_train/'

        # pair = ['947', '232'] # pizza, balloon
        # pair = ['2', '210'] # huskey, german shepard
        pair = ['75', '205'] # tiger, cheeta

        self.a_data = self.data_folder + pair[0]
        self.b_data = self.data_folder + pair[1]
        self.im_per_class = 2600-1
        self.batch_size = 1

        self.learning_rate = 1e-4 * 2

        self.main_folder = 'cin_saves32/'
        make_directory(self.main_folder)
        self.save_folder = self.main_folder + '32_DOG'
        make_directory(self.save_folder)


    def load_pair(self):
        a_im_num = random.randint(0, self.im_per_class)
        b_im_num = random.randint(0, self.im_per_class)

        a = cv2.imread(self.a_data + '/' + str(a_im_num).zfill(7) + '.png')
        b = cv2.imread(self.b_data + '/' + str(b_im_num).zfill(7) + '.png')

        a = np.array(a, dtype=np.float32) / 255.
        b = np.array(b, dtype=np.float32) / 255.

        return a, b

    def load_batch(self):
        a_s = []
        b_s = []

        for i in range(self.batch_size):
            a, b = self.load_pair()
            a_s.append(a)
            b_s.append(b)

        return np.asarray(a_s), np.asarray(b_s)

    def create_net(self):
        self.a_im = tf.placeholder(tf.float32, name='a_image',shape=[self.batch_size] + [32,32,3])
        self.b_im = tf.placeholder(tf.float32, name='b_image',shape= [self.batch_size] + [32,32,3])

        self.is_train = tf.placeholder(tf.bool, name = 'is_train')

        a2b_W_conv1 = weight_variable([7, 7, 3, 32])
        a2b_W_conv2 = weight_variable([5, 5, 32, 64])
        
        a2b_W_conv3_1 = weight_variable([3, 3, 64, 64])
        a2b_W_conv3_2 = weight_variable([3, 3, 64, 64])

        a2b_W_conv4_1 = weight_variable([3, 3, 64, 64])
        a2b_W_conv4_2 = weight_variable([3, 3, 64, 64])
        
        a2b_W_dconv1 = weight_variable([5, 5, 32, 64])
        a2b_b_dconv1 = bias_variable([32])
        a2b_W_dconv2 = weight_variable([7, 7, 3, 32])
        a2b_b_dconv2 = bias_variable([3])

        self.a2b_vars = [a2b_W_conv1, a2b_W_conv2, 
                        a2b_W_conv3_1, a2b_W_conv3_2, a2b_W_conv4_1, a2b_W_conv4_2, 
                        a2b_W_dconv1, a2b_b_dconv1, a2b_W_dconv2, a2b_b_dconv2]

        def a2b(a_image, is_train):
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(a_image, a2b_W_conv1, 2), training=is_train), name="a2b_conv1") #16
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, a2b_W_conv2, 2), training=is_train), name="a2b_conv2") #8

            h_c3_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, a2b_W_conv3_1, 1), training=is_train), name="a2b_h_c3_1")
            h_c3_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c3_1, a2b_W_conv3_2, 1), training=is_train) + h_conv2, name="a2b_h_c3_2")

            h_c4_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c3_2, a2b_W_conv4_1, 1), training=is_train), name="a2b_h_c4_1")
            h_c4_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c4_1, a2b_W_conv4_2, 1), training=is_train) + h_c3_2, name="a2b_h_c4_2")

            osize_dconv1 = h_conv1.get_shape().as_list()
            osize_dconv1[0] = self.batch_size
            h_dconv1 = tf.nn.relu(conv2d_transpose(h_c4_2, a2b_W_dconv1, osize_dconv1,2) + a2b_b_dconv1, name="a2b_dconv1") #16
            osize_dconv2 = a_image.get_shape().as_list()
            osize_dconv2[0] = self.batch_size
            h_dconv2 = tf.nn.sigmoid(conv2d_transpose(h_dconv1, a2b_W_dconv2, osize_dconv2,2) + a2b_b_dconv2, name="a2b_dconv2") #32

            return h_dconv2

        bD_W_conv1 = weight_variable([7, 7, 3, 32])
        bD_W_conv2 = weight_variable([5, 5, 32, 64])
        bD_W_conv3 = weight_variable([3, 3, 64, 64])
        bD_W_f1 = weight_variable([1024, 256])
        bD_b_f1 = bias_variable([256])
        bD_W_f2 = weight_variable([256, 1])
        bD_b_f2 = bias_variable([1])

        self.bD_vars = [bD_W_conv1, bD_W_conv2, bD_W_conv3, bD_W_f1, bD_b_f1, bD_W_f2, bD_b_f2]

        def bD(b_image, is_train):
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(b_image, bD_W_conv1, 2), training=is_train), name="bD_conv1") #16
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, bD_W_conv2, 2), training=is_train), name="bD_conv2") #8
            h_conv3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, bD_W_conv3, 2), training=is_train), name="bD_conv3") #4
            h3_flat = tf.reshape(h_conv3, [-1, 4*4*64])
            h4 = tf.nn.sigmoid(tf.matmul(h3_flat, bD_W_f1) + bD_b_f1, name="bD_f1")
            h5 = tf.matmul(h4, bD_W_f2) + bD_b_f2
            h5_sig = tf.nn.sigmoid(h5, name="bD_f2")

            return h5, h5_sig

        b2a_W_conv1 = weight_variable([7, 7, 3, 32])
        b2a_W_conv2 = weight_variable([5, 5, 32, 64])

        b2a_W_conv3_1 = weight_variable([3, 3, 64, 64])
        b2a_W_conv3_2 = weight_variable([3, 3, 64, 64])

        b2a_W_conv4_1 = weight_variable([3, 3, 64, 64])
        b2a_W_conv4_2 = weight_variable([3, 3, 64, 64])

        b2a_W_dconv1 = weight_variable([5, 5, 32, 64])
        b2a_b_dconv1 = bias_variable([32])
        b2a_W_dconv2 = weight_variable([7, 7, 3, 32])
        b2a_b_dconv2 = bias_variable([3])

        self.b2a_vars = [b2a_W_conv1, b2a_W_conv2,
                        b2a_W_conv3_1, b2a_W_conv3_2, b2a_W_conv4_1, b2a_W_conv4_2,
                        b2a_W_dconv1, b2a_b_dconv1, b2a_W_dconv2, b2a_b_dconv2]

        def b2a(b_image, is_train):
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(b_image, b2a_W_conv1, 2), training=is_train), name="b2a_conv1") #16
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, b2a_W_conv2, 2), training=is_train), name="b2a_conv2") #8

            h_c3_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, b2a_W_conv3_1, 1), training=is_train), name="b2a_h_c3_1")
            h_c3_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c3_1, b2a_W_conv3_2, 1), training=is_train) + h_conv2, name="b2a_h_c3_2")

            h_c4_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c3_2, b2a_W_conv4_1, 1), training=is_train), name="b2a_h_c4_1")
            h_c4_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_c4_1, b2a_W_conv4_2, 1), training=is_train) + h_c3_2, name="b2a_h_c4_2")

            osize_dconv1 = h_conv1.get_shape().as_list()
            osize_dconv1[0] = self.batch_size
            h_dconv1 = tf.nn.relu(conv2d_transpose(h_c4_2, b2a_W_dconv1, osize_dconv1,2) + b2a_b_dconv1, name="b2a_dconv1") #16
            osize_dconv2 = b_image.get_shape().as_list()
            osize_dconv2[0] = self.batch_size
            h_dconv2 = tf.nn.sigmoid(conv2d_transpose(h_dconv1, b2a_W_dconv2, osize_dconv2,2) + b2a_b_dconv2, name="b2a_dconv2") #32

            return h_dconv2

        aD_W_conv1 = weight_variable([7, 7, 3, 32])
        aD_W_conv2 = weight_variable([5, 5, 32, 64])
        aD_W_conv3 = weight_variable([3, 3, 64, 64])
        aD_W_f1 = weight_variable([1024, 256])
        aD_b_f1 = bias_variable([256])
        aD_W_f2 = weight_variable([256, 1])
        aD_b_f2 = bias_variable([1])

        self.aD_vars = [aD_W_conv1, aD_W_conv2, aD_W_conv3, aD_W_f1, aD_b_f1, aD_W_f2, aD_b_f2]

        def aD(a_image, is_train):
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(a_image, aD_W_conv1, 2), training=is_train), name="aD_conv1") #16
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, aD_W_conv2, 2), training=is_train), name="aD_conv2") #8
            h_conv3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, aD_W_conv3, 2), training=is_train), name="aD_conv3") #4
            h3_flat = tf.reshape(h_conv3, [-1, 4*4*64])
            h4 = tf.nn.sigmoid(tf.matmul(h3_flat, aD_W_f1) + aD_b_f1, name="aD_f1")
            h5 = tf.matmul(h4, aD_W_f2) + aD_b_f2
            h5_sig = tf.nn.sigmoid(h5, name="aD_f2")

            return h5, h5_sig
        
        # G
        self.a2b_im = a2b(self.a_im, self.is_train)
        self.b2a_im = b2a(self.b_im, self.is_train)

        self.a2b2a_im = b2a(self.a2b_im, self.is_train)
        self.b2a2b_im = a2b(self.b2a_im, self.is_train)

        a_G_fake, self.a_G_fake_sig = aD(self.b2a_im, self.is_train)
        b_G_fake, self.b_G_fake_sig = bD(self.a2b_im, self.is_train)

        self.a2b_Ad = tf.reduce_mean(tf.square(self.b_G_fake_sig - tf.ones_like(self.b_G_fake_sig)))
        self.b2a_Ad = tf.reduce_mean(tf.square(self.a_G_fake_sig - tf.ones_like(self.a_G_fake_sig)))

        self.cycle_loss = 5 * (tf.reduce_mean(tf.reduce_sum(tf.abs(self.a2b2a_im - self.a_im), 3)) +
                                tf.reduce_mean(tf.reduce_sum(tf.abs(self.b2a2b_im - self.b_im), 3)))

        self.G_loss = self.a2b_Ad + self.b2a_Ad + self.cycle_loss

        # D
        # b_D_fake, self.b_D_fake_sig = bD(self.a2b_im, self.is_train)
        b_D_fake, self.b_D_fake_sig = b_G_fake, self.b_G_fake_sig
        b_D_real, self.b_D_real_sig = bD(self.b_im, self.is_train)

        # a_D_fake, self.a_D_fake_sig = aD(self.b2a_im, self.is_train)
        a_D_fake, self.a_D_fake_sig = a_G_fake, self.a_G_fake_sig
        a_D_real, self.a_D_real_sig = aD(self.a_im, self.is_train)

        self.bD_loss_real = tf.reduce_mean(tf.square(self.b_D_real_sig - tf.ones_like(self.b_D_real_sig)))
        self.bD_loss_fake = tf.reduce_mean(tf.square(self.b_D_fake_sig - tf.zeros_like(self.b_D_fake_sig)))
        self.bD_loss = self.bD_loss_real + self.bD_loss_fake

        self.aD_loss_real = tf.reduce_mean(tf.square(self.a_D_real_sig - tf.ones_like(self.a_D_real_sig)))
        self.aD_loss_fake = tf.reduce_mean(tf.square(self.a_D_fake_sig - tf.zeros_like(self.a_D_fake_sig)))
        self.aD_loss = self.aD_loss_real + self.aD_loss_fake

        self.D_loss = (self.bD_loss + self.aD_loss) / 2


    def train(self, iterations):

        cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Images", 256*3, 256*2)

        self.create_net()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            G_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.G_loss, var_list=(self.a2b_vars + self.b2a_vars))
            D_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.D_loss, var_list=(self.bD_vars + self.aD_vars))

        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord = self.coord)

        print 'Training for', iterations
        for i in range(iterations):
            a_reals, b_reals = self.load_batch()

            _, _, m, a2b, a2b2a, s, b2a, b2a2b, \
                        a2bDS, b2aDS, \
                        a2bD, b2aD, cL, \
                        aDS, aDFS, bDS, bDFS, \
                        aD, mFD, bD, sFD = self.sess.run([G_train_step, D_train_step, 
                                                self.a_im, self.a2b_im, self.a2b2a_im, self.b_im, self.b2a_im, self.b2a2b_im,
                                                self.b_G_fake_sig, self.a_G_fake_sig, 
                                                self.a2b_Ad, self.b2a_Ad, self.cycle_loss,
                                                self.a_D_real_sig, self.a_D_fake_sig, self.b_D_real_sig, self.b_D_fake_sig,
                                                self.aD_loss_real, self.aD_loss_fake, self.bD_loss_real, self.bD_loss_fake
                                                ], 
                                                    feed_dict={self.a_im:a_reals, self.b_im:b_reals,
                                                                self.is_train:True})

            if (i % 100 == 0 or i == iterations-1):
                print i, " =======================================================================", datetime.datetime.now() - start_time
                print "        ", "Max m:", np.max(m[0]), "Max a2b:", np.max(a2b), "Max a2b2a:", np.max(a2b2a)
                print "        ", "Max s:", np.max(s[0]), "Max b2a:", np.max(b2a), "Max b2a2b:", np.max(b2a2b)
                print "        ", "a2bDS:", a2bDS[0], "b2aDS:", b2aDS[0]
                print "        ", "a2bD:", a2bD, "b2aD:", b2aD, "cL:", cL
                print "        ", "aDS:", aDS[0], "aDFS:", aDFS[0], "bDS:", bDS[0], "bDFS:", bDFS[0]
                print "        ", "aD:", aD, "mFD:", mFD, "bD:", bD, "sFD:", sFD

                x = np.concatenate([np.concatenate([m[0], a2b[0], a2b2a[0]], axis = 1),
                                    np.concatenate([s[0], b2a[0], b2a2b[0]], axis = 1)], axis = 0)
                cv2.imshow('Images', x)
                cv2.waitKey(50)

                if (i % 1000 == 0 or i == iterations-1):
                    cv2.imwrite(self.save_folder + "/" + str(i).zfill(7) + ".png", (x * 256).astype(np.uint8))


np.set_printoptions(precision=4)

start_time = datetime.datetime.now()
print "STARTING"
print "================================================================"

with tf.Session() as sess:
    c = cycle_IN(sess)
    c.train(1000000)

print "================================================================"
print "END:", datetime.datetime.now() - start_time