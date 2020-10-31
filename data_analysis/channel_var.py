import sys

sys.stdout = open("channel_variance.txt", "w")

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp


np.set_printoptions(threshold=sys.maxsize)

x1 = [1, 10, 11, 15, 16, 20]

x2 = [[1,148], [5663,5806], [6325,6484], [8988,9134], [9635,9785], [11987,12080]]


data = np.zeros(shape = [1, 2])

image1 = np.zeros(shape = [1, 256])
image2 = np.zeros(shape = [1, 256])


b = 0

for k in range(1): # len(x1)):

    index = x1[k]
    min = x2[k][0]
    max = x2[k][1]


    for i in range(min, min+2): #max):

        num = i+1
        name = str('/lustrehome/redo/seedingcnn/_bal_' + str(index) + '_1_' + str(num) + '_pixelTracksHitDoublets_dnn_doublets.h5')

        b += 1
        # print (b/840)

        if i != 48:
            temp = np.array(pd.read_hdf(name, key='data'), dtype=np.float32)

            temp_img1 = temp[:, 39:295]
            temp_img2 = temp[:, 325:581]
            temp = temp[:, 4:6]
            data = np.concatenate([data, temp])
            image1 = np.concatenate([image1, temp_img1])
            image2 = np.concatenate([image2, temp_img2])



data = data[1:]
image1 = tf.cast(tf.reshape(image1[1:], [-1, 1, 256]), dtype=tf.float32)
image2 = tf.cast(tf.reshape(image2[1:], [-1, 1, 256]), dtype=tf.float32)

max_img1 = tf.math.reduce_max(image1)
max_img2 = tf.math.reduce_max(image2)

nonzero_img1 = tf.cast(tf.math.count_nonzero(image1), dtype = tf.float32)
nonzero_img2 = tf.cast(tf.math.count_nonzero(image2), dtype = tf.float32)

mean_img1 = tf.math.reduce_sum(image1)/nonzero_img1
mean_img2 = tf.math.reduce_sum(image2)/nonzero_img1

data_len = data.shape[0]

channel1 = data[:,0]
channel2 = data[:,1]

fix1_ch1 = np.minimum(channel1, abs(channel1-10))
fix2_ch1 = np.minimum(fix1_ch1, abs(fix1_ch1-12))

onehot_ch1 = tf.reshape(tf.one_hot(fix2_ch1, 10), [-1,10,1])

ch1_act = tf.math.reduce_sum(tf.squeeze(onehot_ch1), 0)/data_len

fix1_ch2 = np.minimum(channel2, abs(channel2-10))
fix2_ch2 = np.minimum(fix1_ch2, abs(fix1_ch2-12))

onehot_ch2 = tf.reshape(tf.one_hot(fix2_ch2, 10), [-1,10,1])

ch2_act = tf.math.reduce_sum(tf.squeeze(onehot_ch2), 0)/data_len

print('chanel_1 activities')
print(ch1_act)
print('channel_2 activities')
print(ch2_act)

norm_img1 = tf.math.sigmoid(image1/mean_img1)
norm_img2 = tf.math.sigmoid(image2/mean_img2)


full_image1 = np.ones([data_len, 10, 256])
full_image1 = tf.math.multiply(full_image1, onehot_ch1)
full_image1 = tf.math.multiply(full_image1, norm_img1)

var_img1 = tf.transpose(full_image1, [0, 2, 1]) # [-1, 256, 10]

var_img1 = tfp.stats.variance(var_img1) # [256, 10]

print(var_img1.shape)

var_sum_img1 = tf.math.reduce_mean(var_img1, axis=0) # [10]

print(var_sum_img1.shape)

full_image2 = np.ones([data_len, 10, 256])
full_image2 = tf.math.multiply(full_image2, onehot_ch2)
full_image2 = tf.math.multiply(full_image2, norm_img2)
 
var_img2 = tf.transpose(full_image2, [0, 2, 1]) # [-1, 256, 10]

var_img2 = tfp.stats.variance(var_img2) # [256, 10]
 
print(var_img2.shape)
 
var_sum_img2 = tf.math.reduce_mean(var_img2, axis=0) 


print(var_sum_img1)
print(var_sum_img2)

print("max and mean")
print(max_img1)
print(max_img2)
print(mean_img1)
print(mean_img2)
