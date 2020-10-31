import sys

sys.stdout = open("channel.txt", "w")

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp


np.set_printoptions(threshold=sys.maxsize)

x1 = [1, 10, 11, 15, 16, 20]

x2 = [[1,148], [5663,5806], [6325,6484], [8988,9134], [9635,9785], [11987,12080]]


data = np.zeros(shape = [1, 2])

b = 0

for k in range(len(x1)):

    index = x1[k]
    min = x2[k][0]
    max = x2[k][1]


    for i in range(min, max):

        num = i+1
        name = str('/lustrehome/redo/seedingcnn/_bal_' + str(index) + '_1_' + str(num) + '_pixelTracksHitDoublets_dnn_doublets.h5')

        b += 1
        # print (b/840)

        if i != 48:
            temp = np.array(pd.read_hdf(name, key='data'), dtype=np.float32)

            temp = temp[:, 4:6]
            data = np.concatenate([data, temp])


data = data[1:]

data_len = data.shape[0]

channel1 = data[:,0]
channel2 = data[:, 1]

fix1_ch1 = np.minimum(channel1, abs(channel1-10))
fix2_ch1 = np.minimum(fix1_ch1, abs(fix1_ch1-12))

onehot_ch1 = tf.one_hot(fix2_ch1, 10)

ch1_act = tf.math.reduce_sum(onehot_ch1, 0)/data_len


fix1_ch2 = np.minimum(channel2, abs(channel2-10))
fix2_ch2 = np.minimum(fix1_ch2, abs(fix1_ch2-12))

onehot_ch2 = tf.one_hot(fix2_ch2, 10)

ch2_act = tf.math.reduce_sum(onehot_ch2, 0)/data_len

print('chanel_1 activities')
print(ch1_act)
print('channel_2 activities')
print(ch2_act)
