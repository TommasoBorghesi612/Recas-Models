import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import h5py

def data_parser():

    x1 = [1, 10, 11, 15, 16, 20]

    x2 = [[1,148], [5663,5806], [6325,6484], [8988,9134], [9635,9785], [11987,12080]]

    data = np.zeros(shape = [1, 4187])
    labels = np.zeros(shape = [1, 2])
    b = 0

    for k in range(len(x1)):

        index = x1[k]
        min = x2[k][0]
        max = x2[k][1]

        for i in range(min, max):
            num = i+1
            file_name = str('/lustrehome/redo/seedingcnn/_bal_' + str(index) + '_1_' + str(num) + '_pixelTracksHitDoublets_dnn_doublets.h5')

            b += 1
            print (b/840)

            if i != 48:
                temp, labels = data_handler(np.array(pd.read_hdf(file_name, key='data'), dtype=np.float32))

                temp_data = make_split(temp)
                temp_labels = make_split(labels)

                data = np.concatenate([data, temp_data])
                labels = np.concatenate([labels, temp_labels])

    return(data, labels)

def make_split(temp):

    temp_len = int(temp.shape[0])
    first_stop = int(temp_len*0.25)
    second_stop = int(first_stop + temp_len*0.25)
    third_stop = int(second_stop + temp_len*0.25)

    data = temp[:first_stop]
    return (data)

def data_handler(data):
    
    channel_1 = channel_to_onehot(data[:,4])
    channel_2 = channel_to_onehot(data[:,5])
    feat_1 = data[:, 6:39]
    img_1 = img_handler(tf.cast(data[:, 39:295], dtype = tf.float32), img_mean = 13418.961)
    feat_2 = data[:, 295:325]
    img_2 = img_handler(tf.cast(data[:, 325:581], dtype = tf.float32), img_mean = 10059.948)
    feat_3 = data[:, 581:589]
    labels = data[:, 590]

    data_len = data.shape[0]

    onehot_channels = tf.concat([channel_1, channel_2], axis=1)
    norm_images = img_composer(img_1, img_2, channel_1, channel_2, data_len)
    labels = labels_handler(labels)
    features = np.concatenate([onehot_channels, feat_1, feat_2, feat_3], axis=1)

    temp = tf.concat([norm_images, features], axis=1)

    return temp, labels

def channel_to_onehot(channel):

    fix1 = np.minimum(channel, abs(channel-10))
    fix2 = np.minimum(fix1, abs(fix1-12))
    onehot = tf.one_hot(fix2, 10)

    return(onehot)


def img_handler(img, img_mean):

    norm_img = tf.math.sigmoid(img/img_mean)
    norm_img = np.reshape(norm_img, [-1, 1, 256])

    return (norm_img)


def img_composer(img1, img2, channel1, channel2, data_len):

    channel1 = tf.reshape(channel1, [-1, 10, 1])
    channel2 = tf.reshape(channel2, [-1, 10, 1])

    full_image1 = np.ones([data_len, 10, 256])                      # [-1, 10, 256]
    full_image1 = tf.math.multiply(full_image1, channel1)     # [-1, 10, 256]
    full_image1 = tf.math.multiply(full_image1, img1)         # [-1, 10, 256]

    full_image2 = np.ones([data_len, 10, 256])                      # [-1, 10, 256]
    full_image2 = tf.math.multiply(full_image2, channel2)     # [-1, 10, 256]
    full_image2 = tf.math.multiply(full_image2, img2)         # [-1, 10, 256]

    mask_img1 = np.array([True, True, True, False, True, True, False, True, True, False])
    mask_img2 = np.array([False, True, True, True, True, True, True, True, True, True])

    complete_img1 = tf.boolean_mask(full_image1, mask_img1, axis=1)  # [-1, 7, 256]
    complete_img2 = tf.boolean_mask(full_image2, mask_img2, axis=1)  # [-1, 9, 256]

    complete_image = tf.reshape(tf.concat([complete_img1, complete_img2], axis=1), [-1, 256*16])   # [-1, 16, 256]

    return complete_image


def labels_handler(labels):

    zeros = np.zeros(labels.shape, dtype = np.float32)
    true_labels = tf.reshape(tf.math.maximum(labels, 0), [-1,1])
    false_labels = tf.reshape(tf.math.minimum(labels, 0), [-1,1]) * (-1)
    new_labels = tf.concat([true_labels, false_labels], axis=1)

    return(new_labels)


data_split, label_split = data_parser()

data_split = pd.DataFrame(data_split)
data_split.to_hdf('data_split.h5', key='data', mode='w')
 
label_split = pd.DataFrame(label_split)
label_split.to_hdf('labels_split.h5', key='data', mode='w')

print('end')
