import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
from datetime import datetime
import os

data_pwd = '/lustrehome/tborghes/compact_data/data_1.h5'
labels_pwd = '/lustrehome/tborghes/compact_data/labels_1.h5'
data_size = 3000000


def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Train Start =", current_time)


def read_data(data_size, data_pwd, labels_pwd):

    print_time()
    data = np.array(pd.read_hdf(data_pwd, key='data', stop=data_size), 
                    dtype = np.float32)

    print_time()
    labels = np.array(pd.read_hdf(labels_pwd, key='data', stop=data_size), 
                      dtype = np.float32)

    image_dims = 4096
    features_num = data.shape[1] - image_dims

    data_len = data.shape[0]
    data_splits = [int(0.85*data_len), int(0.95*data_len)]
    data = data[1:]
    train_data = data[:data_splits[0]]
    valid_data = data[data_splits[0]:data_splits[1]]
    test_data = data[data_splits[1]:]

    labels = labels[1:]
    train_labels = labels[:data_splits[0]]
    valid_labels = labels[data_splits[0]:data_splits[1]]
    test_labels = labels[data_splits[1]:]

    return(train_data, train_labels, valid_data, valid_labels, test_data,
           test_labels, image_dims, features_num)
