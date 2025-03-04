import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
from datetime import datetime
import os

sys.path.insert(1, 'utils')
sys.path.insert(1, 'models')

from train_and_test import Epoch_Routine

from read_data import read_data, print_time, data_pwd, labels_pwd, data_size

from baseline import baseline

main_path = str(os.path.dirname(os.path.realpath(__file__)))

models = [
        baseline,
    ]

epochs = 20

train_data, train_labels, valid_data, valid_labels, test_data, test_labels, \
 image_dims, features_num = read_data(data_size, data_pwd, labels_pwd)

for model in models:
    epoch_routine = Epoch_Routine(model, image_dims, features_num,
                                  epochs, main_path)

    epoch_routine.training(train_data, train_labels, valid_data, valid_labels,
                           test_data, test_labels)
