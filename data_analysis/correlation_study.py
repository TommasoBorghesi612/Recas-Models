import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import tensorflow_probability as tfp


np.set_printoptions(threshold=sys.maxsize)

x1 = [1, 10, 11, 15, 16, 20]

x2 = [[1,148], [5663,5806], [6325,6484], [8988,9134], [9635,9785], [11987,12080]]

data = np.zeros(shape = [1, 645])

b = 0

for k in range(len(x1)):

    index = x1[k]
    min = x2[k][0]
    max = x2[k][1]

    for i in range(min, max):
        num = i+1
        name = str('/lustrehome/redo/seedingcnn/_bal_' + str(index) + '_1_' + str(num) + '_pixelTracksHitDoublets_dnn_doublets.h5')

        b += 1
        print (b/840)

        if i != 48:
            temp = np.array(pd.read_hdf(name, key='data'), dtype=np.float32)

            data = np.concatenate([data, temp])


feat_1 = data[1:, :39]
feat_2 = data[1:, 295:325]
feat_3 = data[1:, 591:599]
feat_4 = data[1:, 600:]

print(data.shape)

features = np.concatenate([feat_1, feat_2, feat_3, feat_4], axis=1)

print(features.shape)

sys.stdout = open("printed_data.txt", "w")

correlated_val = []

features_number = features.shape[1]

devs = []

for feat in range(features_number):
    feat_tensor = np.reshape(features[:,feat], [-1])
    std = np.std(feat_tensor)
    if std == 0:
        devs.append(feat)

print(devs)


for feat_x in range(features_number):
    feat_one = np.reshape(features[:,feat_x], [-1,1])
    for feat_y in range(feat_x+1, features_number):
        feat_two = np.reshape(features[:, feat_y], [-1,1])
        corr = tfp.stats.correlation(feat_one, feat_two, sample_axis=0, event_axis=-1)
        if corr < -0.95 or corr > 0.95:
            correlated_val.append([feat_x, feat_y])

print(correlated_val)


meaning_features = np.delete(features, devs, axis=1)
features_number = meaning_features.shape[1]

correlated_val_M = []

for feat_x in range(features_number):
    feat_one = np.reshape(meaning_features[:,feat_x], [-1,1])
    for feat_y in range(feat_x+1, features_number):
        feat_two = np.reshape(meaning_features[:, feat_y], [-1,1])
        corr = tfp.stats.correlation(feat_one, feat_two, sample_axis=0, event_axis=-1)
        if corr < -0.95 or corr > 0.95:
            correlated_val_M.append([feat_x, feat_y])


print(correlated_val_M)
print(len(devs))
print(meaning_features.shape)

