import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
from datetime import datetime

class model(Model):

    def __init__(self, image_dims, features_num):
        super(model, self).__init__()
        self.conv1 = Conv2D(16, (3, 3), activation='relu', name='conv1')
        self.conv2 = Conv2D(32, (3, 3), activation='relu', name='conv2')
        self.pool1 = MaxPooling2D(pool_size=(2, 2), padding='same',
                                  name='pool1')
        self.conv3 = Conv2D(32, (3, 3), activation='relu', padding='same',
                            name='conv3')
        self.conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',
                            name='conv4')
        self.pool2 = MaxPooling2D(pool_size=(2, 2), padding='same',
                                  name='pool2')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu', name='dense1')
        self.d2 = Dense(64, activation='relu', name='dense2')
        self.d3 = Dense(2, activation='relu', name='output')
        self.b_norm_1 = tf.keras.layers.BatchNormalization(name='norm1')
        self.b_norm_2 = tf.keras.layers.BatchNormalization(name='norm2')
        self.image_dims = image_dims
        self.features_num = features_num

    def call(self, X):
        images, features = tf.split(X, [self.image_dims, self.features_num], axis=1)
        images = tf.reshape(images, [-1, 16, 16, 16])
        c1 = self.conv1(images)
        c2 = self.conv2(c1)
        p1 = self.pool1(c2)
        c3 = self.conv3(p1)
        c4 = self.conv4(c3)
        p2 = self.pool2(c4)
        f = self.flatten(p2)
        norm1 = self.b_norm_1(f)
        norm2 = self.b_norm_2(features)
        flat = tf.concat([norm1, norm2], axis=1)
        d1 = self.d1(flat)
        d2 = self.d2(d1)
        out = self.d3(d2)
        return out

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Load Data =", current_time)

data = np.array(pd.read_hdf('/lustrehome/tborghes/compact_data/data_1.h5', key='data', stop=100000), dtype = np.float32)

# data1 = np.array(pd.read_hdf('/lustrehome/tborghes/compact_data/data_1.h5', key='data', start=1000000, stop=1500000), dtype = np.float32)

# data = tf.concat[data, data1], axis=0)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Load Data =", current_time)

labels = np.array(pd.read_hdf('/lustrehome/tborghes/compact_data/labels_1.h5', key='data', stop=100000), dtype = np.float32)

# labels1 = np.array(pd.read_hdf('/lustrehome/tborghes/compact_data/labels_1.h5', key='data', start=1000000, stop=1500000), dtype = np.float32)

# labels = tf.concat([labels, = labels1], axis=0)
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

sys.stdout = open("baseline_16ch.txt", "w")
sys.stderr = open("baseline_16ch_err.txt", "w")

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_data, train_labels)).batch(64)

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_data, test_labels)).batch(100)

val_ds = tf.data.Dataset.from_tensor_slices(
    (valid_data, valid_labels)).batch(100)

model = model(image_dims, features_num)

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.\
                 CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.\
                CategoricalAccuracy(name='test_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.\
                CategoricalAccuracy(name='val_accuracy')


# Training Function
@tf.function
def train_step(data, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(data, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# Test Function
@tf.function
def test_step(data, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).

    predictions = model(data, training=False)

    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

@tf.function
def val_step(data, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).

    predictions = model(data, training=False)

    v_loss = loss_object(labels, predictions)

    val_loss(v_loss)
    val_accuracy(labels, predictions)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Train Start =", current_time)

# Actual Trainig
EPOCHS = 20
for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for data, labels in train_ds:
        train_step(data, labels)

    for test_data, test_labels in test_ds:
        test_step(test_data, test_labels)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Epoch End =", current_time)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, \
                Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))

for val_data, val_labels in val_ds:
    val_step(val_data, val_labels)

template = 'Validation Results: Accuracy = {}, Loss = {}'
print(template.format(val_accuracy.result()*100,
                      val_loss.result()))

tf.saved_model.save(model, 'baseline_16ch')

sys.stdout.close()
