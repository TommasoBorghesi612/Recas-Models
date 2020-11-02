import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
from datetime import datetime
from pathlib import Path
home = str(Path.home())


def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Train Start =", current_time)


class Epoch_Routine():

    def __init__(self, Model, image_dims, features_num, epochs, main_path):
        self.model = Model(image_dims, features_num)
        self.loss_object = tf.keras.losses.\
            CategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.\
            CategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.\
            CategoricalAccuracy(name='test_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.\
            CategoricalAccuracy(name='val_accuracy')
        self.epochs = epochs
        self.model_name = self.model.model_name
        self.results_path = main_path + "/results/"
        self.models_path = main_path + "/trained/"

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(data, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                           self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    # Test Function
    @tf.function
    def test_step(self, data, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(data, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    @tf.function
    def val_step(self, data, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(data, training=False)
        v_loss = self.loss_object(labels, predictions)
        self.val_loss(v_loss)
        self.val_accuracy(labels, predictions)

    def training(self, train_data, train_labels, valid_data, valid_labels,
                 test_data, test_labels):

        train_ds = tf.data.Dataset.from_tensor_slices(
            (train_data, train_labels)).batch(64)

        test_ds = tf.data.Dataset.from_tensor_slices(
            (test_data, test_labels)).batch(100)

        val_ds = tf.data.Dataset.from_tensor_slices(
            (valid_data, valid_labels)).batch(100)

        out_filename, out_err = self.name_to_string()

        sys.stdout = open(out_filename, "w")
        sys.stderr = open(out_err, "w")

        print('Train start')
        print_time()
        for epoch in range(self.epochs):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for data, labels in train_ds:
                self.train_step(data, labels)

            for test_data, test_labels in test_ds:
                self.test_step(test_data, test_labels)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, \
                        Test Accuracy: {}'
            print(template.format(epoch+1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result()*100,
                                  self.test_loss.result(),
                                  self.test_accuracy.result()*100))
            print_time()

        for val_data, val_labels in val_ds:
            self.val_step(val_data, val_labels)
        template = 'Validation Results: Accuracy = {}, Loss = {}'
        print(template.format(self.val_accuracy.result()*100,
                              self.val_loss.result()))
        tf.saved_model.save(self.model,
                            str(self.models_path + self.model_name))

    def name_to_string(self):
        out_filename = str(self.results_path + self.model_name) + ".txt"
        out_err = str(self.results_path + self.model_name) + "_err.txt"
        return(out_filename, out_err)
