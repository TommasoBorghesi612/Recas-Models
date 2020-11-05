import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from pathlib import Path

home = str(Path.home())


def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Train Start =", current_time)


class Epoch_Routine():

    def __init__(self, Model, image_dims, features_num, main_path, epochs):

        self.model = Model
        self.image_dims = image_dims
        self.features_num = features_num
        self.model_name = "grid_2"

        self.optimizers = [
#            Adadelta,
#            Adagrad,
#            Adam,
#            Adamax,
#            Ftrl,
#            Nadam,
#            RMSprop,
            SGD
        ]            
        self.losses = [
            'CategoricalCrossentropy',
#            'KLDivergence',
#            'MeanAbsoluteError',
#            'Poisson'
        ]
        self.learning_rate = [
            1e-2,
#            1e-3,
#            0.00001,
#            0.000001,
        ]
        self.activation = [
            'relu',
#            'sigmoid',
#            'tanh'
        ]
        self.filters = [
            32,
            16,
#            64
        ]
        self.neurons = [
            256,
            128,
#            512,
#            64
        ]
        self.pooling = [
            2,
            3,
#            4,
#            1
        ]
        self.incr = [
            0,
            1
        ]
        self.filter_inc = [
            1,
            2
        ]
        self.filter_size = [
            3,
            2,
#            4,
#            5
        ]

        self.epochs = epochs
        self.results_path = main_path + "/results/"
        self.models_path = main_path + "/trained/"

    def build_model(self, activation='relu', filters=32, neurons=128, 
                    pooling=2, incr=0, filter_inc=0, filter_size=3, 
                    learning_rate=0.001, optimizer=SGD,
                    loss="CategoricalCrossentropy"):

        model = self.model(self.image_dims, self.features_num, activation,
                           filters, neurons, pooling, incr, 
                           filter_inc, filter_size)

        self.model_name = model.model_name

        model.compile(loss=loss,
                      optimizer=optimizer(lr=learning_rate),
                      metrics=["accuracy"])
        return model

    def name_to_string(self):

        out_filename = str(self.results_path + self.model_name) + ".txt"
        out_err = str(self.results_path + self.model_name) + "_err.txt"
        return(out_filename, out_err)

    def training(self, train_data, train_labels, valid_data, valid_labels,
                 test_data, test_labels):

        out_filename, out_err = self.name_to_string()

        sys.stdout = open(out_filename, "w")
        sys.stderr = open(out_err, "w")

        keras_reg = keras.wrappers.scikit_learn.KerasRegressor(self.build_model)
        param_distribs = {
#            "optimizer" : self.optimizers,
            "loss" : self.losses,
            "learning_rate" : self.learning_rate,
            "activation" : self.activation,
            "filters" : self.filters,
            "neurons" : self.neurons,
            "pooling" : self.pooling,
            "incr" : self.incr,
            "filter_inc" : self.filter_inc,
            "filter_size" : self.filter_size,
        }

        grid_search = GridSearchCV(keras_reg, param_distribs, verbose=1)
        grid_search.fit(train_data, train_labels, epochs=self.epochs,
                        validation_data=(valid_data, valid_labels))


        print(grid_search.cv_results_)
        print(grid_search.best_params_)

        model = grid_search.best_estimator_.model
        model.evaluate(test_data, test_labels)
        
        final_dict = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["accuracy"])],axis=1)

        print(final_dict)
        final_dict.to_csv("grid_s2.csv", mode="w")
