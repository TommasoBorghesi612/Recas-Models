
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model


class baseline_grid(Model):
    
    def __init__(self, image_dims, features_num, activation,
                 filters, neurons, pooling, incr, filter_inc,
                 filter_size):
        super(baseline_grid, self).__init__()
        self.conv1 = Conv2D(filters, (filter_size, filter_size), activation=activation, name='conv1')
        self.conv2 = Conv2D(2*filters, (filter_size+incr, filter_size+incr), activation=activation, name='conv2')
        self.pool1 = MaxPooling2D(pool_size=(pooling, pooling), padding='same',
                                  name='pool1')
        self.conv3 = Conv2D(2*filters*filter_inc, (filter_size+incr, filter_size+incr), activation=activation, padding='same',
                            name='conv3')
        self.conv4 = Conv2D(4*filters*filter_inc, (filter_size+incr, filter_size+incr), activation=activation, padding='same',
                            name='conv4')
        self.pool2 = MaxPooling2D(pool_size=(pooling+2*incr, pooling+2*incr), padding='same',
                                  name='pool2')
        self.flatten = Flatten()
        self.d1 = Dense(neurons, activation=activation, name='dense1')
        self.d2 = Dense(int(neurons/2), activation=activation, name='dense2')
        self.d3 = Dense(2, activation='sigmoid', name='output')
        self.b_norm_1 = tf.keras.layers.BatchNormalization(name='norm1')
        self.b_norm_2 = tf.keras.layers.BatchNormalization(name='norm2')
        self.image_dims = image_dims
        self.features_num = features_num
        self.model_name = 'baseline_grid'

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
