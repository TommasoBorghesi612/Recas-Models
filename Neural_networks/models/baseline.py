import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model

class baseline(Model):

    def __init__(self, image_dims, features_num):
        super(baseline, self).__init__()
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
        self.model_name = 'baseline2'

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
