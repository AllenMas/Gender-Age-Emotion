from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def gender_age_net():
    flatten_1 = Input(shape=(131072,))
    predictions_g = Dense(units=2, kernel_initializer="he_normal", use_bias=False,
                          kernel_regularizer=l2(0.0005), 
                          activation="softmax", name="dense_1")(flatten_1)
    predictions_a = Dense(units=101, kernel_initializer="he_normal", use_bias=False,
                          kernel_regularizer=l2(0.0005), 
                          activation="softmax", name="dense_2")(flatten_1)
    model = Model(inputs = flatten_1, outputs=[predictions_g, predictions_a])
    return model