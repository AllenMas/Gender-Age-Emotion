from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def emotion_net():
    flatten_1 = Input(shape=(131072,))
    predictions_e = Dense(units=5, kernel_initializer="he_normal", use_bias=False, 
                          kernel_regularizer=l2(0.0005), 
                          activation="softmax")(flatten_1)
    model = Model(inputs = flatten_1, outputs=[predictions_e])
    return model