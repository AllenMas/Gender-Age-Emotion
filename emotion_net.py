from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def emotion_net():
#     activation_9 = Input(shape=(32, 32, 256))
#       
#     conv2d_12 = Conv2D(512, kernel_size=(3, 3), strides=2, padding="same",
#                         kernel_initializer="he_normal",
#                         kernel_regularizer=l2(0.0005),
#                         use_bias=False)(activation_9)
#     batch_normalization_10 = BatchNormalization(axis=-1)(conv2d_12)
#     activation_10 = Activation("relu")(batch_normalization_10)
#     conv2d_13 = Conv2D(512, kernel_size=(3, 3), strides=1, padding="same",
#                         kernel_initializer="he_normal",
#                         kernel_regularizer=l2(0.0005),
#                         use_bias=False)(activation_10)
#     conv2d_14 = Conv2D(512, kernel_size=(3, 3), strides=2, padding="same",
#                         kernel_initializer="he_normal",
#                         kernel_regularizer=l2(0.0005),
#                         use_bias=False)(activation_9)
#     add_5 = add([conv2d_13, conv2d_14])
#       
#     batch_normalization_11 = BatchNormalization(axis=-1)(add_5)
#     activation_11 = Activation("relu")(batch_normalization_11)
#     conv2d_15 = Conv2D(512, kernel_size=(3, 3), strides=1, padding="same",
#                         kernel_initializer="he_normal",
#                         kernel_regularizer=l2(0.0005),
#                         use_bias=False)(activation_11)
#     batch_normalization_12 = BatchNormalization(axis=-1)(conv2d_15)
#     activation_12 = Activation("relu")(batch_normalization_12)
#     conv2d_16 = Conv2D(512, kernel_size=(3, 3), strides=1, padding="same",
#                         kernel_initializer="he_normal",
#                         kernel_regularizer=l2(0.0005),
#                         use_bias=False)(activation_12)
#     add_6 = add([conv2d_16, add_5])
#       
#     batch_normalization_13 = BatchNormalization(axis=-1)(add_6)
#     activation_13 = Activation("relu")(batch_normalization_13)
#     average_pooling2d_1 = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(activation_13)
#     flatten_1 = Flatten()(average_pooling2d_1)
#     predictions_g = Dense(units=2, kernel_initializer="he_normal", use_bias=False, 
#                           kernel_regularizer=l2(0.0005),
#                           activation="softmax")(flatten_1)
#     predictions_a = Dense(units=101, kernel_initializer="he_normal", use_bias=False,
#                           kernel_regularizer=l2(0.0005), 
#                           activation="softmax")(flatten_1)
#     predictions_e = Dense(units=5, kernel_initializer="he_normal", use_bias=False, 
#                           kernel_regularizer=l2(0.0005), 
#                           activation="softmax")(flatten_1)
#     model = Model(inputs = activation_9, outputs=[predictions_g, predictions_a, predictions_e])
    
    flatten_1 = Input(shape=(131072,))
#     predictions_g = Dense(units=2, kernel_initializer="he_normal", use_bias=False,
#                           kernel_regularizer=l2(0.0005), 
#                           activation="softmax")(flatten_1)
#     predictions_a = Dense(units=8, kernel_initializer="he_normal", use_bias=False,
#                           kernel_regularizer=l2(0.0005), 
#                           activation="softmax")(flatten_1)
    predictions_e = Dense(units=5, kernel_initializer="he_normal", use_bias=False, 
                          kernel_regularizer=l2(0.0005), 
                          activation="softmax")(flatten_1)
    model = Model(inputs = flatten_1, outputs=[predictions_e])
    return model