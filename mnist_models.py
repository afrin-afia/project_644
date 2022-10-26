from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

def modelA():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='valid', input_shape=(28, 28, 1)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(10))
    return model

def modelB():
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(28,
                                        28,
                                        1)))
    model.add(Conv2D(64, (8, 8),
                            strides=(2, 2),
                            padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (6, 6),
                            strides=(2, 2),
                            padding='valid'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (5, 5),
                            strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(10))
    return model


def modelC():
    model = Sequential()
    model.add(Conv2D(128, (3, 3),
                            padding='valid',
                            input_shape=(28,
                                         28,
                                         1)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(10))
    return model


def modelD():
    model = Sequential()

    model.add(Flatten(input_shape=(28,
                                   28,
                                   1)))

    model.add(Dense(300, kernel_initializer ='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, kernel_initializer ='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, kernel_initializer ='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, kernel_initializer ='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    return model

def modelE():
    model = Sequential()

    model.add(Flatten(input_shape=(28,
                                   28,
                                   1)))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))

    model.add(Dense(10))

    return model

def modelF():
    model = Sequential()

    model.add(Conv2D(32, (5, 5),
                            padding='valid',
                            input_shape=(28,
                                         28,
                                         1)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(10))

    return model

def modelG():
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(512, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10))

    return model

def model_LR():
    model = Sequential()

    model.add(Flatten(input_shape=(28,
                                   28,
                                   1)))

    model.add(Dense(10))

    return model


model = modelG()
print(model.summary())  