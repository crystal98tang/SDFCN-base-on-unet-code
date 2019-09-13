# tensorflow.python.keras.
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
# import pydensecrf.densecrf as dcrf

def SDFCN(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)
    conv1_1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1_2 = Conv2D(64, 3, activation='relu', padding='same')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    sc2 = shortcutblock(128)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(sc2)

    sc3 = shortcutblock(256)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(sc3)

    sc4 = shortcutblock(512)(pool3)
    sc4 = shortcutblock(512)(sc4)

    conv5_1 = Conv2DTranspose(256, 3, strides=2, padding='same')(Concatenate()([sc4, pool3]))
    conv5_2 = shortcutblock(256)(conv5_1)

    conv6_1 = Conv2DTranspose(128, 3, strides=2, padding='same')(Concatenate()([conv5_2, pool2]))
    conv6_2 = shortcutblock(128)(conv6_1)

    conv7_1 = Conv2DTranspose(64, 3, strides=2, padding='same')(Concatenate()([conv6_2, pool1]))
    conv7_2 = Conv2D(64, 3, activation='relu', padding='same')(conv7_1)
    conv7_3 = Conv2D(64, 3, activation='relu', padding='same')(conv7_2)

    # Softmax模式
    # conv8_0 = Conv2D(2, 1, strides=1, padding='same')(conv7_3)
    # conv8 = Activation('softmax')(conv8_0)
    # model = Model(inputs=inputs, outputs=conv8)

    # Sigmoid模式
    conv8 = Conv2D(2, 1, strides=1, padding='same')(conv7_3)
    activation = Activation('softmax')(conv8)

    #CRF
    # dcrf.DenseCRF2D(256,256,2)

    model = Model(inputs=inputs,outputs=activation)

    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # categorical_crossentropy binary_crossentropy
    if (pretrained_weights):
        # one gpu
        model.load_weights(pretrained_weights)

        # gpus--
        model.set_model(pretrained_weights)


    return model

"""
    Old SNFCN
"""
# def SNFCN(pretrained_weights=None, input_size=(256, 256, 1)):
#     inputs = Input(input_size)
#     conv1_1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
#     conv1_2 = Conv2D(64, 3, activation='relu', padding='same')(conv1_1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
#
#     sc2 = shortcutblock(128)(pool1)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(sc2)
#
#     sc3 = shortcutblock(256)(pool2)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(sc3)
#
#     sc4 = shortcutblock(512)(pool3)
#     sc4 = shortcutblock(512)(sc4)
#
#     conv5_1 = Conv2DTranspose(256, 3, strides=2, use_bias=False, padding='same')(sc4)
#     conv5_2 = shortcutblock(256)(conv5_1)
#
#     conv6_1 = Conv2DTranspose(128, 3, strides=2, use_bias=False, padding='same')(conv5_2)
#     conv6_2 = shortcutblock(128)(conv6_1)
#     conv7_1 = Conv2DTranspose(64, 3, strides=2, use_bias=False, padding='same')(conv6_2)
#     conv7_2 = Conv2D(64, 3, activation='relu', padding='same')(conv7_1)
#     conv7_3 = Conv2D(64, 3, activation='relu', padding='same')(conv7_2)
#
#     conv8 = Conv2D(1, 1, activation='softmax')(conv7_3)
#
#     model = Model(inputs=inputs, outputs=conv8)
#
#     model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#
#     if (pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     return model

"""
    ShortCutBlock Layer
"""
def shortcutblock(filter):
    def _create_shortcut_block(inputs):
        conv_main = Conv2D(filter, 1, padding='same')(inputs)
        conv_main = Conv2D(filter, 3, padding='same')(conv_main)
        conv_main = Conv2D(filter, 1, padding='same')(conv_main)
        conv_main = BatchNormalization()(conv_main)

        conv_fine = Conv2D(filter, 1, padding='same')(inputs)
        conv_fine = BatchNormalization()(conv_fine)

        merge = Add()([conv_main, conv_fine])

        conv = Activation('relu')(merge)

        return conv
    return _create_shortcut_block

