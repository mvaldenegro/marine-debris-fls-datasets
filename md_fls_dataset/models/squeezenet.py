import keras
from keras.layers import Conv2D, Concatenate, BatchNormalization, Activation
from keras.layers import Dense, GlobalAveragePooling2D, Input, MaxPooling2D
from keras.models import Model

def fire_module(inp, s11, e11, e33, activation="relu", name=""):
    squeeze = Conv2D(s11, (1, 1), padding = "same", activation = activation)(inp)
    expand1x1 = Conv2D(e11, (1, 1), padding = "same", activation = activation)(squeeze)
    expand3x3 = Conv2D(e33, (3, 3), padding = "same", activation = activation)(squeeze)

    merged = Concatenate(axis=-1)([expand1x1, expand3x3])
    return BatchNormalization()(merged)

def squeezenet(input_shape=(96,96,1), num_classes=18, base_squ_filt=16, base_exp_filt=64):
    inp = Input(shape = input_shape)

    x = Conv2D(8, (7, 7),  strides = (2, 2), padding="same", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    squeeze_filt = base_squ_filt
    expand_filt  = base_exp_filt

    x = fire_module(x, s11 = int(1.0 * squeeze_filt), e11 = int(1.0 * expand_filt), e33 = int(1.0 * expand_filt))
    x = fire_module(x, s11 = int(1.0 * squeeze_filt), e11 = int(1.0 * expand_filt), e33 = int(1.0 * expand_filt))
    x = fire_module(x, s11 = int(2.0 * squeeze_filt), e11 = int(2.0 * expand_filt), e33 = int(2.0 * expand_filt))

    x = MaxPooling2D((2, 2))(x)

    squeeze_filt *= 2
    expand_filt  *= 2

    x = fire_module(x, s11 = int(1.0 * squeeze_filt), e11 = int(1.0 * expand_filt), e33 = int(1.0 * expand_filt))
    x = fire_module(x, s11 = int(1.5 * squeeze_filt), e11 = int(1.5 * expand_filt), e33 = int(1.5 * expand_filt))
    x = fire_module(x, s11 = int(1.5 * squeeze_filt), e11 = int(1.5 * expand_filt), e33 = int(1.5 * expand_filt))
    x = fire_module(x, s11 = int(2.0 * squeeze_filt), e11 = int(2.0 * expand_filt), e33 = int(2.0 * expand_filt))
    
    x = MaxPooling2D((2, 2))(x)

    squeeze_filt *= 2
    expand_filt  *= 2

    x = fire_module(x, s11 = int(1.0 * squeeze_filt), e11 = int(1.0 * expand_filt), e33 = int(1.0 * expand_filt))

    x = Conv2D(num_classes, (5, 5), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    out = Activation("softmax")(x)

    model = Model(inputs = inp, outputs = out)

    return model