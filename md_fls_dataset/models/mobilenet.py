#Adapted from https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py

import keras
from keras import layers, backend, models

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.ReLU(6., name='conv1_relu')(x)

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):    
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

def mobilenet(input_shape,
              num_classes,
              alpha=1.0,
              depth_multiplier=1,
              base_filters=8):

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    num_filters = base_filters

    img_input = layers.Input(shape=input_shape)

    # Block with 32 filters
    x = _conv_block(img_input, num_filters, alpha, strides=(2, 2))

    # Block with 64 filters
    x = _depthwise_conv_block(x, 2 * num_filters, alpha, depth_multiplier, block_id=1)

    num_filters *= 4

    # Block with 128 filters
    x = _depthwise_conv_block(x, num_filters, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, num_filters, alpha, depth_multiplier, block_id=3)

    num_filters *= 2

    # Block with 256 filters
    x = _depthwise_conv_block(x, num_filters, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, num_filters, alpha, depth_multiplier, block_id=5)

    num_filters *= 2

    # Block with 512 filters
    x = _depthwise_conv_block(x, num_filters, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, num_filters, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, num_filters, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, num_filters, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, num_filters, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, num_filters, alpha, depth_multiplier, block_id=11)

    num_filters *= 2

    # Block with 1024 filters
    x = _depthwise_conv_block(x, num_filters, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, num_filters, alpha, depth_multiplier, block_id=13)

    # Add classifier
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes, activation = "softmax", name="classifier")(x)

    # Create model.
    model = models.Model(img_input, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

    return model