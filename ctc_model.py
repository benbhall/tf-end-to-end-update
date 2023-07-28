import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def leaky_relu(features, alpha=0.2, name=None):
  with ops.name_scope(name, "LeakyRelu", [features, alpha]):
    features = ops.convert_to_tensor(features, name="features")
    alpha = ops.convert_to_tensor(alpha, name="alpha")
    return math_ops.maximum(alpha * features, features)



#
# params["height"] = height of the input image
# params["width"] = width of the input image

def default_model_params(img_height, vocabulary_size):
    params = dict()
    params['img_height'] = img_height
    params['img_width'] = None
    params['batch_size'] = 16
    params['img_channels'] = 1
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [32, 64, 128, 256]
    params['conv_filter_size'] = [ [3,3], [3,3], [3,3], [3,3] ]
    params['conv_pooling_size'] = [ [2,2], [2,2], [2,2], [2,2] ]
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    params['vocabulary_size'] = vocabulary_size
    return params

def ctc_crnn(params):
    # TODO Assert parameters

    input = tf.keras.Input(shape=(params['img_height'],
                                  params['img_width'],
                                  params['img_channels']),  # [batch, height, width, channels]
                           dtype=tf.float32,
                           name='model_input')

    # Convolutional blocks
    x = input
    for i in range(params['conv_blocks']):
        x = tf.keras.layers.Conv2D(
            filters=params['conv_filter_n'][i],
            kernel_size=params['conv_filter_size'][i],
            padding="same")(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=params['conv_pooling_size'][i])(x)

    # Swap the time and feature dimensions
    x = tf.keras.layers.Permute((2, 1, 3))(x)

    # Flatten the feature dimension
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    # Recurrent block
    for _ in range(params['rnn_layers']):
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True, dropout=0.5))(x)

    # Output layer
    output = tf.keras.layers.Dense(params['vocabulary_size'] + 1, activation='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model
