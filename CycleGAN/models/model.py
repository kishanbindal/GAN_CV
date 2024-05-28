import tensorflow as tf
import tensorflow_addons as tfa
from models.layers import ReflectionPad2D

def resnet_block(input_tensor,
              n_filters, 
              kernel_size=(3,3),
              padding="valid",
              strides=(1,1),
              use_dropout=False):
    x = ReflectionPad2D()(input_tensor)
    x = tf.keras.layers.Conv2D(n_filters, 
                               kernel_size = kernel_size, 
                               strides = strides, 
                               padding=padding)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    if use_dropout:
        x = tf.keras.layers.Dropout(0.5)(x)

    x = ReflectionPad2D()(x)
    x = tf.keras.layers.Conv2D(
        n_filters, 
        padding=padding,
        kernel_size=kernel_size,
        strides=strides,
    )(x)
    x = tfa.layers.InstanceNormalization()(x)
    output = tf.keras.layers.add([x, input_tensor])
    output = tf.keras.layers.ReLU()(output)
    return output

def create_generator(model_name:str, n_res_blks: int):

    if int(n_res_blks) == 6:
        input = tf.keras.layers.Input(shape=(128, 128, 3))
    elif int(n_res_blks) == 9:
        input = tf.keras.layers.Input(shape=(256, 256, 3))
    else: 
        raise ValueError("Value for n_res_blks can be either 6 or 9 only!")

    # Encoder Section - c7s1-64 , d128, d256 (c-> normal conv, d -> downsample conv)
    x = ReflectionPad2D((3, 3))(input)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=1, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Downsampling within encoder
    # d-128
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # d-256
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides = 2, padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Resnet Transformer Section
    for i in range(n_res_blks):
        x = resnet_block(x, 256)

    # Upsampling with Decoder section of Generator
    x = tf.keras.layers.Conv2DTranspose(128, 3, 2, padding="same", use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, 2, padding="same", use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Resetting to three channel
    x = ReflectionPad2D(3)(x)
    x = tf.keras.layers.Conv2D(3, 7, 1, padding="valid")(x)
    output = tf.keras.activations.tanh(x)

    model = tf.keras.Model(input, output, name=model_name)
    return model

def create_discriminator(
        img_shape,
        kernel_size = (4, 4),
        strides=(2,2),
        padding="same"
):
    """
    The Discriminator follows the architecture for a 70x70(receptive field) patchGAN. 
    architecture - c64-c128-c256-c512
    all Relus are Leaky, with slope = 0.2
    all c layers have k 4x4 filters and stride=2
    c512 has stride of 1.
    c512 followed by Convolution and sigmoid activation

    C64 Does is not accompanied by Normalization Layer. 
    nstanceNormalization is used as normalization layer
    """
    base_filters = 64
    # c64-4x4-s2
    input = tf.keras.layers.Input(shape=img_shape)
    x = tf.keras.layers.Conv2D(base_filters,
                               kernel_size= kernel_size,
                               strides=strides,
                               padding=padding)(input)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # c128-4-s2, c256-4-s2, c512-4-s2
    for _ in range(3):
        base_filters *= 2
        if _ == 2:
            strides = (1, 1)
        x = tf.keras.layers.Conv2D(base_filters, 
                                   kernel_size=kernel_size, 
                                   strides=strides,
                                   padding=padding)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
    output =  tf.keras.layers.Conv2D(1, 
                               kernel_size=kernel_size, 
                               strides=strides,
                               padding=padding)(x)
    model = tf.keras.Model(input, output)
    return model