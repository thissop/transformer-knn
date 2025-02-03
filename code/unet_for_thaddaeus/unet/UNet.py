from keras import models, layers, regularizers, mixed_precision
import keras as K

def UNet_v2(input_shape, input_label_channel_count:int=1, layer_count=64, regularizers=regularizers.l2(0.0001), weights_file=None, summary=False, countbranch=False):
    """ Method to declare the UNet model.
    Args:
        input_shape: tuple(int, int, int, int)
            Shape of the input in the format (batch, height, width, channels).
        input_label_channel_count: int
            index count of label channels, used for calculating the number of channels in model output.
        layer_count: (int, optional)
            Count of kernels in first layer. Number of kernels in other layers grows with a fixed factor.
        regularizers: keras.regularizers
            regularizers to use in each layer.
        weight_file: str
            path to the weight file.
        summary: bool
            Whether to print the model summary
    """

    #NOTE(Jesse): Use "mixed_bfloat16" string for TPUs
    mixed_precision.set_global_policy("mixed_bfloat16")

    input_img = layers.Input(input_shape, dtype="float32", name='Input')
    pp_in_layer = input_img

    c1 = layers.Conv2D(1 * layer_count, (3, 3), activation='relu', padding='same')(pp_in_layer)
    c1 = layers.Conv2D(1 * layer_count, (3, 3), activation='relu', padding='same')(c1)
    n1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D()(n1)

    c2 = layers.Conv2D(2 * layer_count, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(2 * layer_count, (3, 3), activation='relu', padding='same')(c2)
    n2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D()(n2)

    c3 = layers.Conv2D(4 * layer_count, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(4 * layer_count, (3, 3), activation='relu', padding='same')(c3)
    n3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D()(n3)

    c4 = layers.Conv2D(8 * layer_count, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(8 * layer_count, (3, 3), activation='relu', padding='same')(c4)
    n4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D()(n4)

    c5 = layers.Conv2D(16 * layer_count, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(16 * layer_count, (3, 3), activation='relu', padding='same')(c5)
    n5 = layers.BatchNormalization()(c5)

    u6 = attention_up_and_concat(n5, n4)
    c6 = layers.Conv2D(8 * layer_count, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(8 * layer_count, (3, 3), activation='relu', padding='same')(c6)
    n6 = layers.BatchNormalization()(c6)

    u7 = attention_up_and_concat(n6, n3)
    c7 = layers.Conv2D(4 * layer_count, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(4 * layer_count, (3, 3), activation='relu', padding='same')(c7)
    n7 = layers.BatchNormalization()(c7)

    u8 = attention_up_and_concat(n7, n2)
    c8 = layers.Conv2D(2 * layer_count, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(2 * layer_count, (3, 3), activation='relu', padding='same')(c8)
    n8 = layers.BatchNormalization()(c8)

    u9 = attention_up_and_concat(n8, n1)
    c9 = layers.Conv2D(1 * layer_count, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(1 * layer_count, (3, 3), activation='relu', padding='same')(c9)
    n9 = layers.BatchNormalization()(c9)

# =============================================================================
    # density map
    if countbranch:
        d = layers.Conv2D(input_label_channel_count, (1, 1), activation='sigmoid', dtype="float32", kernel_regularizer=regularizers, name = 'output_seg')(n9)

        d2 = layers.Conv2D(input_label_channel_count, (1, 1), activation='linear', dtype='float32', kernel_regularizer= regularizers, name = 'output_dens')(n9)

        seg_model = models.Model(inputs=[input_img], outputs=[d, d2])
# =============================================================================
    else:
        d = layers.Conv2D(input_label_channel_count, (1, 1), activation='sigmoid', dtype='float32', kernel_regularizer=regularizers)(n9)
        seg_model = models.Model(inputs=[input_img], outputs=[d])

    seg_model.build(input_shape)
    if weights_file:
        seg_model.load_weights(weights_file)
    if summary:
        seg_model.summary()

    return seg_model


def attention_up_and_concat(down_layer, layer):
    in_channel = down_layer.shape[3]
    up = layers.UpSampling2D(interpolation="bilinear")(down_layer)
    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4)
    my_concat = layers.Lambda(lambda x: layers.concatenate([x[0], x[1]], axis=3))
    concat = my_concat([up, layer])

    return concat


def attention_block_2d(x, g, inter_channel):
    theta_x = layers.Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
    phi_g = layers.Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
    psi_f = layers.Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = layers.Activation('sigmoid')(psi_f)
    att_x = layers.multiply([x, rate])

    return att_x

def UNet_v1(input_shape, input_label_channel_count:int=1, layer_count=64, regularizers=regularizers.l2(0.0001), weights_file=None):
        """ Method to declare the UNet model.

        Args:
            input_shape: tuple(int, int, int, int)
                Shape of the input in the format (batch, height, width, channels).
            input_label_channel: list([int])
                list of index of label channels, used for calculating the number of channels in model output.
            layer_count: (int, optional)
                Count of kernels in first layer. Number of kernels in other layers grows with a fixed factor.
            regularizers: keras.regularizers
                regularizers to use in each layer.
            weight_file: str
                path to the weight file.
        """

        mixed_precision.set_global_policy("mixed_float16")

        input_img = layers.Input(input_shape, name='Input', dtype="float32")
        pp_in_layer  = input_img
        #pp_in_layer = layers.BatchNormalization()(pp_in_layer)

        c1 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(pp_in_layer)
        c1 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(c1)
        n1 = layers.BatchNormalization()(c1)
        p1 = layers.MaxPooling2D()(n1)

        c2 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(c2)
        n2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D()(n2)

        c3 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(c3)
        n3 = layers.BatchNormalization()(c3)
        p3 = layers.MaxPooling2D()(n3)

        c4 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(c4)
        n4 = layers.BatchNormalization()(c4)
        p4 = layers.MaxPooling2D()(n4)

        c5 = layers.Conv2D(16*layer_count, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(16*layer_count, (3, 3), activation='relu', padding='same')(c5)

        u6 = layers.UpSampling2D()(c5)
        n6 = layers.BatchNormalization()(u6)
        u6 = layers.concatenate([n6, n4])
        c6 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(c6)

        u7 = layers.UpSampling2D()(c6)
        n7 = layers.BatchNormalization()(u7)
        u7 = layers.concatenate([n7, n3])
        c7 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(c7)

        u8 = layers.UpSampling2D()(c7)
        n8 = layers.BatchNormalization()(u8)
        u8 = layers.concatenate([n8, n2])
        c8 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(c8)

        u9 = layers.UpSampling2D()(c8)
        n9 = layers.BatchNormalization()(u9)
        u9 = layers.concatenate([n9, n1], axis=3)
        c9 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(c9)

        d = layers.Conv2D(input_label_channel_count, (1, 1), activation='sigmoid', kernel_regularizer= regularizers, dtype="float32")(c9)

        seg_model = models.Model(inputs=[input_img], outputs=[d])
        seg_model.build(input_shape)
        if weights_file:
            seg_model.load_weights(weights_file)
        #seg_model.summary()
        return seg_model

