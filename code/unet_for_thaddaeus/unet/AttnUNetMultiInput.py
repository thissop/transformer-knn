# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:54:57 2018

@author: Nabila Abraham
@Edited by Jesse Meyer, NASA
"""

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, add, multiply, Lambda, AveragePooling2D, UpSampling2D, BatchNormalization, SpatialDropout2D

import keras as K
from keras import mixed_precision

kinit = "glorot_normal"#"glorot_uniform"

def AttnGatingBlock(x, g, inter_shape, name):
    ''' take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same feature channels (theta_x)
    then, upsample g to be same size as x 
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''
    
    shape_x = x.shape#K.int_shape(x)  # 32
    shape_g = g.shape#K.int_shape(g)  # 16

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name=name + '_xl')(x)  # 16
    shape_theta_x = theta_x.shape# K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same', name=name + "_phi_g")(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same', name=name+'_g_up')(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same', name=name+'_psi')(act_xg)
    sigmoid_xg = Activation('sigmoid', name=name + "_sig_xg")(psi)

    shape_sigmoid = sigmoid_xg.shape#K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]), interpolation="bilinear")(sigmoid_xg)  # 32
    upsample_psi = Lambda(lambda x, repnum: K.ops.repeat(x, repnum, axis=3), arguments={'repnum': shape_x[3]}, name=name+'_psi_up')(upsample_psi)
    
    y = multiply([upsample_psi, x], name=name+'_q_attn')

    result = Conv2D(shape_x[3], (1, 1), padding='same',name=name+'_q_attn_conv')(y)
    result_bn = BatchNormalization(name=name+'_q_attn_bn')(result)
    return result_bn

def UnetConv2D(input, outdim, name):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", activation='relu', name=name+'_1_conv')(input)
    #x = Activation('relu',name=name + '_1_act')(x)
    #x = BatchNormalization(name=name + '_1_bn')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", activation='relu', name=name+'_2_conv')(x)
    #x = Activation('relu', name=name + '_2_act')(x)
    x = BatchNormalization(name=name + '_2_bn')(x)

    return x
	

def UnetGatingSignal(input, name):
    ''' this is simply 1x1 convolution, bn, activation '''
    shape = input.shape#K.int_shape(input)
    x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same", kernel_initializer=kinit, activation='relu', name=name + '_gate_conv')(input)
    #x = Activation('relu', name=name + '_gate_act')(x)
    x = BatchNormalization(name=name + '_gate_bn')(x)
    
    return x


#model proposed in my paper - improved attention u-net with multi-scale input pyramid and deep supervision

def attn_reg_train(input_shape, layer_count=64, weights_file=None):
    mixed_precision.set_global_policy("mixed_bfloat16")
    
    img_input = Input(shape=input_shape, name='input_scale1', dtype='float32')
    scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)
    scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)
    scale_img_4 = AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)

    conv1 = UnetConv2D(img_input, 1 * layer_count, name='conv1')
    pool1 = MaxPooling2D(name='mp1')(conv1)
    pool1 = SpatialDropout2D(0.3, name='1_sdo2d')(pool1)
    
    input2 = Conv2D(2 * layer_count, (3, 3), padding='same', activation='relu', name='conv_scale2')(scale_img_2)
    input2 = concatenate([input2, pool1], axis=3)
    conv2 = UnetConv2D(input2, 2 * layer_count, name='conv2')
    pool2 = MaxPooling2D(name='mp2')(conv2)
    pool2 = SpatialDropout2D(0.3, name='2_sdo2d')(pool2)
    
    input3 = Conv2D(4 * layer_count, (3, 3), padding='same', activation='relu', name='conv_scale3')(scale_img_3)
    input3 = concatenate([input3, pool2], axis=3)
    conv3 = UnetConv2D(input3, 4 * layer_count, name='conv3')
    pool3 = MaxPooling2D(name='mp3')(conv3)
    pool3 = SpatialDropout2D(0.3, name='3_sdo2d')(pool3)
    
    input4 = Conv2D(8 * layer_count, (3, 3), padding='same', activation='relu', name='conv_scale4')(scale_img_4)
    input4 = concatenate([input4, pool3], axis=3)
    conv4 = UnetConv2D(input4, 2 * layer_count, name='conv4')
    pool4 = MaxPooling2D(name='mp4')(conv4)
    pool4 = SpatialDropout2D(0.3, name='4_sdo2d')(pool4)
        
    center = UnetConv2D(pool4, 16 * layer_count, name='center')
    
    g1 = UnetGatingSignal(center,  name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 4 * layer_count, 'attn_1')
    up1 = concatenate([Conv2DTranspose(1 * layer_count, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1')

    g2 = UnetGatingSignal(up1,  name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 2 * layer_count, 'attn_2')
    up2 = concatenate([Conv2DTranspose(2 * layer_count, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2')

    g3 = UnetGatingSignal(up1,  name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 1 * layer_count, 'attn_3')
    up3 = concatenate([Conv2DTranspose(1 * layer_count, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3')
    up4 = concatenate([Conv2DTranspose(1 * layer_count, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4')
    
    conv6 = UnetConv2D(up1, 8 * layer_count, name='conv6')
    conv7 = UnetConv2D(up2, 4 * layer_count, name='conv7')
    conv8 = UnetConv2D(up3, 2 * layer_count, name='conv8')
    conv9 = UnetConv2D(up4, 1 * layer_count, name='conv9')

    out6 = Conv2D(1, (1, 1), activation='sigmoid', name='pred1', dtype='float32')(conv6)
    out7 = Conv2D(1, (1, 1), activation='sigmoid', name='pred2', dtype='float32')(conv7)
    out8 = Conv2D(1, (1, 1), activation='sigmoid', name='pred3', dtype='float32')(conv8)
    out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final', dtype='float32')(conv9)

    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])
    model.build(input_shape)
    if weights_file:
        model.load_weights(weights_file)
 
    return model

def attn_reg(input_shape, layer_count=64, weights_file=None):
    mixed_precision.set_global_policy("mixed_bfloat16")
    
    img_input = Input(shape=input_shape, name='input_scale1', dtype='float32')
    scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)
    scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)
    scale_img_4 = AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)

    conv1 = UnetConv2D(img_input, 1 * layer_count,  name='conv1')
    pool1 = MaxPooling2D(name='mp1')(conv1)
    #pool1 = SpatialDropout2D(0.3, name='1_sdo2d')(pool1)
    
    input2 = Conv2D(2 * layer_count, (3, 3), padding='same', activation='relu', name='conv_scale2')(scale_img_2)
    input2 = concatenate([input2, pool1], axis=3)
    conv2 = UnetConv2D(input2, 2 * layer_count,  name='conv2')
    pool2 = MaxPooling2D(name='mp2')(conv2)
    #pool2 = SpatialDropout2D(0.3, name='2_sdo2d')(pool2)
    
    input3 = Conv2D(4 * layer_count, (3, 3), padding='same', activation='relu', name='conv_scale3')(scale_img_3)
    input3 = concatenate([input3, pool2], axis=3)
    conv3 = UnetConv2D(input3, 4 * layer_count,  name='conv3')
    pool3 = MaxPooling2D(name='mp3')(conv3)
    #pool3 = SpatialDropout2D(0.3, name='3_sdo2d')(pool3)
    
    input4 = Conv2D(8 * layer_count, (3, 3), padding='same', activation='relu', name='conv_scale4')(scale_img_4)
    input4 = concatenate([input4, pool3], axis=3)
    conv4 = UnetConv2D(input4, 2 * layer_count,  name='conv4')
    pool4 = MaxPooling2D(name='mp4')(conv4)
    #pool4 = SpatialDropout2D(0.3, name='4_sdo2d')(pool4)
        
    center = UnetConv2D(pool4, 16 * layer_count,  name='center')
    
    g1 = UnetGatingSignal(center, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 4 * layer_count, 'attn_1')
    up1 = concatenate([Conv2DTranspose(1 * layer_count, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1')

    g2 = UnetGatingSignal(up1, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 2 * layer_count, 'attn_2')
    up2 = concatenate([Conv2DTranspose(2 * layer_count, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2')

    g3 = UnetGatingSignal(up1, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 1 * layer_count, 'attn_3')
    up3 = concatenate([Conv2DTranspose(1 * layer_count, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3')
    up4 = concatenate([Conv2DTranspose(1 * layer_count, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4')

    #NOTE(Jesse): Removed deeply supervised output layers as they do not contribute to to out9
    
    conv9 = UnetConv2D(up4, 1 * layer_count,  name='conv9')
    out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final', dtype='float32')(conv9)

    model = Model(inputs=[img_input], outputs=[out9])
    model.build(input_shape)
    if weights_file:
        model.load_weights(weights_file)
 
    return model

def peel_trained_model(src_model, input_shape):
    #NOTE(Jesse): The attn_reg model for training has extra outputs and weights that are not necessary for inferrence and incur a substantial performance cost.
    # So we "peel" them here.
    #
    # Also, TF / Keras globally namespace layer names, so if two identical models are created, they _do not_ share the same layer names
    # if layer names are not explicit provided.  This is stupid and causes all this dumb code to exist for no reason.
    # These models have over a hundred layers and they continue to grow so I don't think it's reasonable to solve it on a per layer basis
    
    blacklisted_lyrs = ("pred1", "pred2", "pred3", "conv6", "conv7", "conv8")
    dst_model = attn_reg(input_shape)
    base_idx = 0
    for lyr_idx, dst_lyr in enumerate(dst_model.layers):
        while True:
            src_lyr = src_model.get_layer(index=lyr_idx + base_idx)
            for bl_lyrn in blacklisted_lyrs:
                if src_lyr.name.startswith(bl_lyrn):
                    base_idx += 1
                    break
            else:
                break

        src_lyr_wghts = src_lyr.get_weights()
        if len(src_lyr_wghts) == 0:
            continue

        assert src_lyr.output.shape == dst_lyr.output.shape
        
        dst_lyr.set_weights(src_lyr_wghts)

    return dst_model
