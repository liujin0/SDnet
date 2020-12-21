#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from layers.ResidualMultiplicativeBlock import ResidualMultiplicativeBlock as rmb
from layers.VelocityCascadeMultiplicativeUnit import VelocityCascadeMultiplicativeUnit as vcmu

def static_dynamic_net(images, params, num_hidden, filter_size, seq_length=20, input_length=10):
    encoder_length = params['encoder_length']
    decoder_length = params['decoder_length']
    num_hidden = num_hidden[0]
    channels = images.shape[-1]

    with tf.variable_scope('sd_net'):
        # encoder
        encoder_output = []
        for i in range(input_length):
            reuse = bool(encoder_output)
            ims = images[:,i]
            input = cnn_encoders(ims, num_hidden, filter_size, encoder_length, reuse)
            encoder_output.append(input)

        # sd_net & decoder
        summarized_info = hierarchical_vcmu(encoder_output, num_hidden, filter_size, input_length, reuse=False)

        output = []
        for i in range(seq_length - input_length):
            out = cnn_decoders(summarized_info, num_hidden, filter_size, channels, decoder_length, reuse=False, name=str(i))
            output.append(out)       

    # transpose output and compute loss
    gen_images = tf.stack(output)
    # [batch_size, seq_length, height, width, channels]
    gen_images = tf.transpose(gen_images, [1, 0, 2, 3, 4])

    pred_v = gen_images

    # last observed frame
    f9 = images[:, (input_length-1):input_length]  

    # predicted sequence
    gen_images = pred_v + f9

    # ground truth sequence
    gt_images = images[:, input_length:]
    gt_v = gt_images - f9

    loss = tf.norm((gen_images-gt_images), ord=1) + 0.01 * tf.norm((pred_v-gt_v), ord=1)
    return [gen_images, loss]


def cnn_encoders(x, num_hidden, filter_size, encoder_length, reuse):
    with tf.variable_scope('resolution_preserving_cnn_encoders', reuse=reuse):
        x = tf.layers.conv2d(x, num_hidden, filter_size, padding='same', activation=None,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='input_conv')
        for i in range(encoder_length):
            x = rmb('residual_multiplicative_block_'+str(i+1), num_hidden // 2, filter_size)(x)
        return x

def hierarchical_vcmu(xs, num_hidden, filter_size, input_length, reuse, name='frame_prediction'):
    with tf.variable_scope(name, reuse=reuse):
        h11 = vcmu('causal_multiplicative_unit_'+str(1), num_hidden, filter_size)(xs[0], xs[1], stride=False, reuse=False)
        h12 = vcmu('causal_multiplicative_unit_'+str(2), num_hidden, filter_size)(xs[2], xs[3], stride=False, reuse=False)
        h13 = vcmu('causal_multiplicative_unit_'+str(3), num_hidden, filter_size)(xs[4], xs[5], stride=False, reuse=False)
        h14 = vcmu('causal_multiplicative_unit_'+str(4), num_hidden, filter_size)(xs[6], xs[7], stride=False, reuse=False)
        h15 = vcmu('causal_multiplicative_unit_'+str(5), num_hidden, filter_size)(xs[8], xs[9], stride=False, reuse=False)

        h21 = vcmu('causal_multiplicative_unit_'+str(6), num_hidden, filter_size)(h11, h12, stride=False, reuse=False)
        h31 = vcmu('causal_multiplicative_unit_'+str(7), num_hidden, filter_size)(h21, h13, stride=False, reuse=False)
        h41 = vcmu('causal_multiplicative_unit_'+str(8), num_hidden, filter_size)(h31, h14, stride=False, reuse=False)
        h = vcmu('causal_multiplicative_unit_'+str(9), num_hidden, filter_size)(h41, h15, stride=False, reuse=False)
        return h

def cnn_decoders(x, num_hidden, filter_size, output_channels, decoder_length, reuse, name):
    with tf.variable_scope('cnn_decoders'+name, reuse=reuse):
        for i in range(decoder_length):
            x = rmb('residual_multiplicative_block_'+str(i+1), num_hidden // 2, filter_size)(x)
        x = tf.layers.conv2d(x, output_channels, filter_size, padding='same',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='output_conv')
        return x
