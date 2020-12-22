
##########################################################
# %%
# Import Library
##########################################################

import sys, os
import numpy as np
import tensorflow as tf

# keras
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Conv3D, MaxPooling3D, Activation, BatchNormalization, \
    Add, Conv3DTranspose, LeakyReLU, Lambda, Concatenate, ZeroPadding2D

from tensorflow.keras.models import Model

# import custom functions 
import library_common as mf

##########################################################
# %%
# my functions
##########################################################

def conv3D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', USE_BN=True, layer_name=''):
    with K.name_scope(layer_name):
        x = Conv3D(num_out_chan, kernel_size, activation=None, padding='same', kernel_initializer='truncated_normal')(x)
        if USE_BN:
            x = BatchNormalization()(x)
        if activation_type == 'LeakyReLU':
            return LeakyReLU()(x)
        else:
            return Activation(activation_type)(x)

##########################################################
# %%
# MoDL functions
##########################################################

c2r = Lambda(lambda x: tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1))
r2c = Lambda(lambda x: tf.complex(x[..., 0], x[..., 1]))


class Aclass:
    def __init__(self, csm, mask, wPSF, lam):
        with tf.name_scope('Ainit'):
            s = tf.shape(mask)
            self.nrow, self.ncol, self.nslc = s[0], s[1], s[2]
            self.pixels = self.nrow * self.ncol * self.nslc
            self.mask = mask
            self.ncoil = tf.shape(csm)[0]
            self.csm = csm
            self.wPSF = wPSF
            self.SF = tf.complex(tf.sqrt(tf.cast(self.pixels, dtype=tf.float32)), 0.)
            self.lam = lam
            self.zpf = 3
            self.ind_zs = tf.cast((self.zpf - 1) * self.nrow / self.zpf / 2, tf.int32)
            self.ind_ze = tf.cast((self.zpf + 1) * self.nrow / self.zpf / 2, tf.int32)
            self.half_zpad = tf.cast((self.zpf - 1) * self.nrow / self.zpf / 2, tf.int32)

    def myAtA(self, img):
        with tf.name_scope('AtA'):
            coilImages = self.csm * img
            coil_zpad = tf.pad(coilImages, [[0, 0], [self.half_zpad, self.half_zpad], [0, 0], [0, 0]])
            kspace = tf.signal.fft3d(coil_zpad) / self.SF
            wav_dat = tf.signal.fft2d(tf.expand_dims(self.wPSF, axis=0) * tf.signal.ifft2d(kspace))
            temp = wav_dat * tf.expand_dims(self.mask, axis=0)
            unwav_dat = tf.signal.fft2d(tf.math.conj(tf.expand_dims(self.wPSF, axis=0)) * tf.signal.ifft2d(temp))
            coilImgs = tf.signal.ifft3d(unwav_dat) * self.SF
            coil_unzpd = coilImgs[:,  self.ind_zs:self.ind_ze, ]
            coilComb = tf.reduce_sum(coil_unzpd * tf.math.conj(self.csm), axis=0)
            coilComb = coilComb + self.lam * img
        return coilComb


def myCG(A, rhs):
    rhs = r2c(rhs)
    cond = lambda i, rTr, *_: tf.logical_and(tf.less(i, 10), rTr > 1e-5)

    def body(i, rTr, x, r, p):
        with tf.name_scope('cgBody'):
            Ap = A.myAtA(p)
            alpha = rTr / tf.cast(tf.reduce_sum(tf.math.conj(p) * Ap), dtype=tf.float32)
            alpha = tf.complex(alpha, 0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = tf.cast(tf.reduce_sum(tf.math.conj(r) * r), dtype=tf.float32)
            beta = rTrNew / rTr
            beta = tf.complex(beta, 0.)
            p = r + beta * p
        return i + 1, rTrNew, x, r, p

    x = tf.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = tf.cast(tf.reduce_sum(tf.math.conj(r) * r), dtype=tf.float32)
    loopVar = i, rTr, x, r, p
    out = tf.while_loop(cond, body, loopVar, name='CGwhile', parallel_iterations=1)[2]
    return c2r(out)


class myDC(Layer):
    def __init__(self, **kwargs):
        super(myDC, self).__init__(**kwargs)

        self.lam1 = self.add_weight(name='lam1', shape=(1,), initializer=tf.constant_initializer(value=0.03),
                                    dtype='float32', trainable=True)

    def build(self, input_shape):
        super(myDC, self).build(input_shape)

    def call(self, x):
        # DC
        rhs, csm, mask, wPSF = x
        lam2 = tf.complex(self.lam1, 0.)

        def fn(tmp):
            c, m, w, r = tmp
            Aobj = Aclass(c, m, w, lam2)
            y = myCG(Aobj, r)
            return y

        inp = (csm, mask, wPSF, rhs)
        rec = tf.map_fn(fn, inp, dtype=tf.float32, name='mapFn')
        return rec

    def lam_weight(self, x):
        img = x[0]
        res = self.lam1 * img
        return res

    def lam_weight2(self, x):
        in0, in1 = x
        res = self.lam1 * (in0 + in1) / 2
        return res


def RegConvLayers(nx, ny, nz, nLayers, num_filters):
    input_x = Input(shape=(nx, ny, nz, 2), dtype=tf.float32)

    rg_term = input_x
    for lyr in range(0, nLayers):
        rg_term = conv3D_bn_nonlinear(rg_term, num_filters, (3, 3, 2), activation_type='relu', USE_BN=True,
                                      layer_name='')

    # go to image space
    rg_term = conv3D_bn_nonlinear(rg_term, 2, (1, 1, 1), activation_type=None, USE_BN=False, layer_name='')

    # skip connection
    rg_term = Add()([rg_term, input_x])

    return Model(inputs=input_x, outputs=rg_term)


##########################################################
# %%
# Create wave MoDL model
##########################################################


def create_wave_caipi(nx, ny, rz, nc, num_block=10, zpf=3):
    # define the inputs
    input_c = Input(shape=(nc, nx, ny, rz), dtype=tf.complex64)
    input_m = Input(shape=(zpf*nx, ny, rz), dtype=tf.complex64)
    input_w = Input(shape=(zpf*nx, ny, rz), dtype=tf.complex64)
    input_Atb = Input(shape=(nx, ny, rz), dtype=tf.complex64)

    dc_term = c2r(input_Atb)
    UpdateDC = myDC()

    for blk in range(0, num_block):
        rg_term = UpdateDC.lam_weight([dc_term])
        rg_term = Add()([c2r(input_Atb), rg_term])

        # Update DC
        dc_term = UpdateDC([rg_term, input_c, input_m, input_w])

    out_x = dc_term

    return Model(inputs=[input_c, input_m, input_w, input_Atb],
                 outputs=out_x)


def create_wave_modl(nx, ny, rz, nc, nLayers, num_block, num_filters=64, zpf = 3):
    # define the inputs
    input_c = Input(shape=(nc, nx, ny, rz), dtype=tf.complex64)
    input_m = Input(shape=(zpf*nx, ny, rz), dtype=tf.complex64)
    input_w = Input(shape=(zpf*nx, ny, rz), dtype=tf.complex64)
    input_Atb = Input(shape=(nx, ny, rz), dtype=tf.complex64)

    dc_term = c2r(input_Atb)
    RegConv_i = RegConvLayers(nx, ny, rz, nLayers, num_filters)
    RegConv_k = RegConvLayers(nx, ny, rz, nLayers, num_filters)
    UpdateDC = myDC()

    myFFT = mf.tf_fft3_r()
    myIFFT = mf.tf_ifft3_r()

    for blk in range(0, num_block):
        # CNN Regularization
        rg_term_i = RegConv_i(dc_term)
        rg_term_k = myIFFT([RegConv_k(myFFT([dc_term]))])
        rg_term = UpdateDC.lam_weight2([rg_term_i, rg_term_k])
        # AtA update                      
        rg_term = Add()([c2r(input_Atb), rg_term])

        # Update DC
        dc_term = UpdateDC([rg_term, input_c, input_m, input_w])

    out_x = dc_term

    return Model(inputs=[input_c, input_m, input_w, input_Atb],
                 outputs=out_x)


def create_wave_caipi_joint(nx, ny, rz, ns, nc, zpf=3, num_block=10, activation_type='relu', USE_BN=False, depth=4,
                             num_filters=64, init_Unet=False, Unet_name=''):
    # define the inputs
    input_c = Input(shape=(nc, nx, ny, rz), dtype=tf.complex64)
    input_m = Input(shape=(zpf*nx, ny, rz), dtype=tf.complex64)
    input_w = Input(shape=(zpf*nx, ny, rz), dtype=tf.complex64)
    input_Atb = Input(shape=(nx, ny, rz), dtype=tf.complex64)

    wave_caipi = create_wave_caipi(nx, ny, rz, nc, num_block, zpf)
    dc_term = wave_caipi([input_c, input_m, input_w, input_Atb])

    # create Unet for segmentation
    Unet = create_unet_seg(nx=nx, ny=ny, nc=nc, ns=ns, activation_type=activation_type,
                           USE_BN=USE_BN, depth=depth, num_filters=num_filters)

    if init_Unet:
        try:
            Unet.load_weights(Unet_name)
            print('initialize Unet')
        except:
            print('fail to initialize Unet')

    seg_class = K.expand_dims(Unet([dc_term[:, :, :, 0, :]]), axis=-2)
    for slc in range(1, rz):
        t = K.expand_dims(Unet([dc_term[:, :, :, slc, :]]), axis=-2)
        seg_class = Concatenate(axis=-2)([seg_class, t])

    return Model(inputs=[input_c, input_m, input_w, input_Atb],
                 outputs=[dc_term, seg_class])


def create_wave_modl_joint(nx, ny, rz, ns, nc, nLayers, zpf=3, num_block=10, activation_type='relu', USE_BN=False, depth=4,
                               num_filters=64, unet_filter=64, init_Unet=False, Unet_name='', init_modl=False,
                               modl_name=''):
    # define the inputs
    input_c = Input(shape=(nc, nx, ny, rz), dtype=tf.complex64)
    input_m = Input(shape=(zpf*nx, ny, rz), dtype=tf.complex64)
    input_w = Input(shape=(zpf*nx, ny, rz), dtype=tf.complex64)
    input_Atb = Input(shape=(nx, ny, rz), dtype=tf.complex64)

    wave_modl = create_wave_modl(nx, ny, rz, nc, nLayers, num_block, activation_type, USE_BN, num_filters, zpf)

    if init_modl:
        try:
            wave_modl.load_weights(modl_name)
            print('initialize modl')
        except:
            print('fail to initialize modl')

    dc_term = wave_modl([input_c, input_m, input_w, input_Atb])

    # create Unet for segmentation
    Unet = create_unet_seg(nx=nx, ny=ny, nc=nc, ns=ns, activation_type=activation_type,
                           USE_BN=USE_BN, depth=depth, num_filters=unet_filter)

    if init_Unet:
        try:
            Unet.load_weights(Unet_name)
            print('initialize Unet')
        except:
            print('fail to initialize Unet')

    seg_class = K.expand_dims(Unet([dc_term[:, :, :, 0, :]]), axis=-2)
    for slc in range(1, rz):
        t = K.expand_dims(Unet([dc_term[:, :, :, slc, :]]), axis=-2)
        seg_class = Concatenate(axis=-2)([seg_class, t])

    return Model(inputs=[input_c, input_m, input_w, input_Atb],
                 outputs=[dc_term, seg_class])


def create_unet_seg(nx, ny, nc, ns, activation_type, USE_BN, depth=4, num_filters=64):
    # define the inputs
    input_x = Input(shape=(nx, ny, 2), dtype=tf.float32)

    # zero-padding for preventing non integer U-net layer
    pnx, pny = 0, 0
    mnx, mny = nx, ny

    if np.mod(nx, (2 ** depth)) > 0:
        mnx = np.int(np.ceil(nx / (2 ** depth)) * (2 ** depth))
        pnx = np.int((mnx - nx) / 2)
    if np.mod(ny, (2 ** depth)) > 0:
        mny = np.int(np.ceil(ny / (2 ** depth)) * (2 ** depth))
        pny = np.int((mny - ny) / 2)

    # input_padded = ZeroPadding2D( padding=(pnx, pny) )(input_x)
    input_padded = ZeroPadding2D(padding=((0, 2 * pnx), (0, 2 * pny)))(input_x)

    # create Unet for segmentation
    Unet = mf.UNet2D_softmax(nx=mnx, ny=mny, ns=ns, kernel_size=(3, 3), num_out_chan_highest_level=num_filters,
                             depth=depth, num_chan_increase_rate=2, activation_type='LeakyReLU', USE_BN=USE_BN)

    # define model    
    out_padded = Unet(input_padded)
    out_x = out_padded[:, :nx, :ny, :]

    return Model(inputs=input_x,
                 outputs=out_x)
