##########################################################
# %%
# Library for tensorflow 2.2.0 3D MoDL
##########################################################

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
    Add, Conv3DTranspose, LeakyReLU, Lambda, concatenate
    
from tensorflow.keras.models import Model

# import custom functions 
import library_common as mf

##########################################################
# %%
# MoDL functions
##########################################################

c2r=Lambda(lambda x:tf.stack([tf.math.real(x),tf.math.imag(x)],axis=-1))
r2c=Lambda(lambda x:tf.complex(x[...,0],x[...,1]))

class Aclass:
    def __init__(self, csm,mask,lam):
        with tf.name_scope('Ainit'):
            s=tf.shape(mask)
            self.nrow,self.ncol,self.nz =   s[0],s[1],s[2]
            self.pixels         =   self.nrow*self.ncol*self.nz 
            self.mask           =   mask
            self.csm            =   csm
            self.SF             =   tf.complex(tf.sqrt(tf.cast(self.pixels, dtype=tf.float32) ),0.)
            self.lam            =   lam
            
    def myAtA(self,img):
        with tf.name_scope('AtA'):
            coilImages  =   self.csm*img
            kspace      =   tf.signal.fft3d(coilImages)/self.SF
            temp        =   kspace*self.mask
            coilImgs    =   tf.signal.ifft3d(temp)*self.SF
            coilComb    =   tf.reduce_sum(coilImgs*tf.math.conj(self.csm),axis=0)
            coilComb    =   coilComb+self.lam*img
        return coilComb
        

def myCG(A,rhs):
    rhs=r2c(rhs)
    cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,10), rTr>1e-5)
    def body(i,rTr,x,r,p):
        with tf.name_scope('cgBody'):
            Ap      =   A.myAtA(p)
            alpha   =   rTr / tf.cast(tf.reduce_sum(tf.math.conj(p)*Ap),dtype=tf.float32)
            alpha   =   tf.complex(alpha,0.)
            x       =   x + alpha * p
            r       =   r - alpha * Ap
            rTrNew  =   tf.cast( tf.reduce_sum(tf.math.conj(r)*r),dtype=tf.float32)
            beta    =   rTrNew / rTr
            beta    =   tf.complex(beta,0.)
            p       =   r + beta * p
        return i+1,rTrNew,x,r,p

    x       =   tf.zeros_like(rhs)
    i,r,p   =   0,rhs,rhs
    rTr     =   tf.cast( tf.reduce_sum(tf.math.conj(r)*r),dtype=tf.float32)
    loopVar =   i,rTr,x,r,p
    out     =   tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[2]
    return c2r(out)


# needed to be changed to multi-coil
class myDC(Layer):
    def __init__(self, **kwargs):
        super(myDC, self).__init__(**kwargs)
        
        self.lam1 = self.add_weight(name='lam1',shape=(1,), initializer=tf.constant_initializer(value=0.03), 
                                    dtype = 'float32', trainable=True)
        
        
    def build(self, input_shape):
        super(myDC, self).build(input_shape)

    def call(self, x):
        
        rhs, csm, mask  =   x
        lam2 =  tf.complex(self.lam1, 0.)
        
        def fn( tmp ):
            c,m,r=tmp
            Aobj=Aclass( c,m,lam2 )
            y=myCG(Aobj,r)
            return y
        
        inp = (csm,mask,rhs)
        rec = tf.map_fn(fn,inp,dtype=tf.float32,name='mapFn' )
        return rec
    
    def lam_weight(self, x):
        img = x[0]
        res = self.lam1 * img            
        return res      

    def lam_weight2(self, x):
        in0, in1 = x        
        res = self.lam1 * (in0 + in1) / 2            
        return res      
    
def RegConvLayers(nx,ny,nz,nLayers,num_filters):
    input_x     = Input(shape=(nx,ny,nz,2 ), dtype = tf.float32)
    
    rg_term     = input_x
    for lyr in range(0,nLayers):    
        rg_term = conv3D_bn_nonlinear(rg_term, num_filters, (3,3,2), activation_type='relu', USE_BN=True, layer_name='')
        
    # go to image space
    rg_term = conv3D_bn_nonlinear(rg_term, 2, (1,1,1), activation_type=None, USE_BN=False, layer_name='')
    
    # skip connection
    rg_term = Add()([rg_term,input_x])
        
    return Model(inputs     =   input_x, outputs    =   rg_term)


##########################################################
# %%
# Create MoDL model
##########################################################

def create_sense_3d(nx, ny, nz, nc, num_block = 10):

    # define the inputs
    input_c     = Input(shape=(nc,nx,ny,nz), dtype = tf.complex64)
    input_m     = Input(shape=(nx,ny,nz ), dtype = tf.complex64)
    input_Atb   = Input(shape=(nx,ny,nz ), dtype = tf.complex64)

    dc_term     = c2r(input_Atb)
    UpdateDC    = myDC()
    
    for blk in range(0,num_block):
        
        rg_term = UpdateDC.lam_weight([dc_term])
        rg_term = Add()([c2r(input_Atb), rg_term])
            
        # Update DC
        dc_term = UpdateDC([rg_term,input_c,input_m])
        
    out_x = dc_term
    
    return Model(inputs     =   [input_c, input_m, input_Atb],
                 outputs    =   out_x)


def create_modl_3d_ki(nx, ny, nz, nc, nLayers, num_block, activation_type, USE_BN, num_filters = 64):

    # define the inputs
    input_x     = Input(shape=(nx,ny,nz,2 ), dtype = tf.float32)
    input_c     = Input(shape=(nc,nx,ny,nz), dtype = tf.complex64)
    input_m     = Input(shape=(nx,ny,nz ), dtype = tf.complex64)
    input_Atb   = Input(shape=(nx,ny,nz ), dtype = tf.complex64)

    dc_term     = c2r(input_Atb)
    RegConv_k   = RegConvLayers(nx,ny,nz,nLayers,num_filters)
    RegConv_i   = RegConvLayers(nx,ny,nz,nLayers,num_filters)
    UpdateDC    = myDC()

    myFFT       = mf.tf_fft3_r()
    myIFFT      = mf.tf_ifft3_r()    
    
    for blk in range(0,num_block):
        # CNN Regularization
        rg_term_i   = RegConv_i(dc_term)        
        rg_term_k   = myIFFT([RegConv_k(myFFT([dc_term]))])        
        rg_term     = UpdateDC.lam_weight2([rg_term_i,rg_term_k])
        # AtA update                 
        rg_term     = Add()([c2r(input_Atb), rg_term])

        # Update DC
        dc_term = UpdateDC([rg_term,input_c,input_m])

    out_x = dc_term
        
    return Model(inputs     =   [input_x, input_c, input_m, input_Atb],
                 outputs    =   out_x)

