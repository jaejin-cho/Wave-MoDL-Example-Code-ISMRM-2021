##########################################################
# %%
# tensorflow version 2.2.0
##########################################################

import os,sys
os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"

sys.path.append('utils')

##########################################################
# %%
# import some libraries
##########################################################

import  numpy               as  np
import  library_common      as  mf
import  library_modl        as  mm
import  library_wave        as  mw

##########################################################
# %%
# parameters
##########################################################

ry          = 3
rz          = 3

# wave parameter
zpadf       = 3
        
# network size
num_block   = 10
nLayers     = 5
num_filters = 24
num_slc, nx, ny, nc   = 256, 192, 224, 12

##########################################################
# %%
# load data
##########################################################
data    =   np.load('data/data_example.npz')

icsm    =   data['arr_0'] 
imskw   =   data['arr_1']
imskc   =   imskw[:,:nx,]
ipsf    =   data['arr_2']
iAtbw   =   data['arr_3']
iAtbc   =   data['arr_4']
oimg    =   data['arr_5']
# oseg    =   data['arr_6']

##########################################################
# %%
# create sense/wave-caipi network
##########################################################

sens_model = mm.create_sense_3d(nx, ny, rz, nc, num_block = num_block)
sens_model.compile(optimizer=[],loss=mf.nrmse_loss) 

wave_model = mw.create_wave_caipi(nx, ny, rz, nc, num_block = num_block, zpf = zpadf)
wave_model.compile(optimizer=[],loss=mf.nrmse_loss) 


##########################################################
# %%
# create modl network
##########################################################

modl = mm.create_modl(  nx  = nx,
                        ny  = ny,
                        nz  = rz,
                        nc  = nc,
                        num_block       =   num_block, 
                        nLayers         =   nLayers, 
                        num_filters     =   num_filters)    
    
modl.compile(optimizer=[],loss=mf.nrmse_loss) 


wave_modl = mw.create_wave_modl(    nx  = nx,
                                    ny  = ny,
                                    rz  = rz,
                                    nc  = nc,
                                    zpf = zpadf,
                                    num_block       =   num_block, 
                                    nLayers         =   nLayers, 
                                    num_filters     =   num_filters)    
    
wave_modl.compile(optimizer=[],loss=mf.nrmse_loss) 

##########################################################
# %%
# loading the network
##########################################################

modl.load_weights('network/modl.hdf5')
wave_modl.load_weights('network/wave_modl.hdf5')


##########################################################
# %%
# Prediction
##########################################################

input_cartesian = [icsm,imskc,iAtbc]
input_wave      = [icsm,imskw,ipsf,iAtbw]

P_sens          =  sens_model.predict(input_cartesian)
P_modl          =  modl.predict(input_cartesian)
P_wave          =  wave_model.predict(input_wave)
P_wave_modl     =  wave_modl.predict(input_wave)

L_sens          =  sens_model.evaluate(input_cartesian,oimg) 
L_modl          =  modl.evaluate(input_cartesian,oimg) 
L_wave          =  wave_model.evaluate(input_wave,oimg) 
L_wave_modl     =  wave_modl.evaluate(input_wave,oimg) 

##########################################################
# %%
# diplay the results
##########################################################

mf.mosaic(np.rot90(np.abs(P_sens[0,...,0]+1j*P_sens[0,...,1])),1,rz,101,[0,1],'SENSE NRMSE : %.2f' % L_sens)
mf.mosaic(np.rot90(np.abs(P_modl[0,...,0]+1j*P_modl[0,...,1])),1,rz,102,[0,1],'MoDL NRMSE : %.2f' % L_modl)

mf.mosaic(np.rot90(np.abs(P_wave[0,...,0]+1j*P_wave[0,...,1])),1,rz,103,[0,1],'Wave-CAIPI NRMSE : %.2f' % L_wave)
mf.mosaic(np.rot90(np.abs(P_wave_modl[0,...,0]+1j*P_wave_modl[0,...,1])),1,rz,104,[0,1],'Wave-MoDL NRMSE : %.2f' % L_wave_modl)
