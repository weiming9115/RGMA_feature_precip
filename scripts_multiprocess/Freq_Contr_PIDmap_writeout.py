# write monthly frequency PID map + monthly precip contribution PID map

import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

def assign_convection_label(data_overlap):

    # assign PID for deep convection and congestus (17 & 18)
    data_pid = data_overlap.feat_comb_label
    deep_mask = data_pid.where(((data_pid == 16) & (data_overlap.deep_conv_mask == 1)), 0)
    deep_mask = deep_mask.where(deep_mask == 0, -1) # negative for masked grids
    data_pid = data_pid.where(deep_mask == 0, 17)

    congest_mask = data_pid.where(((data_pid == 16) & (data_overlap.nondeep_conv_mask == 1)), 0)
    congest_mask = congest_mask.where(congest_mask == 0, -1) # negative for masked grids
    data_pid = data_pid.where(congest_mask == 0, 18)
    
    return data_pid

def process_PID_writeout(i):
    # at each grid, return the PID that occurs most frequently and the PID accounts for most rainfall

    PID_timeseries = PID_1d[:,i]
    prec_timeseries = prec_1d[:,i]

    freq_counts = np.zeros(18) # 18 PIDs
    rain_sum = np.copy(freq_counts) # 18 PIDs

    if len(PID_timeseries) == len(prec_timeseries): # both with the same length
        
        for idt, idx in enumerate(PID_timeseries):
            
            if idx > 0: # if identified PID exits --> raining            
    
                freq_counts[idx-1] += 1
                rain_sum[idx-1] += prec_timeseries[idt]

        if np.sum(freq_counts) > 0:
            PID_freq = np.argmax(freq_counts) + 1 # return defined PID
            PID_Rcontr = np.argmax(rain_sum) + 1 # return defined PID
            Rcontr_percent = rain_sum/freq_counts
            freq_percent = freq_counts/np.sum(freq_counts)            

        else:
            PID_freq = 0
            PID_Rcontr = 0
            Rcontr_percent = np.zeros(18)
            freq_percent = np.zeros(18)
    else:
        raise ValueError('data time dimension not identical...')

    return PID_freq, PID_Rcontr, Rcontr_percent, freq_percent


if __name__ == '__main__':

    from multiprocessing import Pool

    start_time = datetime.now()

    yr = sys.argv[1]  

    RGMA_DIR = Path('/neelin2020/RGMA_feature_mask/data_product/{}/MERGED_FP'.format(yr))
    MERGE_DIR = Path('/neelin2020/RGMA_feature_mask/MERGE-IR/{}/'.format(yr))

    files_gpm = sorted(list(RGMA_DIR.glob('*_expand.nc')))
    files_PID = sorted(list(RGMA_DIR.glob('*_convmask.nc')))

    data_FP_merged = xr.open_mfdataset(files_gpm, engine='netcdf4')
    data_gpm = data_FP_merged.precipitationCal
    data_overlap = xr.open_mfdataset(files_PID, engine='netcdf4')

    # assgin DC and CG as PID 17 and 18, respectively
    data_pid = assign_convection_label(data_overlap)

    data_xr_list = [] # empty list for saving stats of individual months
    for mon in range(1,13):

        PID_sub = data_pid.sel(time=data_pid.time.dt.month.isin([mon])).compute() # select specific month
        avail_time = PID_sub.time # NOTE: 2012 12 31 is missing for MCS PYFLEXTRKR, so ensure mathced times here
        prec_sub = data_gpm.sel(time=data_gpm.time.dt.month.isin([mon])) # select specific month
        prec_sub = prec_sub.sel(time=avail_time).compute()

        PID_1d = PID_sub.values.reshape(len(PID_sub.time), len(PID_sub.latitude)*len(PID_sub.longitude))
        prec_1d = prec_sub.values.reshape(len(prec_sub.time), len(prec_sub.latitude)*len(prec_sub.longitude))

        # start multiprocessing
        pool = Pool(processes=10)
        result = pool.map(process_PID_writeout, range(PID_1d.shape[1]))

        # create empty arrays for saving
        freq_PID_array = []
        contr_PID_array = []
        contr_percent_array = []
        freq_percent_array = []       

        freq_PID_map = np.zeros((len(PID_sub.latitude), len(PID_sub.longitude)))
        contr_PID_map = np.copy(freq_PID_map)
        contr_percent_map = np.zeros((len(PID_sub.latitude), len(PID_sub.longitude), 18))
        freq_percent_map = np.copy(contr_percent_map)

        for n in range(len(result)):
            freq_PID_array.append(result[n][0])
            contr_PID_array.append(result[n][1])
            contr_percent_array.append(result[n][2])
            freq_percent_array.append(result[n][3])
        
        # reshape 1d output into gridded map and gernerate netcdf 
        freq_PID_map = np.reshape(np.asarray(freq_PID_array), (len(PID_sub.latitude),len(PID_sub.longitude)))
        contr_PID_map = np.reshape(np.asarray(contr_PID_array), (len(PID_sub.latitude),len(PID_sub.longitude)))
        contr_percent_map = np.reshape(np.asarray(contr_percent_array), (len(PID_sub.latitude),len(PID_sub.longitude),18))
        freq_percent_map = np.reshape(np.asarray(freq_percent_array), (len(PID_sub.latitude),len(PID_sub.longitude),18))

        freq_PID_xr = xr.Dataset(data_vars=dict(
                             PID_freq=(['latitude','longitude'], freq_PID_map, {'description':'precipitation feature ID'}),
                             PID_Rcontr=(['latitude','longitude'], contr_PID_map, {'description':'precipitation feature ID'}),
                             Rcontr_percent=(['latitude','longitude','PID'], contr_percent_map, {'description':'rainfall contribution ratio'}),
                             freq_percent=(['latitude','longitude','PID'], freq_percent_map, {'description':'occurrence ratio'}
                                 )
                                ),
                                 coords=dict(
                                 longitude=(['longitude'], PID_sub.longitude.values),
                                 latitude=(['latitude'], PID_sub.latitude.values),
                                 PID=(['PID'], np.arange(1,19)),
                                ),
                                 attrs=dict(description='Statistics based on 2001-2019 feature-associated precipitation dataset')
        )               

        data_xr_list.append(freq_PID_xr)

        pool.close()
        pool.join()
    #------------- end of multiprocessing ---------------

    # merge individual months into a single file
    OUT_DIR = Path('/neelin2020/RGMA_feature_mask/data_product/')

    FID_map_sum = xr.concat(data_xr_list, pd.Index(np.arange(1,13), name='month'))
    FID_map_sum.to_netcdf(OUT_DIR/'PID_freq_Rcontr_{}.nc'.format(yr),
                         encoding={'PID_freq': {'dtype': 'int8'}, 
                                  'PID_Rcontr': {'dtype': 'int8'},
                                  'Rcontr_percent': {'dtype': 'float32'},
                                  'freq_percent': {'dtype': 'float32'}})

    end_time = datetime.now()
    print('Execution time spent: {}'.format(end_time - start_time))
