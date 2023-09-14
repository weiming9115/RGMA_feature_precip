# multiprocessing GPM netcdf files
import sys
import os
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
import h5py

from multiprocessing import Pool

import warnings

# year selected
year = sys.argv[1]
###############

RGMA_DIR = Path('/neelin2020/RGMA_feature_mask')
GPM_DIR = Path('/neelin2020/IMERG_V06/')
OUT_DIR = RGMA_DIR / 'GPM_ncfiles_{}'.format(year)
if OUT_DIR.exists() == False:
    os.system('mkdir -p {}'.format(OUT_DIR))  

def process_writeout(datetime_sel):

    data_dir = Path(GPM_DIR / '{}/{}'.format(datetime_sel.year, str(datetime_sel.month).zfill(2)))
    date_str = datetime_sel.strftime('%Y%m%d') # e.g., 20140101
    time_str = datetime_sel.time().strftime('%H%M') # e.g., 0030

    # get file id
    file_id = list(data_dir.glob('3B-HHR.MS.MRG.3IMERG.{}-S{}00*'.format(date_str, time_str)))[0]
    data = h5py.File(file_id)

    lat = data['Grid']['lat'][:]
    lon = data['Grid']['lon'][:]
    precip = data['Grid']['precipitationCal'][0,:,:] # get numpy array
    precip_3d = precip.reshape((1, precip.shape[0], precip.shape[1]))
    time_stamp = np.datetime64(int(data['Grid']['time'][:][0].astype('int64')), 's')

    precip_xr = xr.Dataset(data_vars=dict(precipitationCal=(['time','lon','lat'], precip_3d, {'long_name': 'precipitationCal', 'unit': 'mm/h'})
                                     ),
                       coords=dict(time=(['time'], [time_stamp]),
                                   lat=(['lat'], lat , {'long_name': 'latitude', 'unit': 'degree'}),
                                   lon=(['lon'], lon , {'long_name': 'longitude', 'unit': 'degree'})
                                     ),
                      )

    # selecting areas
    precip_valid = precip_xr.where(precip_xr.precipitationCal >= 0) # replace -9999 by np.nan
    
    precip_valid.to_netcdf(OUT_DIR / 'GPM_IMERGE_V06_{}{}{}_{}00.nc'.format(datetime_sel.year, str(datetime_sel.month).zfill(2),
                                                               str(datetime_sel.day).zfill(2), str(datetime_sel.hour).zfill(2)))


if __name__ == '__main__':
    
    date_range = pd.date_range(datetime(int(year),1,1,0), datetime(int(year),12,31,23), freq='1H')

    pool = Pool(processes=10)
    pool.map(process_writeout, date_range)

    pool.close()
    pool.join()

    print('data processing finished')
