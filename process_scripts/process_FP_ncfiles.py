# processing precipitaion-feature product 
# note: LPS: defined as 10-deg box centered at the feature centroid

import os
import sys
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import warnings
warnings.filterwarnings('ignore')

year = sys.argv[1]

def coordinates_processors(data):
    """
    converting longitude/latitude into lon/lat
    data: xarray.dataset coordinated horizontally in lat/lon
    """

    coord_names = []
    for coord_name in data.coords:
        coord_names.append(coord_name)

    if (set(coord_names) & set(['lon','lat'])): # if coordinates set this way...

        data2 = data.rename({'lat': 'latitude'})
        data2 = data2.rename({'lon': 'longitude'})

    else:
        data2 = data

    # check if lon from -180
    if data2.longitude[0] != 0: # -180 to 180

        lon_reset = data2.longitude
        lon_reset = lon_reset.where(lon_reset > 0, 360+lon_reset) # converting lon as 0 to 359.75
        data2.coords['longitude'] = lon_reset # converting lon as -180 to 180
        data2= data2.sortby('longitude')

    # check if latitutde is decreasing
    if (data2.latitude[1] - data2.latitude[0]) < 0:
        data2 = data2.isel(latitude=slice(None, None, -1)) # flipping latitude accoordingly

    return data2

########################################################
# declare directories
RGMA_DIR = Path('/neelin2020/RGMA_feature_mask/')
GPM_DIR = Path('/neelin2020/RGMA_feature_mask/GPM_ncfiles_{}/'.format(year))
DATA_DIR = RGMA_DIR / 'data_product/{}'.format(year)
AR_DIR = DATA_DIR / 'AR'
MCS_DIR = DATA_DIR / 'MCS_orig'
LPS_DIR = DATA_DIR / 'LPS'
Front_DIR = DATA_DIR / 'Front_expand'
OUT_DIR = DATA_DIR / 'MERGED_FP'
if OUT_DIR.exists() == False:
    os.system('mkdir -p {}'.format(OUT_DIR))

lat_range = [-60, 60] # MSC only tracked within 60 deg.

#########################################################

# generate feature-precipitation dataset separated by months
for mon in range(3,13): 
    print('current month processed: {}'.format(str(mon).zfill(2)))

    # load feature datasets 
    files = list(AR_DIR.glob('*_{}_*'.format(str(mon).zfill(2))))[0]
    data_AR = xr.open_dataset(files)
    data_AR = coordinates_processors(data_AR)
    data_AR = data_AR.sel(latitude=slice(lat_range[0], lat_range[1]))

    files = list(Front_DIR.glob('Front_cold*_{}_*'.format(str(mon).zfill(2))))[0]
    data_FT_c = xr.open_dataset(files)
    data_FT_c = coordinates_processors(data_FT_c)
    data_FT_c = data_FT_c.sel(latitude=slice(lat_range[0], lat_range[1]))

    files = list(Front_DIR.glob('Front_warm*_{}_*'.format(str(mon).zfill(2))))[0]
    data_FT_w = xr.open_dataset(files)
    data_FT_w = coordinates_processors(data_FT_w)
    data_FT_w = data_FT_w.sel(latitude=slice(lat_range[0], lat_range[1]))

    files = list(Front_DIR.glob('Front_stat*_{}_*'.format(str(mon).zfill(2))))[0]
    data_FT_s = xr.open_dataset(files)
    data_FT_s = coordinates_processors(data_FT_s)
    data_FT_s = data_FT_s.sel(latitude=slice(lat_range[0], lat_range[1]))
    
    files =  list(LPS_DIR.glob('*_{}_*'.format(str(mon).zfill(2))))[0]
    data_LPS = xr.open_dataset(files)
    data_LPS = coordinates_processors(data_LPS)
    data_LPS = data_LPS.sel(latitude=slice(lat_range[0], lat_range[1]))

    files =  list(MCS_DIR.glob('*_{}_*'.format(str(mon).zfill(2))))[0]
    data_MCS = xr.open_dataset(files)
    data_MCS = coordinates_processors(data_MCS)
    data_MCS = data_MCS.sel(latitude=slice(lat_range[0], lat_range[1]))
    
    # GPM hrly output format
    files = sorted(list(GPM_DIR.glob('GPM_IMERGE_V06_{}{}*'.format(year, str(mon).zfill(2)))))
    data_GPM = xr.open_mfdataset(files)
    data_GPM = coordinates_processors(data_GPM)
    data_GPM_sub = data_GPM.sel(latitude=slice(lat_range[0], lat_range[1]))

    ##########################################################

    # reset variable name as feature_tag (ar_tag, mcs_tag, lps_tag, front_tg)
    data_AR = data_AR.rename({'ar_binary_tag': 'ar_tag'})
    data_FT_c = data_FT_c.rename({'fronts': 'front_c_tag'})
    data_FT_w = data_FT_w.rename({'fronts': 'front_w_tag'})
    data_FT_s = data_FT_s.rename({'fronts': 'front_s_tag'})
    data_LPS = data_LPS.rename({'feature_mask': 'lps_tag'})
    data_MCS = data_MCS.rename({'feature_mask': 'mcs_tag'})
 
    lon_era5 = data_AR.longitude
    lat_era5 = data_AR.latitude

    # MCS is defined at xx:30 based on MERGE-IR, we assume small changes in the feature within 
    # the 30-min time frame for simplicity

    MCS_array = np.zeros(data_AR.ar_tag.shape)
    MCS_array[:len(data_MCS.mcs_tag.time),:,:] = data_MCS.mcs_tag.values # fill the new array by the original tag

    data_MCS_re = xr.Dataset(data_vars=dict(mcs_tag=(['time','latitude','longitude'], MCS_array.astype('int8'))),
                             coords=dict(time=data_AR.time.values,
                                         latitude=(['latitude'], data_MCS.latitude.values),
                                         longitude=(['longitude'], data_MCS.longitude.values)))

    # interpolate GPM into ERA5 grid using quadratic fitting
    data_GPM_intp = data_GPM_sub.interp(longitude=lon_era5, latitude=lat_era5, method='linear').fillna(0)
    data_GPM_intp = data_GPM_intp.transpose('time','latitude','longitude')

    # final step: merge into a feature-precip (FP) xarray dataset
    # note: resample 6H due to front data 
    data_FP_merged = xr.merge([data_GPM_intp.resample(time='6H').nearest(),
                               data_AR.resample(time='6H').nearest(),
                               data_FT_c.resample(time='6H').nearest(),
                               data_FT_w.resample(time='6H').nearest(),
                               data_FT_s.resample(time='6H').nearest(),
                               data_LPS.resample(time='6H').nearest(),
                               data_MCS_re.resample(time='6H').nearest()])
    data_FP_merged['precipitationCal'] = data_FP_merged['precipitationCal'].where(data_FP_merged['precipitationCal'] > 0, 0)

    # reassign time 
    data_FP_merged.coords['time'] = data_AR.resample(time='6H').nearest().time.values
    data_FP_merged.to_netcdf(OUT_DIR / 'GPM_feature_merged_{}_v2.nc'.format(str(mon).zfill(2)), 
                            encoding={'ar_tag': {'dtype': 'int8'},
                                      'mcs_tag': {'dtype': 'int8'},
                                      'lps_tag': {'dtype': 'int8'},
                                      'front_c_tag': {'dtype': 'int8'},
                                      'front_w_tag': {'dtype': 'int8'},
                                      'front_s_tag': {'dtype': 'int8'} })
    print('GPM_feature_merged_{}_v2.nc saved...'.format(str(mon).zfill(2)))

