import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from datetime import datetime
from pathlib import Path
from itertools import combinations

import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import warnings
warnings.filterwarnings('ignore')

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

def generate_label_maps(data_FP_merged, tag_binary_labels, assigned_label):
    """tag_labels: [list of binary labels] features that users are interested. 
       order: ar_tag, front_a_tag, lps_tag, mcs_tag. 
       usage: [1,0,0,1] for AR-MCS cooccurrence
       output: feature combination labels 
    """
    
    # if overlapping exists, extract the overlapped areas of precipitation

    precip_FP_selected = data_FP_merged.precipitationCal.where((data_FP_merged.ar_tag == tag_binary_labels[0]) &
                                                  (data_FP_merged.front_a_tag == tag_binary_labels[1]) &
                                                  (data_FP_merged.lps_tag == tag_binary_labels[2]) &
                                                  (data_FP_merged.mcs_tag == tag_binary_labels[3]))
    precip_FP_selected = precip_FP_selected.fillna(0)
    precip_label = precip_FP_selected.where(precip_FP_selected == 0, assigned_label)
    precip_label = precip_label.to_dataset()
    precip_label = precip_label.rename({'precipitationCal': 'tag_{}'.format(assigned_label)})
    
    return precip_label

def differentiate_unexplained(unexplained, data_prec, data_BT):
    
    # filter out the unexplained for deep convection and congestus over the tropics
    
    cc_threshold = 241 # [K]
    Tb_threshold = 310 # [K]
    rr_threshold = 0.5 # [mm/hr]
    
    deep_conv = unexplained.where(data_BT < cc_threshold, 0)
    deep_conv = deep_conv.where(deep_conv == 0, 1) # binary tag: deep_conv or not
    
    nondeep_conv = unexplained.where((data_BT > cc_threshold) & (data_BT < Tb_threshold) &
                                     (data_prec > rr_threshold), 0)
    nondeep_conv = nondeep_conv.where(nondeep_conv == 0, 1) # binary tag: deep_conv or not
    
    return deep_conv, nondeep_conv

if __name__ == '__main__':

    year = sys.argv[1]

    RGMA_DIR = Path('/neelin2020/RGMA_feature_mask/data_product/{}/MERGED_FP'.format(year))
    MERGE_DIR = Path('/neelin2020/RGMA_feature_mask/data_product/{}/MERGE-IR'.format(year))

    for mon in range(1,13):

        fid = list(RGMA_DIR.glob('*{}_v2.nc'.format(str(mon).zfill(2))))[0]
        data_FP_merged = xr.open_dataset(fid)

        data_FP_merged['tags_sum'] = (data_FP_merged.ar_tag + data_FP_merged.mcs_tag 
                                + data_FP_merged.lps_tag + data_FP_merged.front_c_tag 
                                + data_FP_merged.front_w_tag + data_FP_merged.front_s_tag)
        data_FP_merged = data_FP_merged.sel(latitude=slice(-60,60), longitude=slice(0,359.75))
    
        # read MERGE_IR data for differentiating UE into DC and Congestus
        fid = list(MERGE_DIR.glob('Tb_MERGE_IR_{}_{}*'.format(year,str(mon).zfill(2))))[0]
        data_BT = xr.open_dataset(fid)

        data_BT_sub = data_BT 
        data_BT_sub = data_BT.resample(time='6H').nearest()
        data_BT_sub = coordinates_processors(data_BT_sub)

        # check datetime information (2012 MCS missing 12/31) and slice FP_merged accordingly
        date_bt_str = data_BT_sub.time[0]
        date_bt_end = data_BT_sub.time[-1]
        data_FP_merged = data_FP_merged.sel(time=slice(date_bt_str, date_bt_end))

        lon_era5 = np.arange(0,360,0.25)
        lat_era5 = np.arange(-60,60.25,0.25)
        data_BT_re = data_BT_sub.interp(longitude=lon_era5, latitude=lat_era5)    

        # assign one tag for all fronts for simplicity: front_a_tag
        tmp = data_FP_merged['front_c_tag'] + data_FP_merged['front_w_tag'] + data_FP_merged['front_s_tag']
        tmp = tmp.where(tmp == 0, 1)
        data_FP_merged['front_a_tag'] = tmp

        # generate all the combinations of 4 feature cateogries (AR, front, MCS, LPS)
        tag_labels = ['ar_tag','front_a_tag','mcs_tag','lps_tag']

        list_of_comb = []
        for i in [1,2,3,4]: # c41 c42 c43 c44
            tmp = list(combinations(tag_labels, i))
            for j in tmp:
                list_of_comb.append(j)

        # starting with [0, 0, 0, 0] and replaced by 1 if corresponding features are selected
        output_list_exclusive = [] # list of datasets representing different feature selections

        for label, selected_tags in enumerate(list_of_comb):
            #selected_tags = ['mcs_tag', 'ar_tag']
            tag_binary_labels = [0,0,0,0]
            for i in selected_tags:
                if i == 'ar_tag':
                    tag_binary_labels[0] = 1
                if i == 'front_a_tag':
                    tag_binary_labels[1] = 1   
                if i == 'lps_tag':
                    tag_binary_labels[2] = 1
                if i == 'mcs_tag':
                    tag_binary_labels[3] = 1

            # generate a feature-assocaited label ranging from 1 to 15 which correspond to the overlapping condition 
            output_list_exclusive.append(generate_label_maps(data_FP_merged, tag_binary_labels, assigned_label=label+1)) # number 1 to 15
        unexplained = generate_label_maps(data_FP_merged, [0,0,0,0], assigned_label=len(output_list_exclusive)+1) # number 16
    
        deep_conv, nondeep_conv = differentiate_unexplained(unexplained, data_FP_merged.precipitationCal
                                                       ,data_BT_re.tb)
    
        output_feature_combine = output_list_exclusive + [unexplained] #+ [deep_conv] + [nondeep_conv]
        Feature_combination_xr = xr.merge(output_feature_combine)

        Feature_combination_xr['feat_comb_label'] = (Feature_combination_xr['tag_1'] + Feature_combination_xr['tag_2'] +
                                                 Feature_combination_xr['tag_3'] + Feature_combination_xr['tag_4'] +
                                                 Feature_combination_xr['tag_5'] + Feature_combination_xr['tag_6'] +
                                                 Feature_combination_xr['tag_7'] + Feature_combination_xr['tag_8'] +
                                                 Feature_combination_xr['tag_9'] + Feature_combination_xr['tag_10'] +
                                                 Feature_combination_xr['tag_11'] + Feature_combination_xr['tag_12'] +
                                                 Feature_combination_xr['tag_13'] + Feature_combination_xr['tag_14'] +
                                                 Feature_combination_xr['tag_15'] + Feature_combination_xr['tag_16'])
        Feature_combination_xr['deep_conv_mask'] = deep_conv.tag_16
        Feature_combination_xr['nondeep_conv_mask'] = nondeep_conv.tag_16
    
        # save feature
        Feature_combination_xr = Feature_combination_xr[['feat_comb_label','deep_conv_mask','nondeep_conv_mask']]
        Feature_combination_xr.attrs = {'feature indicator': '(1) AR    (2) Front    (3) MCS    (4) LPS    ' + 
                                     '(5) AR + Front    (6) AR + MCS    (7) AR + LPS    (8) Front + MCS    ' +
                                     '(9) Front + LPS    (10) MCS + LPS    (11): AR + Front + MCS    (12) AR + Front + LPS    ' +
                                     '(13) AR + MCS + LPS    (14) Front + MCS + LPS    (15): All    (16) Unexplained',
                                   
                                    'description': 'Feature-assocaited indicator of a raining pixel. Each indicator is mutually exclusive to one another. ' +
                                                   'For instance, label 1 is the raining pixel that is associated with AR only and label 5 is the pixel '+ 
                                                    'where AR and MCS are overlapped. Deep_conv_mask (BT < 241K). nondeep_conv_mask' +
                                                    ' (241 < BT < 310 K) and (rainrate > 0.5 mm/hr).'}
    
        Feature_combination_xr.to_netcdf(RGMA_DIR / 'FeatureOverlap_indicator_map_{}_v2.nc'.format(str(mon).zfill(2)), 
                                    encoding={'feat_comb_label': {'dtype': 'int8'}, 'deep_conv_mask': {'dtype': 'int8'},
                                              'nondeep_conv_mask': {'dtype': 'int8'}})

        print('FeatureOverlap_indicator_map_{}_v2.nc'.format(str(mon).zfill(2)))
