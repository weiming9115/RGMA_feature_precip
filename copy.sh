#!/bin/bash

RGMA_DIR='/neelin2020/RGMA_feature_mask'
scripts_multi='/neelin2020/RGMA_feature_mask/scripts_multiprocess'
scripts_process='/neelin2020/RGMA_feature_mask/data_product'

cp $RGMA_DIR/*ipynb ./notebooks
cp $scripts_multi/*py ./scripts_multiprocess
cp $scripts_process/*py ./process_scripts
