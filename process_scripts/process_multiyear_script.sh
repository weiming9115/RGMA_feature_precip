#!/bin/bash

for year in {2001..2019}
do 
    echo "process_FP_ncfiles: " $year
    python process_FP_ncfiles.py $year
    sleep 2
done
