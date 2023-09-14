#!/bin/bash

for year in {2001..2019}
do 
    echo "process_PID_ncfiles: " $year
    python process_PID_writeout.py $year
    sleep 2
done
