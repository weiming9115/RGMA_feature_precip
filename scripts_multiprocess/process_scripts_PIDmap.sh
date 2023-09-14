#!/bin/bash/

for year in {2001..2019}
do 
    echo "writeout PID freq + Rcontr: " $year
    python Freq_Contr_PIDmap_writeout.py $year
    sleep 2
done
