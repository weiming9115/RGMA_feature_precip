#!/bin/bash/

for year in {2001..2008}
do 
    echo "writeout GPM gridded: " $year
    python RGMA_GPM_writeout.py $year
    sleep 2
done
