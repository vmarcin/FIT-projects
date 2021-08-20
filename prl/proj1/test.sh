#!/bin/bash

display_usage() {  
	echo -e "test.sh: ./test.sh n\n\tGenerates n random numbers and then sorts them from the smallest to the largest.\n\n\tOptions:\n\tn - number of elements to be sorted" 
	} 

if [ $# -lt 1 ];then 
    display_usage
    exit 1
else
    numbers=$1;
fi;

#compile
mpicc --prefix /usr/local/share/OpenMPI -o oets ots.c

#generate numbers
dd if=/dev/random bs=1 count=$numbers of=numbers > /dev/null 2>&1

#run
mpirun --prefix /usr/local/share/OpenMPI -np $numbers oets

#clean
rm -f oets numbers
