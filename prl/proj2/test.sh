#!/bin/bash

display_usage() {  
	echo -e "test.sh: ./test.sh n\n\tLine-of-sight\n\n\tOptions:\n\tn - string with altitudes \"observer, altitude1, ..., altitudeN\"" 
	} 

#pocet cisel bud zadam nebo 10 :)
if [ $# -lt 1 ] || [ $# -gt 1 ];then 
    display_usage
    exit 1
else
    numbers=$1;
fi;

#input size
n=$(( `echo $numbers | tr -cd , | wc -c` ))

#number of CPUs
N=`python3 -c "from scipy import optimize; import scipy; import numpy; 
optimal = scipy.optimize.fsolve(lambda x: $n - numpy.lib.scimath.log2(x) * x, 8)[0]; 
print(int(numpy.power(2, numpy.floor(numpy.lib.scimath.log2(optimal)))))"`

# echo "input size:    " $n
# echo "number of CPUs:" $N

#preklad cpp zdrojaku
mpic++ --prefix /usr/local/share/OpenMPI -o vid vid.cpp

#spusteni
mpirun --prefix /usr/local/share/OpenMPI -np $N vid $numbers

#uklid
rm -f vid