#!/bin/bash

echo "Start repairing classs: "$1
echo "Clients: "$2
echo "Model: "$3
echo "Selected GPU: "$4

./fl_script.sh $3 $1 Cl$1_Mod$3_N$2 $2 $4
./opt_script.sh $3 $1 Cl$1_Mod$3_N$2 100 100 $2 $4