#!/bin/bash
echo "Repairing model: " $1
echo "Negative class: "$2
echo "Run id: "$3
echo "Creating output directory outputs/"$1"/"$3
echo "Number of clients"$4
echo "Running on GPU"$5
mkdir outputs/$1/$3

if [ $(($3)) -gt 0 ]
then
        echo "NOT copying positive results from first run"
fi

CUDA_VISIBLE_DEVICES=$5 repair localize --method=FedRep --model_dir=outputs/$1 --positive_inputs_dir=outputs/$1/positive --target_data_dir=outputs/$1/negative/$2 --output_dir=outputs/$1/$3 --n_clients=$4 > exp_$1_$2_$3.txt