#!/bin/bash
echo "Repairing model: " $1
echo "Negative class: "$2
echo "Run id: "$3
echo "Population size: "$4
echo "Number of generations: "$5
echo "Number of clients: "$6
echo "Running on GPU: "$7
echo "Creating output directory outputs/"$1"/"$3
mkdir outputs/$1/$3

if [ $(($3)) -gt 0 ]
then
        echo "NOT copying positive results from first run"
fi

CUDA_VISIBLE_DEVICES=$7 repair optimize  --method=FedRep --model_dir=outputs/$1 --positive_inputs_dir=outputs/$1/positive --target_data_dir=outputs/$1/negative/$2 --output_dir=outputs/$1/$3 --num_gen=$5 --num_pop=$4 --n_clients=$6 > expOpt__$1_$2_$3.txt