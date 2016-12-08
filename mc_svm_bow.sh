#!/bin/bash

ITER=20

feature="bow"
algorithm="mc_svm"

printf "FEATURE: "
printf $feature
printf "\n"
printf "ALGORITHM: "
printf $algorithm
printf "\n"

python main.py --mode feature --folder data --feature-file $feature.feature.file --feature-algorithm $feature

for i in $(seq 0 $ITER);
do
	printf "Iteration: "
	printf $i + 1
	printf "\n"
	python main.py --mode train --feature-file $feature.feature.file --training-algorithm $algorithm --model-file $algorithm.$feature.model.file
	python main.py --mode test --feature-file $feature.feature.file --model-file $algorithm.$feature.model.file --predictions-file $algorithm.$feature.predictions
done	