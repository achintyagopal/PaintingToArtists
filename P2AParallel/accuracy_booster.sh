#!/bin/bash

ITER=20

feature1="color"
feature2="hog"
filename="hog2"
algorithm="boost"

printf "FEATURE: "
printf $feature
printf "\n"
printf "ALGORITHM: "
printf $algorithm
printf "\n"

python main.py --mode feature --folder data2 --feature-file $feature2.feature.file --feature-algorithm $feature2
python main.py --mode feature --folder data --feature-file $filename.feature.file --feature-algorithm $feature2

python main.py --mode feature --folder data --feature-file $feature1.feature.file --feature-algorithm $feature1

# for i in $(seq 0 $ITER);
# do
# 	printf "Iteration: "
# 	printf $i + 1
# 	printf "\n"
# 	python main.py --mode train --feature-file $feature.feature.file --training-algorithm $algorithm --model-file $algorithm.$feature.model.file
# 	python main.py --mode test --feature-file $feature.feature.file --model-file $algorithm.$feature.model.file --predictions-file $algorithm.$feature.predictions
# done	