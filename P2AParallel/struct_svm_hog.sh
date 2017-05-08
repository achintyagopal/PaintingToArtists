#!/bin/bash

ITER=20

feature="hog"
algorithm="struct_svm"

printf "FEATURE: "
printf $feature
printf "\n"
printf "ALGORITHM: "
printf $algorithm
printf "\n"

# # NATIVE
# python main.py --mode feature --folder ../data --procs 1 --feature-file $feature.feature.file --feature-algorithm $feature
# python main.py --mode feature --folder ../data --procs 2 --feature-file $feature.feature.file --feature-algorithm $feature
python main.py --mode feature --partition true --platform ipython --num_parts 10 --folder ../data --procs 3 --feature-file $feature.feature.file --feature-algorithm $feature
python main.py --mode feature --partition true --platform ipython --direct false --num_parts 10 --folder ../data --procs 3 --feature-file $feature.feature.file --feature-algorithm $feature
# printf "pypy"
# printf "new data\n"
# python main.py --mode feature --folder ../cv --procs 1 --feature-file $feature.feature.file --feature-algorithm $feature
# python main.py --mode feature --folder ../cv --procs 2 --feature-file $feature.feature.file --feature-algorithm $feature
# python main.py --mode feature --folder ../cv --procs 3 --feature-file $feature.feature.file --feature-algorithm $feature

# 1 ipyhon inst
# python main.py --mode feature --folder ../data --platform ipython --feature-file $feature.feature.file --feature-algorithm $feature
# printf "new data\n"
# python main.py --mode feature --folder ../cv --platform ipython --feature-file $feature.feature.file --feature-algorithm $feature


# 1 ipython lbv inst
# python main.py --mode feature --platform ipython --direct false --folder ../data --feature-file $feature.feature.file --feature-algorithm $feature
# printf "new data\n"
# python main.py --mode feature --folder ../cv --platform ipython --direct false  --feature-file $feature.feature.file --feature-algorithm $feature



# python main.py --mode feature --folder ../cv --platform ipython --direct false  --feature-file $feature.feature.file --feature-algorithm $feature

# for i in $(seq 0 $ITER);
# do
# 	printf "Iteration: "
# 	printf $i + 1
# 	printf "\n"
# 	python main.py --mode train --feature-file $feature.feature.file --training-algorithm $algorithm --model-file $algorithm.$feature.model.file
# 	python main.py --mode test --feature-file $feature.feature.file --model-file $algorithm.$feature.model.file --predictions-file $algorithm.$feature.predictions
# done	