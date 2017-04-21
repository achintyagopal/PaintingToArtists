
# Things that can be parallel
## Data Processing
* Resizing all images to a standard size

## Feature Vector Encodings For Training and Testing 
* HOG
* Color Histogram
* Bag of Words

## Parallelize the Prediction of Images
* Main
** MultiSVM
** DualSVM

## Machine Learning Training Algorithms
### Multi SVM
* Trains 1 SVM per label on all images, so can do training in parallel. i.e. train 3 Dual SVM at once




