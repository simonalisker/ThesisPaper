# File types classification 
## General Overview
This code includes two novel architectures
<br>(1) Transformers NLP for file classification
<br>(2) Handcrafted statistical with FiFTy ensemble
## Directory structure
1. The structure of the /src directory is as follows
  
![image](https://github.com/simonalis/ThesisPaper/assets/104734787/2b1b23d4-b6ee-47ac-83d5-6b398e01fe39)
2. train.npz, test.npz, val.npz files from FiFTy dataset should be located under /src/512_1/ directory
3. 2 models created are  under /src/512_1/ directory

![image](https://github.com/simonalis/ThesisPaper/assets/104734787/2bb71063-e507-4813-8e2e-c6e2bf617bf9)
4. All outputs will be created under /src/512_1/ directory
## Files description
1. LoadData - loads the data from .npz files and is used for both models
2. TorchDataLoaderWithNLP - creates NLP based architecture, trains and predicts accuracy using FiFTy dataset (*.npz files)
This file is for related to the implementation of architecture (1).
3. The architecture (2) is created in 2 steps - creates statistical features, creates and trains statistical model (a) and then creates ensemble using statistical model and Fifty model, then we are doing train with Transfer Learning and predict (b).
(a)StatisticalFeatures
(b)StatDNNwithFiFTy
## How to execute
In order explore architecture
1. (1), please use the TorchDataLoaderWithNLP file as main
2. (2), please use StatDNNwithFiFTy file as main

It is possible to run only predict function for each model as models are loaded to this repository.
(1) saved_weights_512_1_full_chunked_roberta.pt
(2) tf_stat_fifty_weights.h5
The above models can be downloaded from my drive:

You will need to change the code appropriately, comment train function and uncomment lines which load weights near train function.

It is also possible to train models from scratch and the run predict function.

## File[test.npz](src%2F512_1%2Ftest.npz) Types
Classifier to 75 file types
TBD
## References
https://github.com/mittalgovind/fifty/tree/master
