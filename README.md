# Convolutional Neural Networks for Disease Classification
Three PyTorch implemented CNN architectures and their respective trainers for 
evaluation over the ChestX-ray14 dataset of thorax diseases. This dataset presents a
class imbalanced multi-label classification problem, with 112,120 
frontal-view X-rays labelled for fourteen different diseases. The architectures
and trainers demonstrate use of the following techniques and optimizations: 

- K-fold cross validation and early stopping with restoration of best models
- Undersampling of training set to address class imbalance problems 
- Xavier initialization  
- Confusion, Accuracy, Loss, Precision, and Recall recording and plotting 
- Batch normalization
- Adam optimization
- ... and more! See modules for additional information. 


### Example training metrics and results 
##### Training and Validation Set Performance 
<img src="https://github.com/GerardMaggiolino/ChestX-ray14-CNN/blob/master/sample/accuracy.png" width="50%" height="50%">

##### Class Confusion Matrices 
<img src="https://github.com/GerardMaggiolino/ChestX-ray14-CNN/blob/master/sample/confusion.png" width="50%" height="50%">

##### CNN Filter Visualizations 
<img src="https://github.com/GerardMaggiolino/ChestX-ray14-CNN/blob/master/sample/filter.png" width="25%" height="25%">

##### Results of Class Balancing through Undersampling 
<img src="https://github.com/GerardMaggiolino/ChestX-ray14-CNN/blob/master/sample/undersampling.png" width="20%" height="20%">

### Dataset citation: 
Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017, July).
Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on
weakly-supervised classification and localization of common thorax diseases. In
Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on (pp.
3462-3471). IEEE.
