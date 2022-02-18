# computer-vision-ETH
Python assignments for the class 'Computer Vision' (Prof. Pollefeys) @ ETH Zurich

## Assignment 1: Simple 2D Classifier & Digit Classifier using PyTorch
### Simple 2D Classifier

The first task is training a simple binary classifier on the 2D data visualized in the following figure: 
<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment1/images/2DClassifier.png width="350" title="clustered data">
</p>
In the figure, red points are part of cluster 0, and blue points part of cluster 1.

Both a linear classifier and a multi-layer perceptron (MLP) classifier were implemented using PyTorch, where naturally only the MLP showed near-perfect classification results (over 99% accuracy), due to the non-linear (rather circular) separation between the two clusters. Similarly, a coordinate change to polar coordinates while using the linear classifier also showed good results (over 90% accuracy).

### Digit Classifier

For this part, a MLP using 1 hidden layer as well as a conovolution neural network (CNN) were used to classify digits from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset:
<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment1/images/DigitClassifier.png width="350" title="MNIST digits">
</p>
While needing less parameters, the CNN achieved better accuracies (over 98%) than the MLP. This shows the advantage of CNN for certain types of tasks.
Finally, a so-called confusion matrix for both the MLP (left) and the CNN (right) has been produced, showing the performance of a classifier; M<sub>i,j</sub> is the number of test samples for which the network predicted <em>i</em>, but the ground-truth label was <em>j</em>:

<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment1/images/ConfusionMatrix_MLP.png width="350" title="MLP confusion matrix">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment1/images/ConfusionMatrix_Conv.png width="350" title="CNN confusion matrix">
</p>


## Assignment 2: Mean-Shift Algorithm and SegNet

### Mean-Shift Algorithm for Image segmentation

A vectorized version of the mean-shift algorithm was implemented in order to segment an image of a cow shown on the left; the ground-truth is shown in the middle and the result on the right:
<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment2/images/cow.jpg width="250" title="original cow image">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment2/images/cow_sample_result.png width="250" title="ground truth image segmentation">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment2/images/cow_result.png width="250" title="resulting mage segmentation">
</p>

### Implement and train a simplified version of SegNet

In the second part, a simplified version of the SegNet CNN has been implemented, as depicted below. The resulting architecture has then been trained and shown to achieve accuracies (mean Intersection over Union (IoU)) of over 85%.:
<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment2/images/SegNet.png width="250" title="Simplified SegNet architecture">
</p>
