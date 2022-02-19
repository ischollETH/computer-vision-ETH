# computer-vision-ETH
Python assignments for the class 'Computer Vision' (Prof. Pollefeys) @ ETH Zurich

## Assignment 1: Classifiers using PyTorch
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
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment2/images/SegNet.png width="750" title="Simplified SegNet architecture">
</p>


## Assignment 3: Camera Calibration and Structure from Motion (SfM)

### Camera Calibration

A camera has been calibrated based on correspondences between an image and points on a calibration target. Given the correspondences, the (simplified) process of calibrating the camera consists of data normalization, estimating the projection matrix using Direct Linear Transform (DLT), optimizing based on the reprojection errors and then decomposing it into camera intrinsics (K matrix) and pose (rotation matrix R and translation vector T). The result with the reprojected points from the calculated pose and intrinsics are shown in the following figure:

<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment3/images/calibration_result.png width="750" title="Resulting camera pose and reprojected points">
</p>

### Structure from Motion (SfM)

From provided extracted keypoints and feature matches, the SfM pipeline combines reative pose estimation, absolute pose estimation and point triangulation. To initialize the scene, the relative pose between two images has been estimated; afterwards, the first points in the scene have been triangulated and then iteratively more images were registered and new points triangulated. Using this, for the fountain shown in the left image, the resulting 3D pointcloud including the camera poses of the used views can be obtained as in the right image:

<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment3/images/fountain.png width="400" title="Fountain that has been photographed from different angles">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment3/images/SfM_result_filtered_small.png width="200" title="Resulting 3D pointcloud and camera poses of different views">
</p>


## Assignment 4: Model fitting and Multi-View Stereo (MVS)

### Model fitting using Random Sample Consensus (RANSAC)

To demonstrate the working principle of RANSAC it has been applied to the simple case of a 2D line estimation on data with outliers; the results can be seen in the following:

<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment4/images/fitting_plot.jpg width="500" title="Results without and with RANSAC of a 2D line estimation with outliers">
</p>

### Multi-View Stereo (MVS)

This describes the task of reconstructing the dense geometry of an observed scene. In this assignment, the multi-view stereo problem is solved using deep learning. The pipeline of the method is shown in the following figure: 

<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment4/images/MultiViewStereo.png width="500" title="Pipeline of the MVS method used">
</p>

First, the features are extracted for several images. Second, given uniformly distributed depth samples, source features are warped into the reference view. Third, the matching similarity between the reference view and each source view is computed. Fourth, the matching similarity from all the source views is integrated. Fifth, regularization on integrated matching similarity to output probability volume is performed. Finally, using depth regression the final depth map is obtained. On CPU, the model needs several hours for training.
This has been applied to several scenes of the [DTU MVS dataset](https://roboimagedata.compute.dtu.dk/?page_id=36). First, the ground truth (left) and estimated depth map (right) can be observed for example for a scan of a 'rabbit':

<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment4/images/train_depth_gt_scan007.jpg width="300" title="Ground truth depth map">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment4/images/train_depth_est_scan007.jpg width="300" title="Estimated depth map">
</p>

One can also observe the L1 loss between estimated depth and the groun truth converging over the 4 epochs of training in the following:
<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment4/images/mvs_training_loss.jpg width="500" title="Convergence of L1 loss over the 4 epochs of training">
</p>

Some of the resulting 3D colored point clouds using the trained model are shown in the following:

<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment4/images/scan005_mvs_point_cloud_visualization.jpg width="250" title="Colored 3D point cloud of different scans">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment4/images/scan007_mvs_point_cloud_visualization.jpg width="250" title="Colored 3D point cloud of different scans">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment4/images/scan009_mvs_point_cloud_visualization.jpg width="250" title="Colored 3D point cloud of different scans">
</p>


## Assignment 5: Object recognition

### Bag-of-words (BoW) classifier
The classifier should decide whether an image contains the back of a car (sample image on the left) or not (right):

<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment5/images/CarFromBehind.png height="170" title="Sample image containing the back of a car">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment5/images/StreetScene.png height="170" title="Sample image not containing the back of a car">
</p>

The approach of this classifier is the following: after detecting features on an image of the training dataset, they are described using the histogram of oriented gradients (HoG) descriptor and all descriptors from all images are then clustered into a visual vocabulary using k-means clustering. Each image is described using a Bag-of-words (BoW) histogram, and a test image is then classified according to its nearest neighbor in the BoW histogram space.
This rather basic approach has shown to achieve accuracies of 95% and higher on both the positive and negative samples.

### CNN-based classifier

A simplified version of the [VGG image classification network](https://arxiv.org/pdf/1409.1556.pdf) on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) has been implemented. The goal is to correctly classify images into one of 10 different classes. After training in 80 epochs, accuracies of around 85% can be achieved even with this simplified version:

<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment5/images/Training_accuracy.jpg width="500" title="Evolution of the classifying accuracy over 80 epochs of training">
</p>


## Assignment 6: Tracking

### CONDENSATION tracker

The [CONDENSATION algorithm](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated2/isard_condensation_visual.pdf) - CONDitional DENSity propagaTION over time - is a sampled-based solution of a recursive Bayesian filter applied to a tracking problem. Here, the tracker used a color histogram to follow a template in the image over several frames. For each new frame, it checks samples in a cloud of points around the previous position, weighs them according to their similarity with the template, takes the weighted average position and updates the color histogram as a convex combination between the previous and current template color histogram (which can be controlled using a parameter &alpha;).
The tracking performance can be tuned using several parameters and then applied to the simple example of following a hand in open space (left), and the more advanced examples of following a hand through an occlusion and in front of a noisy background(middle) as well as following a ball bouncing of a wall (right):

<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment6/plots/video1/plot_36.png width="250" title="Resulting trajectory of tracking a hand in open space">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment6/plots/video2/plot_35.png width="250" title="Resulting trajectory of tracking a hand through occlusion and noisy background">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment6/plots/video3/plot_45.png width="250" title="Resulting trajectory of tracking a ball bouncing off a wall">
</p>

For the second example of following a hand through an occlusion and in front of a noisy background an animated version of the tracking can be seen in the following:

<p align="center">
  <img src=https://github.com/ischollETH/computer-vision-ETH/blob/main/Assignment6/plots/Video2.gif width="500" title="Animated trajectory of tracking a hand through occlusion and noisy background">
</p>
