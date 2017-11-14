
# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image10]: ./examples/car_not_car_1.png
[image11]: ./examples/hog_feature_car_1.png
[image12]: ./examples/hog_feature_noncar_1.png
[image13]: ./examples/sliding_windows_1.jpg
[image14]: ./examples/sliding_windows_res.jpg
[image15]: ./examples/test_res_1_small.png
[image16]: ./examples/test_res_3_small.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is implemented in the _feature_extractor.py_. I focused on extracting HOG features from an image and omit investigation of spatially binned color and color histograms.
The implementation is provided in _HogFeature_ class. Due 3 parameters:
* orientation (number of orientation bins)
* pix_per_cell (size (in pixels) of a cell)
* cells_per_block (Number of cells in each block)

calculation of HOG can be controlled.

This class is used by training of classifier and later by prediction of cars in project video. Both methods : `extract_features_for_image_list` and `extract_features` are used by training of classifier. It is possible to define channel number for that the histogram will be calculated. Setting the parameter to "ALL" value will result in calculation of histogram for all channels. The  `get_features` method is used by detection of cars.
All of these method use `skimage.feature hog()` function required  all parameters I listed above. Interesting of explanation of HOG can be found [here](http://vision.stanford.edu/teaching/cs231b_spring1213/slides/hog_rafael_tao.pdf)

From the data provided in the project - it means the `vehicle` and `non-vehicle` images I choose one sample representing vehicle and not vehicle:

![alt text][image10]

Using parametrization `orientation=11`, `pix_per_cell=16` and `cells_per_block=2` I tested my implementation for two color spaces : RGB and YUV for every channel separately.

The result of HOG extracting in YUV color space for Y, U and V channel is shown bellow:

* vehicle

![alt text][image11]

* non vehicle

![alt text][image12]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters like color space (RGB and YUV) and `orientation` (11 or 9) and  `pix_per_cell` (16 or 8) by training of classifiers.
The statistic for following combination is shown below:

* color space = YUV, orientation=11, pix_per_cell=16
```
Training time:  43.7
Best params  :  {'kernel': 'rbf'}
Best score  :  0.999136655155
Test accuracy:  1.0
```

* color space = RGB, orientation=11, pix_per_cell=16
```
Training time:  52.72
Best params  :  {'kernel': 'rbf'}
Best score  :  0.988529847065
Test accuracy:  0.987179487179
```

* color space = YUV, orientation=9, pix_per_cell=8
```
Training time:  42.09
Best params  :  {'kernel': 'rbf'}
Best score  :  0.998149975333
Test accuracy:  0.999013806706
```

Based on this statistic I decided to use YUV color space and orientation=11, pix_per_cell=16.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The class `Classifier` in file classifier.py provides implementation of SVM classifier. To avoid training of classifier every time I launch program I implemented functionality to load and save classifier to a file.
I have read that it's been shown that the linear kernel is a degenerate version of `rbf` (aka Gaussian), hence the linear kernel is never more accurate than a properly tuned rbf kernel.
I decided to use not-linear classifier. I applied sklearn.model_selection.GridSearchCV to find the optimal parameters for training the SVM for this kernel with default parameters.
The statistic I achieved for training data gave me a hope that my choice is a good. Let's how it will work by vehicle detection.

As suggested in the project I did following steps:
* ensured that training data are balanced, it means have as many positive as negative examples. I reduced the number of vehicle and non-vehicle images to common number
* random shuffling of the data
* splitting the data into a training and testing set (20% of testing set and 80% for training set)
* normalization of features vectors by using of `sklearn.preprocessing.StandardScaler`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for the sliding window search is located in the `vehicle_detection.py` file. Here I implemented a class `VehicleDetector` that provide a pipeline for detecting of vehicles in an image.
Two methods : `dectect_car_boxes` and `find_car_boxes` are relevant here. The "find_car_boxes" is an adaptation of "find_cars" from the lesson materials.

The method combines HOG feature extraction with a sliding window search. It extracts the features from the image once, goes over the different blocks extracted from the image,
 scales the features for this block, runs the prediction on this block and if the prediction is positive adds a bounding box of it to result list of bounding boxes.
One of the parameters used in this method defined a window for that prediction must be done. Each window is defined by a scaling factor and y coordinate min and max values.

I tried some configurations of window sizes and positions, with various overlaps in the X and Y directions. The following image shows the configurations of search windows for small (1x), medium (1.5x, 2x), and large (3x) windows:

![alt text][image13]

The image below shows the boundix boxes returned by find_car_boxes drawn onto "test1.jpg" image.

![alt text][image14]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The implementation of my pipeline is provided in `VehicleDetector` class in the method `run' in the vehicle_detection.py file. Using of sliding window search concept and HOG feature extraction for YUV color space for all channels I achieve following result represented in the images below:

* Frame 1

![alt text][image15]

* Frame 2

![alt text][image16]

The images represent a two frames from "test_video.mpg".

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/GosiaP/CarND-Vehicle-Detection/blob/master/output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The implementation of filtering of false positives is done in `HeatMap` class in "vehicle_detection.py" file.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.
I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

As suggested in lesson materials it makes sense to integrate a heat map over several frames of video what I did (for 10 video frames).
The result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` are represented by images above : Frame 1 and Frame 2 (Point 2).


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First of all I didn't go approach to train my classifier using spatially binned color and color histograms (lack of time). I really don't know if this would provide much better classifier but this approach was suggested in the lesson material.

As you can see on the output project_video there are some frames where the false positives were not filtered correctly like e.g a frame where a road railings are visible on the right side.
I supposed that my classifier was not optimal even if the statistic, I provided a the begin of the projects, showed :-(

Even so I made compromises regarding the number of scales and the window overlap and even I used Hog sub-sampling window search approach to calculate the HOGs only once time, the algorithm is still taking about 1s per frame which disqualifies it for real time applications.

The pipeline is probably most likely to fail in cases where vehicles don't resemble those in the training data set, but lighting and any environment conditions. They might also play a role e.g. a white car on a white background or black car in the shadow.




