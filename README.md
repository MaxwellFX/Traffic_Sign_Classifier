# **Traffic Sign Recognition** 



[//]: # (Image References)

[image1]: ./writeUpImages/HistogramForOriginalTrainingSet.png "Histogram for training set"
[image2]: ./writeUpImages/original_vs_grascale_normalized.png "Before vs after grayscale normalization"
[image3]: ./writeUpImages/original_vs_randome_augmentation.png "Image augmentation"
[image4]: ./writeUpImages/HistogramForAugmentedDataSet.png "Histogram of augemented dataset"
[image5]: ./writeUpImages/valida_accuracy_vs_epochs.png "Validation accuracy vs Epochs"
[image6]: ./writeUpImages/OnlineImages.png "images found from the web"
[image7]: ./writeUpImages/predictions.png "Image prediction"
[image8]: ./writeUpImages/placeholder.png "Traffic Sign 5"


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an histogram of original training dataset:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)



Here is an example of a traffic sign image before and after grayscaling normalization.

![alt text][image2]

I normalized the image because having the pixel value ranges widely from 0 to 255 makes it difficult for single learning rate to train

Also, as we can observe from the histogram, some traffic sign class have much smaller sample size than average. Have such training setup would make the model skew towards other class that has more samples, making the predict for classes with fewer samples inaccurate. To improve on such matter, we can perform data augmentation to artificially generate more data for classes that have fewer samples. 

Extra samples for classes with fewer sample counts are generated through following steps:
1. All training images will be normalized to grayscale image, (pixel - 128)/ 128, before any data augmentation starts
2. Number of samples for each class are measured against the average samples for all sign classes
3. If the sample count for current class is fewer than average, extra sign images will be artificially generated until the total sample size (existing + new) is no less than the average sample size
4. Each aritificially generated image are randomed applied with image warp, zoom, translation and brightness adjustments

Here is an example of an original image and an augmented image:

![alt text][image3]

Here is an histogram of augmented training dataset:

![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale normalized image   			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride,  outputs 14x14x6 			        |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride,  outputs 5x5x16 			        |
| Flatten				| outputs 400									|
| Fully connected		| outputs 120        							|
| RELU					|												|
| Dropout 				| keep_prob = 0.5								|
| Fully connected		| outputs 84        							|
| RELU					|												|
| Dropout 				| keep_prob = 0.5								|
| Fully connected		| outputs 43        							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Here are the model parameters:

| Parameters     | value	         | 
|:--------------:|:-----------------:| 
| Optimizer      | adam optimizer    | 
| Input     	 | 32 x 32 x 1 image |
| Dropout		 | 0.5				 |
| Epochs	     | 50 			     |
| Batch Size	 | 128               |
| learning rate	 | 0.0009			 |
| mu	         | 0 			     |
| sigma			 | 0.1				 |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of around 97% 
* test set accuracy of around 94.6%

I simply adopted LeNet architecture as it performs reasonably well with my augmented data sets. After trying it out, the validation accuracy kept increase as each epoch was trained, thus I believe the 'classic" solution is sufficient for this project. Here is the graph of validation accuracy against each epoch:

![alt text][image5]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the web:

![alt text][image6]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


The model was able to correctly guess 9 to 10 of the 10 traffic signs, which gives an accuracy of 90 - 100%. However, if the traffic sign has complicated graphical features, it is expected to have lower accuracy as the insanely low resolution coupled with dim lightly would be very difficult to recognize even to human eyes

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook. 

Here are the results of the prediction:

![alt text][image7]
