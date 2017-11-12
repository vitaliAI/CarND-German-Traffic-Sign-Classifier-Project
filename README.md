#**Traffic Sign Recognition** 

##Writeup Vitali Mueller

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/datainfo.jpg "Visualization"
[image3]: ./examples/grayscale.jpg "Grayscaling"
[image2]: ./examples/visualization.jpg "Visualization"
[image4]: ./new_signs/2.png "Traffic Sign 1"
[image5]: ./new_signs/6.png "Traffic Sign 2"
[image6]: ./new_signs/priority_road_12.png "Traffic Sign 3"
[image7]: ./new_signs/road_work_25.jpg "Traffic Sign 4"
[image8]: ./new_signs/road_work_25_1.jpg "Traffic Sign 5"
[image9]: ./new_signs/school.jpg "Traffic Sign 6"
[image10]: ./new_signs/speed-limit-50_2.png "Traffic Sign 7"
[image11]: ./new_signs/stop.jpg "Traffic Sign 8"
[image12]: ./new_signs/stop_2.jpg "Traffic Sign 9"
[image13]: ./examples/results.jpg "Traffic Sign 9"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First a distribushen over datasets is showen

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I was using original image format 32x32x3 and had a training result about 92%. The next step was to preprocess the image to gray scale and then apply Histogram Equalization to enhence contrast and provide more significant information to the traffic signs.

Here is an example of a traffic sign image before and after grayscaling.

Original Images:

![alt text][image2]

Preprocessed Images:

![alt text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is similar to LeNet-5 Architecture with additional Fully connected layer and also drop out is applied to avoid overfitting. The architecture consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale Image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 32x32x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x66 				|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 10x10x16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 400 Nodes  		300 Output							|
| Fully connected		Dropout applied| 300 Nodes  		120 Output							|
| Fully connected		| 120 Nodes  		84 Output					|
| Fully connected		| 84 Nodes  		43 Output					|
| Softmax			43	| etc.        									|
|						|												|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer and I had set Epochs to EPOCHS = 100 and
BATCH_SIZE = 128. Learning rate was set to rate = 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My first run did not meet the accuracy requirement of 0.93 and I had to improve my Architecture by adding another Fully Connected layer to my Network. This change plus adding dropout I was able to reach the requirement of 0.93. Results are depicted in the image below. Also as we can see we have an overfitting problem. Still some work could be done to decrease overfitting. Adding more dropout to other layers.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 93.2%
* test set accuracy of 93.8%

![alt text][image13]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

LeNet-5 modified was chosen. 

* What were some problems with the initial architecture?

Overfitting but overall was a good architure to work with

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I have added additional Fully Connected Layer and a dropout
* Which parameters were tuned? How were they adjusted and why?
Batch size were achanged and also #Nodes were changed
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Dropout helps in order to reduce overfitting. Plus it makes the Network more stabil in terms of redundancy. Also Convolutional helps to take neighbor pixel into consideration and account them for the next pixel

If a well known architecture was chosen:
* What architecture was chosen?
LeNet-5
* Why did you believe it would be relevant to the traffic sign application?
Because it was proven architecture for digit recognition. Traffic recognition is part of character and image pattern recognition which would work too
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
It has relative high accuracy. Obviously it could be trained further. More Nodes can be added. Trying AlexNet or ResNets which should work good too for this task.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11] ![alt text][image12]



####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h       		| 30 km/h   									| 
| Turn Left     		| Turn Left										|
| Priority road			| Priority road									|
| Road work	      		| Road work 					 				|
| Road work	      		| Road work 					 				|
| Children crossing		| No passing         							|
| 50 km/h       		| Turn Right    								| 
| Stop Sign      		| Priority road									| 
| Stop Sign      		| Stop sign   				

The model was able to correctly guess 2 of the 9 traffic signs, which gives an accuracy of 20%. This compares not in favor to the accuracy on the test set. That would indicating strongly overfitting and also it could be an issue that i haven't used data augmentation in order to train traffic sign from a different angle. Images provided had different size and through resizing of the images the quality reduced significantly. Even for myself it is difficult to recognise the image correctly. That wouldn't mean that the CNN i have trained is not working, provided data information is just not good enough.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


