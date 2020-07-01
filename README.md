## **Traffic Sign Recognition with TensorFlow and SageMaker**
---
### Intro
This project trains a TensorFlow model in SageMaker for traffic sign recognition.   

The files contained in this repo includes:
* readme.md
* Traffic_sign_recognition_with_TensorFlow.ipynb
* train.py

Run Traffic_sign_recognition_with_TensorFlow.ipynb in SageMaker as a notebook instance.

### Dataset
Necessary files not included in this repo can be downloaded from Kaggle: [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

Here is an overview of the dataset:
* Each image has slightly different sizes
* Number of training examples = 39209
* Number of testing examples = 12630
* Number of classes = 43

### Model
The network architecture is similar to LeNet and the details of each layer is as follows:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image  			    		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 		     		|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 		     		|
| Fully connected		| Input = 400. Output = 120.        			|
| Fully connected		| Input = 120. Output = 84.        			|
| Dropout	          	|         			|
| Fully connected		| Input = 84. Output = 43.        			|
| Softmax				| Output layer      							|

See complete tutorial [here](https://medium.com/@junma/train-a-tensorflow-model-in-amazon-sagemaker-e2df9b036a8?sk=96dfa2af2602c5440359e44b16ad2f18)
