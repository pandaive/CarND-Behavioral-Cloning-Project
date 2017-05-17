# **Behavioral Cloning** 

## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2017_05_15_19_00_57_782.jpg "steer_right1"
[image2]: ./examples/center_2017_05_15_19_00_58_840.jpg "steer_right2"
[image3]: ./examples/center_2017_05_15_19_00_59_055.jpg "steer_right3"
[image4]: ./examples/center_2017_05_15_19_00_59_348.jpg "steer_right4"
[image5]: ./examples/center_2017_05_15_19_01_23_941.jpg "steer_right5"
[image6]: ./examples/center_2017_05_15_19_01_24_180.jpg "steer_right6"
[image7]: ./examples/center_2017_05_15_19_01_24_577.jpg "steer_right7"
[image8]: ./examples/center_2017_05_15_19_01_25_316.jpg "steer_right8"
[image9]: ./examples/center_2017_05_15_19_02_06_307.jpg "steer_left1"
[image10]: ./examples/center_2017_05_15_19_02_06_593.jpg "steer_left2"
[image11]: ./examples/center_2017_05_15_19_02_07_368.jpg "steer_left3"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* model_with_generator.py containing the script using a generator, that I wasn't using throughout training (explained later)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
I didn't use recommended python generator, as on my PC it was really slow. I didn't figure out what was the reason, I discussed the problem with my mentor and shared my model with him, he ran it without any problems, but for me it was taking much more time than with usual approach. Because of that I wasn't using it. I provided additional file, *model_with_generator.py* to show that I knew how to use it.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 2 lambda layers, a cropping layer, 3 convolutional neural networks with two of them followed by max pooling layer, and 3 fully connected layers. I started from recommended Nvidia End-To-End model and modified it so it matches dimensions of my input data. I also added Gaussian Noise to avoid overfitting.

| Layer | Description |
| --- | --- |
| Lambda 1 | Resize images from 160x320 to 80x160 |
| Lambda 2 | Normalization |
| Cropping | by 35 from top, 12 from bottom |
| Gaussian Noise | With rate 0.2 to avoid overfitting |
| Convolution | 24x5x5, activation: RELU |
| Max Pooling | Default values |
| Convolution | 48x5x5, activation: RELU |
| Max Pooling | Default values |
| Convolution | 64x5x5, activation: RELU |
| Fully connected | Input: 2112 Output 100, activation: RELU |
| Fully connected | Input: 100 Output 50, activation: RELU |
| Fully connected | Input: 50 Output 1 |


#### 2. Attempts to reduce overfitting in the model

The model contains Gaussian Noise layer in order to reduce overfitting (model.py lines 73). 

The model was trained and validated on dataset provided by Udacity and also by data gathered by me. Second dataset included smooth drive through the track as well as examples with driving from outside the track, examples:


The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 84).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. As stated in #2 I recorded smooth drive through the track (usual direction as well as the other), recovering from outside the road, recovery from driving on a lane and recovery from driving on the sides of the bridge.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to focus on very good dataset, then start from LeNet, try my network from P2 and finally try recommended Nvidia network and ajdust it to my model. I wanted to use LeNet and my network from P2 because I was already familiar with them and I could easily play with the parameters to see what should be suitable for this project. Well, those networks were doing quite well, but the car was unstable on tests and sometimes drove into lane line. At the end, I decided that Nvidia network will be the best to start.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I started with producing more data, driving from outside the road, etc. I was also playing with dropout and regularization, but it turned out, that simple Gaussian Noise applied to all images plus single Dropout after first fully connected layer did their job, and low error on training set implied low error on validation set and at the end - good results on testing.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 69-82) was already described in #1.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving and one lap in opposite direction.
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from steering away from the center. These images show what a recovery looks like:

##### Recovery 1
![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]

##### Recovery 2
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

##### Recovery 3
![alt text][image9]
![alt text][image10]
![alt text][image11]

As at the beginning my car tended to steer away to one side, I also used recommended flipping of images, for those of which measurement is higher than 0.2 (or lower than -0.2) to provide data for learning how to handle curves.

I also used images from right and left cameras for the steering in some direction situations, to provide more data for recovering from getting to close to the lane line.

After the collection process, I had almost 20k number of data points (15k for training and 4k for validation). The only preprocessing I did was normalization, resize (to half its size) and cropping. I assumed that in this case other preprocessing won't be necessary. Also the Gaussian Noise layer was used for preprocessing for avoiding overfitting.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the fact, that after that the network was usually overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
