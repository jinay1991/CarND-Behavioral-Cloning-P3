# **Behavioral Cloning**

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_net.png "Model Visualization"
[image2]: ./examples/center_2018_03_24_19_56_26_549.jpg "center image"
[image3]: ./examples/left_2018_03_24_20_38_58_218.jpg "Recovery Image"
[image4]: ./examples/center_2018_03_24_20_38_58_218.jpg "Recovery Image"
[image5]: ./examples/right_2018_03_24_20_38_58_218.jpg "Recovery Image"
[image6]: ./examples/left_2018_03_24_20_38_58_218.jpg "Normal Image"
[image7]: ./examples/left_2018_03_24_20_38_58_218_fliped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 100 (model.py lines 80-98). The model was first represented by NVIDIA for Controlling Steering angle and made open source [Link]('https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf').

The model includes RELU layers to introduce nonlinearity (code line 80-98), and the data is normalized in the model using a Keras lambda layer (code line 81).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 80-98).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 84). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture which can predict steering angle by seeing center, left and right camera images.

My first step was to use a convolution neural network model similar to the Nvidia Net. I thought this model might be appropriate because it converges well for regression problem like this here predicting steering angle.

In order to gauge how well the model was working, I split my images and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting, although that model did succeed by keeping vehicle on track, but it was overfitted.

To combat the overfitting, I modified the model so that it does not overfit. For this I have added Dropout layers after each Convolution Layer, details described in next section.

Then I retrained with same dataset but found that model and try running it again on track one of simulator and found that it did not work well.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track specifically when vehicle requires to take sharp turn and staying away from the track edges. To improve the driving behavior in these cases, I re-collected data for specific cases where vehicle was falling off the track, as part of giving information of how to correctly make a turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 80-98) consisted of a convolution neural network with the following layers and layer sizes.

| Layer (type)                  | Output Shape        | Param # |
| ----------------------------- | ------------------- | ------- |
| lambda_1 (Lambda)             | (None, 160, 320, 3) | 0       |
| cropping2d_1 (Cropping2D)     | (None, 65, 320, 3)  | 0       |
| conv2d_1 (Conv2D)             | (None, 31, 158, 24) | 1824    |
| dropout_1 (Dropout)           | (None, 31, 158, 24) | 0       |
| conv2d_2 (Conv2D)             | (None, 14, 77, 36)  | 21636   |
| dropout_2 (Dropout)           | (None, 14, 77, 36)  | 0       |
| conv2d_3 (Conv2D)             | (None, 5, 37, 48)   | 43248   |
| dropout_3 (Dropout)           | (None, 5, 37, 48)   | 0       |
| conv2d_4 (Conv2D)             | (None, 3, 35, 64)   | 27712   |
| dropout_4 (Dropout)           | (None, 3, 35, 64)   | 0       |
| conv2d_5 (Conv2D)             | (None, 1, 33, 64)   | 36928   |
| dropout_5 (Dropout)           | (None, 1, 33, 64)   | 0       |
| flatten_1 (Flatten)           | (None, 2112)        | 0       |
| dense_1 (Dense)               | (None, 100)         | 211300  |
| dense_2 (Dense)               | (None, 50)          | 5050    |
| dense_3 (Dense)               | (None, 10)          | 510     |
| dense_4 (Dense)               | (None, 1)           | 11      |
|                               |                     |         |
| **Total params:** 348,219     |                     |         |
| **Trainable params:** 348,219 |                     |         |
| **Non-trainable params:** 0   |                     |         |

And it contains 348,219 trainable params.

I have used *NVIDIA Quadro M1200 4GB* for training my model for *4461 data points* and *348,219 trainable params* and it took me around *20 minutes* for completing 2 epochs.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the edges of track. These images show what a recovery looks like starting from left (left, center and right):

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on same track but in reverse

To augment the data sat, I also flipped images and angles thinking that this would help generate more data and avoid model to be biased to only left turns as track one consist mostly left turns. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

For slightly better learning, I have corrected steering angle for left and right camera images by 0.2 and -0.2, respectively for allowing model to correct steering angle more promptly.

After the collection process, I had 4461 number of data points. I then preprocessed this data by cropping image as specified in classroom so that only road part is visible. For this I clipped top 70 rows (background) and bottom 25 rows (car) to help model learn better.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by me. I used an adam optimizer so that manually training the learning rate wasn't necessary.


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.