# AI-based-indian-license-plate-detection


Approach:
We need to build a system that is capable of-
Taking in the image/video (series of images) from surrounding:
at the hardware end, we need a pc (or raspberry pi) along with a camera and at the software end, we need a library to capture and process the data (image). I’ve used OpenCV (4.1.0) and Python (3.6.7) for this project.
Looking for a license plate in the image:
To detect an object(license plate) from an image we need another tool that can recognize an Indian license plate so for that I’ve used Haar cascade, pre-trained on Indian license plates (will be updating soon to YOLO v3).
Analyzing and performing some image processing on the License plate:
Using OpenCV’s grayscale, threshold, erode, dilate, contour detection and by some parameter tuning, we may easily be able to generate enough information about the plate to decide if the data is useful enough to be passed on to further processes or not (sometime if the image is very distorted or not proper, we may only get suppose 8 out of 10 characters, then there’s no point passing the data down the pipeline but to ignore it and look at the next frame for the plate), also before passing the image to the next process we need to make sure that it is noise-free and processed.
Segmenting the alphanumeric characters from the license plate:
if everything in the above steps works fine, we should be ready to extract the characters from the plate, this can be done by thresholding, eroding, dilating and blurring the image skillfully such that at the end the image we have is almost noise-free and easy for further functions to work on. We now again use contour detection and some parameter tuning to extract the characters.
Considering the characters one by one, recognizing the characters, concatenating the results and giving out the plate number as a string:
Now comes the fun part! Since we have all the characters, we need to pass the characters one by one into our trained model, and it should recognize the characters and voilà! We’ll be using Keras for our Convolutional Neural Network model.
Prerequisites:
OpenCV: OpenCV is a library of programming functions mainly aimed at real-time computer vision plus its open-source, fun to work with and my personal favorite. I have used version 4.1.0 for this project.
Python: aka swiss army knife of coding. I have used version 3.6.7 here.
IDE: I’ll be using Jupyter here.
Haar cascade: It is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of ​​ features proposed by Paul Viola and Michael Jones in their paper “Rapid Object Detection using a Boosted Cascade of Simple Features” in 2001. More info
Keras: Easy to use and widely supported, Keras makes deep learning about as simple as deep learning can be.
Scikit-Learn: It is a free software machine learning library for the Python programming language.
And of course, do not forget the coffee!
Step 1
Creating a workspace.
I recommend making a conda environment because it makes project management much easier. Please follow the instructions in this link for installing miniconda. Once installed open cmd/terminal and create an environment using-
conda create -n 'name_of_the_environment' python=3.6.7
Now let’s activate the environment:
conda activate 'name_of_the_environment'
This should set us inside our virtual environment. Time to install some libraries-
# installing OpenCV
pip install opencv-python==4.1.0
# Installing Keras
pip install keras
# Installing Jupyter
pip install jupyter
#Installing Scikit-Learn
pip install scikit-learn
Step 2
Setting up the environment!
We’ll start with running jupyter notebook and then importing necessary libraries in our case OpenCV, Keras and sklearn.
# in your conda environment run
jupyter notebook
This should open Jupyter notebook in the default web browser. 
Once open, let’s import the libraries
#importing openCV
import cv2
#importing numpy
import numpy as np
#importing keras and sub-libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
Step 3
Number plate detection:
Let’s start simple by importing a sample image of a car with a license plate and define some functions:

The above function works by taking image as input, then applying ‘haar cascade’ that is pre-trained to detect Indian license plates, here the parameter scaleFactor stands for a value by which input image can be scaled for better detection of license plate (know more). minNeighbors is just a parameter to reduce false positives, if this value is low, the algorithm may be more prone to giving a misrecognized outputs. (you can download the haar cascade file as ‘indian_license_plate.xml’ file from my github profile.)

input image

output image with detected plate highlighted

output image of detected license plate
Step 4
Performing some image processing on the License plate.
Now let’s process this image further to make the character extraction process easy. We’ll start with defining some more functions for that.

The above function takes in image as input and performs the following operation on it-
resizes it to a dimension such that all characters seem distinct and clear
convert the colored image to a grey scaled image i.e instead of 3 channels (BGR), the image only has a single 8-bit channel with values ranging from 0–255 where 0 corresponds to black and 255 corresponds to white. We do this to prepare the image for the next process.
now the threshold function converts the grey scaled image to a binary image i.e each pixel will now have a value of 0 or 1 where 0 corresponds to black and 1 corresponds to white. It is done by applying a threshold that has a value between 0 and 255, here the value is 200 which means in the grayscaled image for pixels having value above 200, in the new binary image that pixel will be given a value of 1. And for pixels having value below 200, in the new binary image that pixel will be given a value of 0.
The image is now in binary form and ready for the next process Eroding.
Eroding is a simple process used for removing unwanted pixels from the object’s boundary meaning pixels that should have a value of 0 but are having a value of 1. It works by considering each pixel in the image one by one and then considering the pixel’s neighbor (the number of neighbors depends on the kernel size), the pixel is given a value 1 only if all its neighboring pixels are 1, otherwise it is given a value of 0.
The image is now clean and free of boundary noise, we will now dilate the image to fill up the absent pixels meaning pixels that should have a value of 1 but are having value 0. The function works similar to eroding but with a little catch, it works by considering each pixel in the image one by one and then considering the pixel’s neighbor (the number of neighbors depends on the kernel size), the pixel is given a value 1 if at least one of its neighboring pixels is 1.
The next step now is to make the boundaries of the image white. This is to remove any out of the frame pixel in case it is present.
Next, we define a list of dimensions that contains 4 values with which we’ll be comparing the character’s dimensions for filtering out the required characters.
Through the above processes, we have reduced our image to a processed binary image and we are ready to pass this image for character extraction.
Step 5
Segmenting the alphanumeric characters from the license plate.

After step 4 we should have a clean binary image to work on. In this step, we will be applying some more image processing to extract the individual characters from the license plate. The steps involved will be-
Finding all the contours in the input image. The function cv2.findContours returns all the contours it finds in the image. Contours can be explained simply as a curve joining all the continuous points (along the boundary), having the same color or intensity.

https://www.oipapio.com/static-img/4698620190220123940948.jpg

plate with contours drawn in green
After finding all the contours we consider them one by one and calculate the dimension of their respective bounding rectangle. Now consider bounding rectangle is the smallest rectangle possible that contains the contour. Let me illustrate the bounding rectangle by drawing them for each character here.

Since we have the dimensions of these bounding rectangle, all we need to do is do some parameter tuning and filter out the required rectangle containing required characters. For this, we will be performing some dimension comparison by accepting only those rectangle that has a width in a range of 0, (length of the pic)/(number of characters) and length in a range of (width of the pic)/2, 4*(width of the pic)/5. If everything works well we should have all the characters extracted as binary images.

The binary images of 10 extracted characters.
The characters may be unsorted but don’t worry, the last few lines of the code take care of that. It sorts the character according to the position of their bounding rectangle from the left boundary of the plate.
Step 6
Creating a Machine Learning model and training it for the characters.
The data is all clean and ready, now it’s time do create a Neural Network that will be intelligent enough to recognize the characters after training.

https://mesin-belajar.blogspot.com/2016/05/topological-visualisation-of.html
For modeling, we will be using a Convolutional Neural Network with 3 layers.
## create model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.4))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=37, activation='softmax'))
To keep the model simple, we’ll start by creating a sequential object.
The first layer will be a convolutional layer with 32 output filters, a convolution window of size (5,5), and ‘Relu’ as activation function.


Next, we’ll be adding a max-pooling layer with a window size of (2,2).
Max pooling is a sample-based discretization process. The objective is to down-sample an input representation (image, hidden-layer output matrix, etc.), reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned.

max-pooling layer
Now, we will be adding some dropout rate to take care of overfitting.
Dropout is a regularization hyperparameter initialized to prevent Neural Networks from Overfitting. Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. We have chosen a dropout rate of 0.4 meaning 60% of the node will be retained.
Now it’s time to flatten the node data so we add a flatten data for that. Flatten layer takes data from the previous and represents it in a single dimension.

Finally, we will be adding 2 dense layers, one with the dimensionality of the output space as 128, activation function=’relu’ and other, our final layer with 37 outputs for categorizing the 26 alphabets (A-Z) + 10 digits (0–9) and activation function=’ softmax’
Step 7
Training our CNN model.
The data we will be using contains images of alphabets (A-Z) and digits (0–9) of size 28x28x1, also the data is balanced so we won’t have to do any kind of data tuning here.
I have created a CSV file with pixel values of each image as features and their respective labels. To import the data we need to run the code below.
# visit my github for the .csv file.
data = pd.read_csv('image_data.csv') #reading the csv file
X = []
Y = data['y'] #storing the labels in Y
del data['y']
for i in range(data.shape[0]): #iterating over all the rows.
    flat_pixels = data.iloc[i].values[1:] #extracting pixel values
    image = np.reshape(flat_pixels, (28,28)) #reshaping to 28x28p
    X.append(image) #adding to input feature list
Time to split the data for training and testing, One-hot encoding the output classes and reshaping the train/test data.
# split the data into training (50%) and testing (50%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.30, random_state=seed)
# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
#reshaping data
X_train = X_train.reshape(-1,28,28,1)
X_test  = X_test.reshape(-1,28,28,1)
It’s time to train our model now!
we will use ‘categorical_crossentropy’ as loss function, ‘Adam’ as optimization function and ‘Accuracy’ as our error matrix.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#performing fit
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test,Y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
After training for 50 epochs, our model achieved an accuracy of 99.45%
and CNN Error of 0.43%

Step 8
The output.
Finally, its time to test our model, remember the binary images of extracted characters from number plate? Let’s feed the images to our model!
