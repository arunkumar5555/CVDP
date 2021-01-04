# DEEP LEARNING FOR IMAGE MATCHING

#### In this repository shows how to match your selfies with the passportsize photo of you.
## What is Deep Learning ?

 ### Deep learning is an AI function that mimics the workings of the human brain in processing data for use in detecting objects, recognizing speech, translating languages, and making decisions. Deep learning AI is able to learn without human supervision, drawing from data that is both unstructured and unlabeled.
 
 ## STEPS
 
 ### 1. Download the Dataset [click]()
 ### 2. preprocessing the dataset
 ### 3. Defining the Model 
 ### 4. Defining the Triplet Loss Function
 ### 5. start  Training the model
 ### 6. Implementation of the model
 
 
## 1. Download the Dataset

### open google [colab](https://colab.research.google.com) to use free GPU

#### mount drive to use your drive

from google.colab import drive
drive.mount('/content/drive')

#### read the folder to download your input file

!cd '/content/drive/MyDrive/Colab Notebooks/Deep_Learning_ComputerVision/input'

#### mention the link to download the training dataset
!gdown --id "12_WTFi9ppvD-loaWUWpUar25Z3nT5k9P"
#### unzip training dataset
!unzip '/content/drive/MyDrive/Colab Notebooks/DLCV/input/trainset.zip'
## 2. preprocessing the dataset

#### In this step understand the dataset AND PREPARE the dataset for trainng purpose.
### Import the required libraries
#### cv2
#### numpy
#### keras
#### tensorflow etc

### defining the path to the dataset folder and face feature of harrcascade file
### Defining required variables
### preprocess the dataset .understand the script(passportsize image),selfies
###count number of employee and their script and selfi images

## 3. Defining the Model
### What is one shot Learning?


### create model using CNN Architecture
### compile it and save the architecture

plot_model(model,to_file='/content/drive/MyDrive/Colab Notebooks/DLCV/output/Inception_one_shot.png')


## 4. Defining the Triplet Loss Function
### Defining the Triplet Loss Function
### Function to resize the image to match the input shape of the model
### Definig the generator
###  Defining model for triplet loss
### compile model
### summarize themodel
### plot the model Architecture
### 
## 5. start  Training the model
### We will be training the model for 5 epoch and with steps_per_epoch as 100 .These hyperparameters can be changed as per the availablity of computional power
### Since batch size is 32 and total number of samples is almost 3200 so steps_per_epoch=100

triplet_model.fit(data_gen(),steps_per_epoch=100,epochs=5)


Defining model for triplet loss

Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_5 (InputLayer)            [(None, 3, 96, 96)]  0                                            
__________________________________________________________________________________________________
input_7 (InputLayer)            [(None, 3, 96, 96)]  0                                            
__________________________________________________________________________________________________
input_6 (InputLayer)            [(None, 3, 96, 96)]  0                                            
__________________________________________________________________________________________________
FaceRecognotionModel (Functiona (None, 128)          3743280     input_5[0][0]                    
                                                                 input_7[0][0]                    
                                                                 input_6[0][0]                    
__________________________________________________________________________________________________
concatenate_8 (Concatenate)     (None, 384)          0           FaceRecognotionModel[0][0]       
                                                                 FaceRecognotionModel[1][0]       
                                                                 FaceRecognotionModel[2][0]       
==================================================================================================
Total params: 3,743,280
Trainable params: 3,733,968
Non-trainable params: 9,312
__________________________________________________________________________________________________

Training the model

Epoch 1/5
100/100 [==============================] - 2106s 21s/step - loss: 0.5245
Epoch 2/5
100/100 [==============================] - 2032s 20s/step - loss: 0.4066
Epoch 3/5
100/100 [==============================] - 2050s 20s/step - loss: 0.3638
Epoch 4/5
100/100 [==============================] - 2102s 21s/step - loss: 0.3445
Epoch 5/5
100/100 [==============================] - 2071s 21s/step - loss: 0.3096

<tensorflow.python.keras.callbacks.History at 0x7f75a206cac8>

Implementation of the model

Function to preprocess the image according to the model requirements.

It uses haar cascade to detect the face and crops the face to remove the unwanted noise from the image and then resize it to (96,96).

### save the model
### load the model for testing

## 6. Implementation of the model
### Function to preprocess the image according to the model requirements.
### It uses haar cascade to detect the face and crops the face to remove the unwanted noise from the image and then resize it to (96,96).      

## resize the image
## Function to convert the image to embeddings
## Function to calculate the distance between the embeddings and confidence score
## Testing

### matching the script image to selfies
