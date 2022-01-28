# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:58:44 2022

@author: Ribert
"""

from keras_preprocessing.image import ImageDataGenerator

import pandas as pd
import cv2


import splitfolders


path = 'C:/Users/Ribert/OneDrive/Kandidat/3 semester/DS807 Anvendt Maskinlæring/Exam/'

df = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["nr","year"])
#print(df)

"""
input_folder = path + 'DIDA_12000_String_Digit_Images/'

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#Train, val, test
splitfolders.ratio(input_folder, output= path + 'cell_image', 
                   seed=42, ratio=(.7, .2, .1), 
                   group_prefix=None) # default values
"""



def append_ext(fn):
    return str(fn) + ".jpg"

df["nr"]=df["nr"].apply(append_ext)


def change_to_cat_CC(ye):

    x = str(ye)   
    if len(x) == 4:
        if x[0] == "1" and x[1] == "8":
            return str(0) # Zero is for 18
        else:
            return str(1)
    return str(1) # One is for not 18

df["CC"]=df["year"].apply(change_to_cat_CC)

print(df)



datagen=ImageDataGenerator(
    rescale=1./255.,
    rotation_range=5, # data agumentation -> 	Int. Degree range for random rotations. #Kopieret
    zoom_range=0.2, # data agumentation -> If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]. #Kopiere fra https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    horizontal_flip=True, # data agumentation -> Boolean. Randomly flip inputs horizontally. #Kopieret
    vertical_flip=False
    )

datagen_test=ImageDataGenerator(rescale=1./255.)


train_generator_CC=datagen.flow_from_dataframe(
dataframe=df,
directory=path + "cell_image/train/DIDA_1/",
x_col="nr",
y_col="CC",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(150,150))



valid_generator_CC=datagen_test.flow_from_dataframe( #maybe use augmentation on val also?
dataframe=df,
directory= path + "cell_image/val/DIDA_1/",
x_col="nr",
y_col="CC",
#subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(150,150))



test_generator_CC=datagen_test.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/test/DIDA_1/",
x_col="nr",
y_col="CC",
batch_size=32,
seed=42,
shuffle=False,
class_mode="categorical",
target_size=(150,150))



#%%

df_2 = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["nr","year"])

def change_to_cat_D(ye): # maybe change name

    x = str(ye)
    
    if len(x) != 4:
        return str(10)
    else:
        return str(x[2])       
            


df["D"]=df_2["year"].apply(change_to_cat_D)
print(df)


print(df["D"].value_counts()) # how many are there in each category





train_generator_D=datagen.flow_from_dataframe(
dataframe=df,
directory=path + "cell_image/train/DIDA_1/",
x_col="nr",
y_col="D",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(32,32))


valid_generator_D=datagen.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/val/DIDA_1/",
x_col="nr",
y_col="D",
#subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(32,32))



test_generator_D=datagen_test.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/test/DIDA_1/",
x_col="nr",
y_col="D",
batch_size=32,
seed=42,
shuffle=False,
class_mode="categorical",
target_size=(32,32))







#%%

df_3 = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["nr","year"])


def change_to_cat_Y(ye): # maybe change name

    x = str(ye)
    
    if len(x) != 4:
        return str(10)
    else:
        return str(x[3])       
            


df["Y"]=df_3["year"].apply(change_to_cat_Y)
print(df)


print(df["Y"].value_counts()) # how many are there in each category



train_generator_Y=datagen.flow_from_dataframe(
dataframe=df,
directory=path + "cell_image/train/DIDA_1/",
x_col="nr",
y_col="Y",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(150,150))


valid_generator_Y=datagen.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/val/DIDA_1/",
x_col="nr",
y_col="Y",
#subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(150,150))



test_generator_Y=datagen_test.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/test/DIDA_1/",
x_col="nr",
y_col="Y",
batch_size=1,
seed=42,
shuffle=False,
class_mode="categorical",
target_size=(150,150))



#%%
import os, sys
import re
def sorted_alphanumeric(data): # https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)




#%%

import numpy as np
from os import listdir



#from sklearn.metrics import accuracy_score
#from sklearn import ensemble

#df_4 = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["nr","year"])

path_CC_D_Y = path + 'cell_image/train/DIDA_1/'
CC_train = np.arange(0)
D_train = np.arange(0)
Y_train = np.arange(0)
all_files_train = []
loaded_images_train = []


for filename_train in listdir(path_CC_D_Y):
    img_data_train = cv2.imread(filename_train ,0)
    loaded_images_train.append(img_data_train) # append the images to a list
    all_files_train.append(filename_train)
    if len(all_files_train) == 8400: #we got 8400 training images
        sorted_alphanumeric(all_files_train) # sort the images as 1.jpg,2,3,5.. insted of 1,10,100.
        for i in range(len(df["nr"])): # Itereate through every label and add those who is pressent in training
            if df["nr"][i] in all_files_train: # if label is present in all_files it means the picture acures in the training data
                CC_train = np.append(CC_train, df["CC"][i]) # add the CC label to training labels
                D_train = np.append(D_train, df["D"][i]) # add the D label to training labels
                Y_train = np.append(Y_train, df["Y"][i]) # add the Y label to training labels
   
     

#%%

path_CC_D_Y = path + 'cell_image/val/DIDA_1/' # path is at validation folder
CC_val = np.arange(0)
D_val = np.arange(0)
Y_val = np.arange(0)
all_files_val = []
loaded_images_val = []


for filename_val in listdir(path_CC_D_Y):
    img_data_val = cv2.imread(filename_val ,0)
    loaded_images_val.append(img_data_val) # append the images to a list
    all_files_val.append(filename_val)
    if len(all_files_val) == 2400: #we got 8400 training images
        sorted_alphanumeric(all_files_val) # sort the images as 1.jpg,2,3,5.. insted of 1,10,100.
        for i in range(len(df["nr"])): # Itereate through every label and add those who is pressent in training
            if df["nr"][i] in all_files_val: # if label is present in all_files it means the picture acures in the training data
                CC_val = np.append(CC_val, df["CC"][i]) # add the CC label to training labels
                D_val = np.append(D_val, df["D"][i]) # add the D label to training labels
                Y_val = np.append(Y_val, df["Y"][i]) # add the Y label to training labels


#%%

path_CC_D_Y = path + 'cell_image/test/DIDA_1/' # path is at test folder
CC_test = np.arange(0)
D_test = np.arange(0)
Y_test = np.arange(0)
all_files_test = []
loaded_images_test = []


for filename_test in listdir(path_CC_D_Y):
    img_data_test = cv2.imread(filename_test ,0)
    loaded_images_test.append(img_data_test) # append the images to a list
    all_files_test.append(filename_test)
    if len(all_files_test) == 1200: #we got 8400 training images
        sorted_alphanumeric(all_files_test) # sort the images as 1.jpg,2,3,5.. insted of 1,10,100.
        for i in range(len(df["nr"])): # Itereate through every label and add those who is pressent in training
            if df["nr"][i] in all_files_test: # if label is present in all_files it means the picture acures in the training data
                CC_test = np.append(CC_test, df["CC"][i]) # add the CC label to training labels
                D_test = np.append(D_test, df["D"][i]) # add the D label to training labels
                Y_test = np.append(Y_test, df["Y"][i]) # add the Y label to training labels
                
#%%
print(len(loaded_images_test))


#%% Resize images TRAIN_DATA

resized_images_train = []

for images in loaded_images_train:
    resized_images_train.append(cv2.resize(images, (28,28)))
    
#print(resized_images[1100].shape)

#%% feature extraction TRAIN_DATA
features_train = []
for i in resized_images_train:
    features_train.append(np.reshape(i, 28*28))

print(len(features_train[0]))

#%% TRAIN_DATA
train_array = np.empty((0),int)
for i in range(len(features_train)):
    train_array = np.append(train_array, features_train[i])
    
        
#print(test_array)
train_array_2D = 0
train_array_2D = train_array.reshape(8400,784) 
       
print(train_array_2D.shape)


#%% Resize images VAL_DATA

resized_images_val = []

for images in loaded_images_val:
    resized_images_val.append(cv2.resize(images, (28,28)))
    
#print(resized_images[1100].shape)

#%% feature extraction VAL_DATA
features_val = []
for i in resized_images_val:
    features_val.append(np.reshape(i, 28*28))

print(len(features_val[0]))

#%% VAL_DATA
val_array = np.empty((0),int)
for i in range(len(features_val)):
    val_array = np.append(val_array, features_val[i])
    
        
#print(test_array)
val_array_2D = 0
val_array_2D = val_array.reshape(2400,784) 
       
print(val_array_2D.shape)


#%% Resize images #TEST_DATA

resized_images_test = []

for images in loaded_images_test:
    resized_images_test.append(cv2.resize(images, (28,28)))
    
#print(resized_images[1100].shape)

#%% feature extraction #TEST_DATA
features_test = []
for i in resized_images_test:
    features_test.append(np.reshape(i, 28*28))

print(len(features_test[0]))

#%% #TEST_DATA
test_array = np.empty((0),int)
for i in range(len(features_test)):
    test_array = np.append(test_array, features_test[i])
    
        
#print(test_array)
test_array_2D = 0
test_array_2D = test_array.reshape(1200,784) 
       
print(test_array_2D.shape)





#%%

print(CC_train.shape)


#%% CC_Model

from sklearn.metrics import accuracy_score
from sklearn import ensemble # ensemble instead of tree

X_train = train_array_2D
X_test = test_array_2D

y_train = CC_train
y_test = CC_test
# Initialize
gbt = ensemble.GradientBoostingClassifier()

# Fit
gbt.fit(X_train, y_train)

# Predict
y_test_hat = gbt.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat)
print(f'''Gradient boosted DTs with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')


#%%

X_train = train_array_2D
X_test = test_array_2D

y_train = Y_train
y_test = Y_test
# Initialize
gbt = ensemble.GradientBoostingClassifier()

# Fit
gbt.fit(X_train, y_train)

# Predict
y_test_hat = gbt.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat)
print(f'''Gradient boosted DTs with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')














