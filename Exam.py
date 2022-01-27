# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:58:44 2022

@author: Ribert
"""

from keras_preprocessing.image import ImageDataGenerator

import pandas as pd



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
from matplotlib import image


#from sklearn.metrics import accuracy_score
#from sklearn import ensemble

#df_4 = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["nr","year"])

path_CC_D_Y = path + 'cell_image/train/DIDA_1/'
CC_train = []
D_train = []
Y_train = []
all_files_train = []
loaded_images_train = []


for filename_train in listdir(path_CC_D_Y):
    img_data_train = image.imread(path_CC_D_Y + filename_train)
    loaded_images_train.append(img_data_train) # append the images to a list
    all_files_train.append(filename_train)
    if len(all_files_train) == 8400: #we got 8400 training images
        sorted_alphanumeric(all_files_train) # sort the images as 1.jpg,2,3,5.. insted of 1,10,100.
        for i in range(len(df["nr"])): # Itereate through every label and add those who is pressent in training
            if df["nr"][i] in all_files_train: # if label is present in all_files it means the picture acures in the training data
                CC_train.append(df["CC"][i]) # add the CC label to training labels
                D_train.append(df["D"][i]) # add the D label to training labels
                Y_train.append(df["Y"][i]) # add the Y label to training labels
   
    
        
    #print('> loaded %s %s' % (filename, img_data.shape))
#%%

path_CC_D_Y = path + 'cell_image/val/DIDA_1/' # path is at validation folder
CC_val = []
D_val = []
Y_val = []
all_files_val = []
loaded_images_val = []


for filename_val in listdir(path_CC_D_Y):
    img_data_val = image.imread(path_CC_D_Y + filename_val)
    loaded_images_val.append(img_data_val) # append the images to a list
    all_files_val.append(filename_val)
    if len(all_files_val) == 2400: #we got 8400 training images
        sorted_alphanumeric(all_files_val) # sort the images as 1.jpg,2,3,5.. insted of 1,10,100.
        for i in range(len(df["nr"])): # Itereate through every label and add those who is pressent in training
            if df["nr"][i] in all_files_val: # if label is present in all_files it means the picture acures in the training data
                CC_val.append(df["CC"][i]) # add the CC label to training labels
                D_val.append(df["D"][i]) # add the D label to training labels
                Y_val.append(df["Y"][i]) # add the Y label to training labels


#%%

path_CC_D_Y = path + 'cell_image/test/DIDA_1/' # path is at test folder
CC_test = []
D_test = []
Y_test = []
all_files_test = []
loaded_images_test = []


for filename_test in listdir(path_CC_D_Y):
    img_data_test = image.imread(path_CC_D_Y + filename_test)
    loaded_images_test.append(img_data_test) # append the images to a list
    all_files_test.append(filename_test)
    if len(all_files_test) == 1200: #we got 8400 training images
        sorted_alphanumeric(all_files_test) # sort the images as 1.jpg,2,3,5.. insted of 1,10,100.
        for i in range(len(df["nr"])): # Itereate through every label and add those who is pressent in training
            if df["nr"][i] in all_files_test: # if label is present in all_files it means the picture acures in the training data
                CC_test.append(df["CC"][i]) # add the CC label to training labels
                D_test.append(df["D"][i]) # add the D label to training labels
                Y_test.append(df["Y"][i]) # add the Y label to training labels



#%%
train_images_CC = np.array(loaded_images_train)
train_labels_CC = np.array(CC_train)

val_images_CC = np.array(loaded_images_val)
val_labels_CC = np.array(CC_val)

test_images_CC = np.array(loaded_images_test)
test_labels_CC = np.array(CC_test)


train_images_D = np.array(loaded_images_train)
train_labels_D = np.array(D_train)

val_images_D = np.array(loaded_images_val)
val_labels_D = np.array(D_val)

test_images_D = np.array(loaded_images_test)
test_labels_D = np.array(D_test)


train_images_Y = np.array(loaded_images_train)
train_labels_Y = np.array(Y_train)

val_images_Y = np.array(loaded_images_val)
val_labels_Y = np.array(Y_val)

test_images_Y = np.array(loaded_images_test)
test_labels_Y = np.array(Y_test)

#%% 

train_images_CC, val_images_CC = train_images_CC / 255.0, val_images_CC / 255.0

#%%
resized_images = []

for i in range(len(loaded_images_train)):
    image_resize = loaded_images_train[i].resize(150,150)
    resized_images.append(image_resize)

#%%

import cv2
from sklearn.metrics import accuracy_score
from sklearn import ensemble # ensemble instead of tree


imGray_1 = cv2.imread('1.jpg',0)
imGray_2 = cv2.imread('2.jpg',0)
imGray_3 = cv2.imread('3.jpg',0)
#print(loaded_images_train[0].shape)

x = []
ååå = np.reshape(imGray_1,(imGray_1.shape[1]*imGray_1.shape[0]))
åå = (np.reshape(imGray_3,(imGray_3.shape[1]*imGray_3.shape[0])))
x.append(ååå)
x.append(åå)
x_test = np.reshape(imGray_2,(imGray_2.shape[0]*imGray_2.shape[1]))

#print(len(x[0]))
my_array = np.array(x[0][0])


#my_array_2D = my_array.reshape(2)

for i in range(len(x)):
    for j in range(1,len(x[i])):
        my_array = np.append(my_array, x[i][j])
        #if j == len(x[i])-1 and i == 0:
         #   my_array_2D = my_array.reshape(2,)

my_array_2D = my_array.reshape(2,15552)          
print(my_array_2D)
    


#%%
#x_train = np.transpose(x_train)

y_train = [6]
y_test = [6]

gbt = ensemble.GradientBoostingClassifier()

# Fit
gbt.fit(x_train, y_train)

import matplotlib


#print(imGray.shape)

#%%
    
























#%%

from sklearn.metrics import accuracy_score
from sklearn import ensemble # ensemble instead of tree

X_train_CC, X_val_CC, y_train_CC, y_val_CC = train_images_CC, val_images_CC, train_labels_CC, val_labels_CC
print(X_train_CC.shape, X_val_CC.shape, y_train_CC.shape, y_val_CC.shape)

#%%

y_train_CC = y_train_CC.reshape(1,-1)

#%%
# Initialize
gbt = ensemble.GradientBoostingClassifier()

# Fit
gbt.fit(X_train_CC, y_train_CC)

# Predict
#y_test_hat = gbt.predict(X_val_CC)
#accuracy = accuracy_score(y_val_CC, y_test_hat)
#print(f'''Gradient boosted DTs with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')

#%%
print(train_images.shape, train_labels.shape)

"""
gbt = ensemble.GradientBoostingClassifier()

# Fit
gbt.fit(train_generator_eighteen, valid_generator_eighteen)

# Predict
y_test_hat = gbt.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat)
print(f'''Gradient boosted DTs with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')


"""



















