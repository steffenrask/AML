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
target_size=(32,32))



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
target_size=(32,32))



test_generator_CC=datagen_test.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/test/DIDA_1/",
x_col="nr",
y_col="CC",
batch_size=32,
seed=42,
shuffle=False,
class_mode="categorical",
target_size=(32,32))



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
target_size=(32,32))


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
target_size=(32,32))



test_generator_Y=datagen_test.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/test/DIDA_1/",
x_col="nr",
y_col="Y",
batch_size=32,
seed=42,
shuffle=False,
class_mode="categorical",
target_size=(32,32))


#%%


import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16


#from sklearn.metrics import accuracy_score
#from sklearn import ensemble

#df_4 = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["nr","year"])


SIZE = 256

train_images = []
train_labels = []

for directory_path in glob.glob("images/classification/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)


"""
gbt = ensemble.GradientBoostingClassifier()

# Fit
gbt.fit(train_generator_eighteen, valid_generator_eighteen)

# Predict
y_test_hat = gbt.predict(X_test)
accuracy = accuracy_score(y_test, y_test_hat)
print(f'''Gradient boosted DTs with default settings achieved {round(accuracy * 100, 1)}% accuracy.''')


"""



















