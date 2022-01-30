# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 10:09:16 2022

@author: 45414
"""

# Imports
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization
import matplotlib.pyplot as plt

path = 'C:/Users/45414/Desktop/Anvendt maskinelæring/DIDA/'

df = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["Index","Label"])

#%%
def append_ext(fn):
    return str(fn) + ".jpg"

df["nr"]=df["Label"].apply(append_ext)

# Create new labels for CC-D-Y models
# 18 century (0) or not (1)
def label_CC(label):

    x = str(label)   
    if len(x) == 4:
        if x[0] == "1" and x[1] == "8":
            return str(0)
        else:
            return str(1)
    return str(1)

# Decade or residual (10)
def label_D(label):

    x = str(label)
    if len(x) != 4:
        return str(10)
    else:
        return str(x[2])  

# Year or residual (10)
def label_Y(label):

    x = str(label)
    if len(x) != 4:
        return str(10)
    else:
        return str(x[3]) 
    
def filename(index):
   return str(index) + '.jpg'
    

df["CC"]=df["Label"].apply(label_CC)
df["D"]=df["Label"].apply(label_D)
df["Y"]=df["Label"].apply(label_Y)
df["Filename"]=df["Index"].apply(filename)

# Check the ratio of samples in each class:
print(df["CC"].value_counts()) # how many are there in each category
print(df["D"].value_counts()) # how many are there in each category
print(df["Y"].value_counts()) # how many are there in each category


input_folder = path + 'DIDA_12000_String_Digit_Images/'
splitfolders.ratio(input_folder, output= path + 'cell_image', 
                   seed=42, ratio=(.7, .2, .1), 
                   group_prefix=None) # default values


#%% CC
datagen=ImageDataGenerator(
    rescale=1./255.,
    rotation_range=5,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
    )

datagen_test=ImageDataGenerator(rescale=1./255.)

train_generator_CC=datagen.flow_from_dataframe(
dataframe=df,
directory=path + "cell_image/train/DIDA_1/",
x_col="Filename",
y_col="CC",
batch_size=200,
seed=42,
shuffle=True,
class_mode="binary",
target_size=(32,64))



valid_generator_CC=datagen_test.flow_from_dataframe( #maybe use augmentation on val also?
dataframe=df,
directory= path + "cell_image/val/DIDA_1/",
x_col="Filename",
y_col="CC",
#subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="binary",
target_size=(32,64))



test_generator_CC=datagen_test.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/test/DIDA_1/",
x_col="Filename",
y_col="CC",
batch_size=1200,
seed=42,
shuffle=False,
class_mode="binary",
target_size=(32,64))



#%% TRANSFER LEARNING MODEL CC

pre_trained_model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 64, 3), # refers to the shape we transfer from
    include_top=False, # cut off the head
    weights='imagenet', # pretrained on the ImageNet data
)
for layer in pre_trained_model.layers: #freeze the base model to not train it 
    layer.trainable=False
pre_trained_model.summary()
#%%



model_CC = tf.keras.models.Sequential([
    pre_trained_model, # the pre-trained part
    tf.keras.layers.Flatten(), # flatten before fully connected part
    tf.keras.layers.Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    tf.keras.layers.Dense(2, activation='sigmoid'),
    ])
model_CC.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    )
model_CC.summary()
#%%

from tensorflow.keras import layers
STEP_SIZE_TRAIN=train_generator_CC.n//train_generator_CC.batch_size
STEP_SIZE_VALID=valid_generator_CC.n//valid_generator_CC.batch_size
STEP_SIZE_TEST=test_generator_CC.n//test_generator_CC.batch_size

history = model_CC.fit_generator(generator=train_generator_CC,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator_CC,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1
)
#%% D
train_generator_D=datagen.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/train/DIDA_1/",
x_col="Filename",
y_col="D",
batch_size=200,
seed=42,
shuffle=True,
class_mode="sparse",
target_size=(32,64))

valid_generator_D=datagen_test.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/val/DIDA_1/",
x_col="Filename",
y_col="D",
batch_size=32,
seed=42,
shuffle=True,
class_mode="sparse",
target_size=(32,64))

test_generator_D=datagen_test.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/test/DIDA_1/",
x_col="Filename",
y_col="D",
batch_size=1200,
seed=42,
shuffle=False,
class_mode="sparse",
target_size=(32,64))

#%% TRANSFER LEARNING MODEL D

pre_trained_model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 64, 3), # refers to the shape we transfer from
    include_top=False, # cut off the head
    weights='imagenet', # pretrained on the ImageNet data
)
for layer in pre_trained_model.layers: #freeze the base model to not train it 
    layer.trainable=False
pre_trained_model.summary()
#%%
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization


model_D = tf.keras.models.Sequential([
    pre_trained_model, # the pre-trained part
    tf.keras.layers.Flatten(), # flatten before fully connected part
    tf.keras.layers.Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    tf.keras.layers.Dense(11, activation='softmax'),
    ])
model_D.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    )
model_D.summary()
#%%

STEP_SIZE_TRAIN=train_generator_D.n//train_generator_D.batch_size
STEP_SIZE_VALID=valid_generator_D.n//valid_generator_D.batch_size
STEP_SIZE_TEST=test_generator_D.n//test_generator_D.batch_size

historyD = model_D.fit_generator(generator=train_generator_D,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator_D,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1
)

#%% Y
train_generator_Y=datagen.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/train/DIDA_1/",
x_col="Filename",
y_col="Y",
batch_size=200,
seed=42,
shuffle=True,
class_mode="sparse",
target_size=(32,64)
)

valid_generator_Y=datagen_test.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/val/DIDA_1/",
x_col="Filename",
y_col="Y",
batch_size=32,
seed=42,
shuffle=True,
class_mode="sparse",
target_size=(32,64)
)

test_generator_Y=datagen_test.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/test/DIDA_1/",
x_col="Filename",
y_col="Y",
batch_size=1200,
seed=42,
shuffle=False,
class_mode="sparse",
target_size=(32,64)
)
#%% TRANSFER LEARNING MODEL Y

pre_trained_model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 64, 3), # refers to the shape we transfer from
    include_top=False, # cut off the head
    weights='imagenet', # pretrained on the ImageNet data
)
for layer in pre_trained_model.layers: #freeze the base model to not train it 
    layer.trainable=False
pre_trained_model.summary()
#%%
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization


model_Y = tf.keras.models.Sequential([
    pre_trained_model, # the pre-trained part
    tf.keras.layers.Flatten(), # flatten before fully connected part
    tf.keras.layers.Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    tf.keras.layers.Dense(11, activation='softmax'),
    ])
model_Y.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    )
model_Y.summary()
#%%

STEP_SIZE_TRAIN=train_generator_Y.n//train_generator_Y.batch_size
STEP_SIZE_VALID=valid_generator_Y.n//valid_generator_Y.batch_size
STEP_SIZE_TEST=test_generator_Y.n//test_generator_Y.batch_size

historyY = model_Y.fit(train_generator_Y,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator_Y,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1
)

#%%
import numpy as np
# FOR CC
X_test, y_test = next(test_generator_CC)
real_labels_CC = y_test
real_labels_CC=real_labels_CC.astype(int)
predictions = model_CC.predict(X_test)
print(predictions)
predict_labels_CC = np.argmax(predictions, axis=-1)
print('predict for CC',predict_labels_CC)

# FOR D
X_test, y_test = next(test_generator_D)
real_labels_D = y_test
real_labels_D=real_labels_D.astype(int)
predictions = model_D.predict(X_test)
print(predictions)
predict_labels_D = np.argmax(predictions, axis=-1)
print('predict for D',predict_labels_D)

# FOR Y
X_test, y_test = next(test_generator_Y)
real_labels_Y = y_test
real_labels_Y=real_labels_Y.astype(int)
predictions = model_Y.predict(X_test)
print(predictions)
predict_labels_Y = np.argmax(predictions, axis=-1)
print('predict for Y',predict_labels_Y)


#%%

p1=(predict_labels_CC==real_labels_CC)
p2=(predict_labels_D==real_labels_D)
p3=(predict_labels_Y==real_labels_Y)

CNN_p_CC = 0
for i in range(len(predict_labels_CC)):
    if p1[i] == True:
        CNN_p_CC = CNN_p_CC + 0.33
CNN_p_D = 0
for i in range(len(predict_labels_CC)):
    if p2[i] == True:
        CNN_p_D = CNN_p_D + 0.33
CNN_p_Y = 0
for i in range(len(predict_labels_CC)):
    if p3[i] == True:
        CNN_p_Y = CNN_p_Y + 0.3

CNN_p_sequence = 0
for i in range(len(p1)):
    if p1[i] == p2[i] == p3[i] == True:
       CNN_p_sequence = CNN_p_sequence + 1
#%%
print('point for CC-model:', round(CNN_p_CC))
print('point for D-model:', round(CNN_p_D))
print('point for Y-model:', round(CNN_p_Y))
print('point for character acc:', round(CNN_p_CC + CNN_p_D + CNN_p_Y), 'out of', len(real_labels_CC))
print('point for correct sequence:', CNN_p_sequence, 'out of', len(real_labels_CC))

#%% QUESTION 3, 2

'''
2. Investigate if your model’s performance is
particularly good or bad at correctly 
classifying certain classes (i.e., it might be very good 
at correctly classifying centuries 
but not years, or it might be good at correctly classifying 
some decades but not certain other decades). Does it mix up 
certain classes? If yes, does this surprise you (explain why or why not)?
'''

print(real_labels_CC)
print(real_labels_D)
print(real_labels_Y)
print('predict for CC',predict_labels_CC)
print('predict for Y',predict_labels_Y)
print('predict for D',predict_labels_D)

#%%
n_list = []
for i in range(len(real_labels_CC)):
    if predict_labels_CC[i] != real_labels_CC[i]:
        n_list.append(predict_labels_CC[i])
        
print(n_list)
#%%
nn_list = []
for i in range(len(real_labels_D)):
    if predict_labels_D[i] != real_labels_D[i]:
        nn_list.append(predict_labels_D[i])
        
print(nn_list)

#%%
nnn_list = []
for i in range(len(real_labels_Y)):
    if predict_labels_Y[i] != real_labels_Y[i]:
        nnn_list.append(predict_labels_Y[i])
        
print(nnn_list)

#%%
from collections import Counter
print(Counter(n_list))
print(Counter(nn_list))
print(Counter(nnn_list))