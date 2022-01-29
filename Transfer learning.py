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
batch_size=32,
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
batch_size=32,
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
    tf.keras.layers.Dense(1, activation='sigmoid'),
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
                    epochs=5
)
#%% D
train_generator_D=datagen.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/train/DIDA_1/",
x_col="Filename",
y_col="D",
batch_size=32,
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
batch_size=32,
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
                    epochs=5
)

#%% Y
train_generator_Y=datagen.flow_from_dataframe(
dataframe=df,
directory= path + "cell_image/train/DIDA_1/",
x_col="Filename",
y_col="Y",
batch_size=32,
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
batch_size=32,
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
                    epochs=5
)
