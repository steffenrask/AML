# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:58:44 2022

@author: Ribert
"""
# Imports libraries 
import pandas as pd
import cv2
import splitfolders
import numpy as np
import os
from os import listdir
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.preprocessing import image
from sklearn import ensemble # ensemble instead of tree
from tqdm import tqdm
import tensorflow as tf
#%%
#set path
path = 'C:/Users/45414/Desktop/Anvendt maskinelæring/DIDA/'

# read in the labels as as datafram and set columns as nr and year
df = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["nr","year"])

# Set the directory for the images
input_folder = path + 'DIDA_12000_String_Digit_Images/'

# Split the data into three seperate folders whit a ratio of 70/20/10: Train, val, test
splitfolders.ratio(input_folder, output= path + 'cell_image', 
                   seed=42, ratio=(.7, .2, .1), 
                   group_prefix=None) # default values
#%%
# Add ".jpg" to the every number in the nr column
def append_ext(fn):
    return str(fn) + ".jpg"

df["nr"]=df["nr"].apply(append_ext)


# Make a new column and  
def change_to_cat_CC(ye):
    x = str(ye)   
    if len(x) == 4:
        if x[0] == "1" and x[1] == "8":
            return str(0) # Zero is for 18
        else:
            return str(1)
    return str(1) # One is for not 18

df["CC"]=df["year"].apply(change_to_cat_CC)


#
df_2 = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["nr","year"])

def change_to_cat_D(ye): # maybe change name

    x = str(ye)
    
    if len(x) != 4:
        return str(10)
    else:
        return str(x[2])       
            
df["D"]=df_2["year"].apply(change_to_cat_D)


df_3 = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["nr","year"])


def change_to_cat_Y(ye): # maybe change name

    x = str(ye)
    
    if len(x) != 4:
        return str(10)
    else:
        return str(x[3])       
            
df["Y"]=df_3["year"].apply(change_to_cat_Y)


#%%

def sorted_alphanumeric(data): # https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


#%%

path_CC_D_Y = path + 'cell_image/train/DIDA_1/'
os.chdir(path_CC_D_Y)
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
os.chdir(path_CC_D_Y)
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
os.chdir(path_CC_D_Y)
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
                
#%% Resize images TRAIN_DATA

resized_images_train = []

for images in loaded_images_train:
    resized_images_train.append(cv2.resize(images, (32,64)))
    
    
#%% feature extraction and standardization #TRAIN_DATA

features_train = []

for i in resized_images_train:
    features_train.append(np.reshape(i, 32*64)/255)


#%% TRAIN_DATA

train_array = np.empty((0),int)

for i in range(len(features_train)):
    train_array = np.append(train_array, features_train[i])
        
train_array_2D = train_array.reshape(8400,2048) 
       

#%% Resize images VAL_DATA

resized_images_val = []

for images in loaded_images_val:
    resized_images_val.append(cv2.resize(images, (32,64)))
    

#%% feature extraction and standardization #VAL_DATA

features_val = []

for i in resized_images_val:
    features_val.append(np.reshape(i, 32*64)/255)


#%% VAL_DATA

val_array = np.empty((0),int)

for i in range(len(features_val)):
    val_array = np.append(val_array, features_val[i])
    
val_array_2D = val_array.reshape(2400,2048) 
       

#%% Resize images #TEST_DATA

resized_images_test = []

for images in loaded_images_test:
    resized_images_test.append(cv2.resize(images, (32,64)))
    

#%% feature extraction and standardization #TEST_DATA

features_test = []

for i in resized_images_test:
    features_test.append(np.reshape(i, 32*64)/255)


#%% #TEST_DATA

test_array = np.empty((0),int)

for i in range(len(features_test)):
    test_array = np.append(test_array, features_test[i])
    
test_array_2D = test_array.reshape(1200,2048)


#%% CC_Model

X_train = train_array_2D
X_val = val_array_2D
X_test = test_array_2D

y_train_CC = CC_train
y_val_CC = CC_val
y_test_CC = CC_test

y_train_D = D_train
y_val_D = D_val
y_test_D = D_test

y_train_Y = Y_train
y_val_Y = Y_val
y_test_Y = Y_test


#%% !WARNING takes long time to run this block: (5+ hours)

n_estimators_list = [5, 10, 20]
learning_rate_list = [0.01, 0.1, 0.3]
min_samples_split_list = [2, 5, 10]
min_samples_leaf_list = [5, 10, 20]

results_CC = []

for n_estimators in n_estimators_list:
    for min_samples_split in min_samples_split_list:
        for min_samples_leaf in min_samples_leaf_list:
            for learning_rate in learning_rate_list:
                gbt_current_CC = ensemble.GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    )
                gbt_current_CC.fit(X_train, y_train_CC)
                y_val_hat_CC = gbt_current_CC.predict(X_val)
                acc = accuracy_score(y_val_CC, y_val_hat_CC)

                results_CC.append([acc, n_estimators, min_samples_split, min_samples_leaf, learning_rate])

results_CC = pd.DataFrame(results_CC)
results_CC.columns = ['Accuracy', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'learning_rate']
print(results_CC)

n_estimators_optimal_CC = results_CC.loc[results_CC['Accuracy'].idxmax()]['n_estimators'].astype(int)
min_samples_split_optimal_CC = results_CC.loc[results_CC['Accuracy'].idxmax()]['min_samples_split'].astype(int)
min_samples_leaf_optimal_CC = results_CC.loc[results_CC['Accuracy'].idxmax()]['min_samples_leaf'].astype(int)
learning_rate_opt_CC = results_CC.loc[results_CC['Accuracy'].idxmax()]['learning_rate'].astype(float)
acc_opt_CC = results_CC.loc[results_CC['Accuracy'].idxmax()]['Accuracy'].astype(float)

print(acc_opt_CC, n_estimators_optimal_CC, min_samples_split_optimal_CC, min_samples_leaf_optimal_CC, learning_rate_opt_CC)

#%% !WARNING takes long time to run this block: (5+ hours)

n_estimators_list = [5, 10, 20]
learning_rate_list = [0.01, 0.1, 0.3]
min_samples_split_list = [2, 5, 10]
min_samples_leaf_list = [5, 10, 20]

results_D = []

for n_estimators in n_estimators_list:
    for min_samples_split in min_samples_split_list:
        for min_samples_leaf in min_samples_leaf_list:
            for learning_rate in learning_rate_list:
                gbt_current_D = ensemble.GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    )
                gbt_current_D.fit(X_train, y_train_D)
                y_val_hat_D = gbt_current_D.predict(X_val)
                acc = accuracy_score(y_val_D, y_val_hat_D)

                results_D.append([acc, n_estimators, min_samples_split, min_samples_leaf, learning_rate])

results_D = pd.DataFrame(results_D)
results_D.columns = ['Accuracy', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'learning_rate']
print(results_D)

n_estimators_optimal_D = results_D.loc[results_D['Accuracy'].idxmax()]['n_estimators'].astype(int)
min_samples_split_optimal_D = results_D.loc[results_D['Accuracy'].idxmax()]['min_samples_split'].astype(int)
min_samples_leaf_optimal_D = results_D.loc[results_D['Accuracy'].idxmax()]['min_samples_leaf'].astype(int)
learning_rate_opt_D = results_D.loc[results_D['Accuracy'].idxmax()]['learning_rate'].astype(float)
acc_opt_D = results_D.loc[results_D['Accuracy'].idxmax()]['Accuracy'].astype(float)

print(acc_opt_D, n_estimators_optimal_D, min_samples_split_optimal_D, min_samples_leaf_optimal_D, learning_rate_opt_D)

#%% !WARNING takes long time to run this block: (5+ hours)

n_estimators_list = [20]
learning_rate_list = [0.1]
min_samples_split_list = [5]
min_samples_leaf_list = [10]

results_Y = []

for n_estimators in n_estimators_list:
    for min_samples_split in min_samples_split_list:
        for min_samples_leaf in min_samples_leaf_list:
            for learning_rate in learning_rate_list:
                gbt_current_Y = ensemble.GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    )
                gbt_current_Y.fit(X_train, y_train_Y)
                y_val_hat_Y = gbt_current_Y.predict(X_val)
                acc = accuracy_score(y_val_Y, y_val_hat_Y)

                results_Y.append([acc, n_estimators, min_samples_split, min_samples_leaf, learning_rate])

results_Y = pd.DataFrame(results_Y)
results_Y.columns = ['Accuracy', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'learning_rate']
print(results_Y)

n_estimators_optimal_Y = results_Y.loc[results_Y['Accuracy'].idxmax()]['n_estimators'].astype(int)
min_samples_split_optimal_Y = results_Y.loc[results_Y['Accuracy'].idxmax()]['min_samples_split'].astype(int)
min_samples_leaf_optimal_Y = results_Y.loc[results_Y['Accuracy'].idxmax()]['min_samples_leaf'].astype(int)
learning_rate_opt_Y = results_Y.loc[results_Y['Accuracy'].idxmax()]['learning_rate'].astype(float)
acc_opt_Y = results_Y.loc[results_Y['Accuracy'].idxmax()]['Accuracy'].astype(float)

print(acc_opt_Y, n_estimators_optimal_Y, min_samples_split_optimal_Y, min_samples_leaf_optimal_Y, learning_rate_opt_Y)

#%%

default = (96.4, 24.9, 11.0)

CC_opti = ["n_esti = 10", "min_samples_split = 2", "min_samples_leaf = 10", "learning_rate = 0.1 (default)"]
#Mange forskellige varriarbler som giver samme resultat: Acc: 0.97041666

D_opti = ["n_esti = 20", "min_samples_split = 5", "min_samples_leaf = 10", "learning_rate = 0.1 (default)"]
# Acc: 0.2691666

Y_opti = ["n_esti = 20, min_sample_split = 5, min_samples_leaf = 10, learning_rate = 0.1"]
#Acc: 0.11375

#%% FINAL MODELS
#%% ModelCC
# Initialize

gbt_CC = ensemble.GradientBoostingClassifier(n_estimators=n_estimators_optimal_CC,
                                             min_samples_split=min_samples_split_optimal_CC,
                                             min_samples_leaf=min_samples_leaf_optimal_CC,
                                             learning_rate=learning_rate_opt_CC)

# Fit
gbt_CC.fit(X_train, y_train_CC)

# Predict
y_test_hat_CC = gbt_CC.predict(X_test)
accuracy_CC = accuracy_score(y_test_CC, y_test_hat_CC)
print(f'''Gradient boosted DTs with optimal settings for CC achieved {round(accuracy_CC * 100, 1)}% accuracy.''')


#%% ModelD
# Initialize

gbt_D = ensemble.GradientBoostingClassifier(n_estimators=n_estimators_optimal_D,
                                             min_samples_split=min_samples_split_optimal_D,
                                             min_samples_leaf=min_samples_leaf_optimal_D,
                                             learning_rate=learning_rate_opt_D)

# Fit
gbt_CC.fit(X_train, y_train_D)

# Predict
y_test_hat_D = gbt_CC.predict(X_test)
accuracy_D = accuracy_score(y_test_D, y_test_hat_D)
print(f'''Gradient boosted DTs with optimal settings for D achieved {round(accuracy_CC * 100, 1)}% accuracy.''')


#%% ModelY
# Initialize

gbt_Y = ensemble.GradientBoostingClassifier(n_estimators=n_estimators_optimal_Y,
                                             min_samples_split=min_samples_split_optimal_Y,
                                             min_samples_leaf=min_samples_leaf_optimal_Y,
                                             learning_rate=learning_rate_opt_Y)

# Fit
gbt_Y.fit(X_train, y_train_Y)

# Predict
y_test_hat_Y = gbt_CC.predict(X_test)
accuracy_Y = accuracy_score(y_test_Y, y_test_hat_Y)
print(f'''Gradient boosted DTs with optimal settings for Y achieved {round(accuracy_CC * 100, 1)}% accuracy.''')


#%%

p1=(y_test_hat_CC==CC_test)
p2=(y_test_hat_D==D_test)
p3=(y_test_hat_Y==Y_test)

p_CC = 0
for i in range(len(y_test_hat_CC)):
    if p1[i] == True:
        p_CC = p_CC + 0.33
p_D = 0
for i in range(len(y_test_hat_D)):
    if p2[i] == True:
        p_D = p_D + 0.33
p_Y = 0
for i in range(len(y_test_hat_Y)):
    if p3[i] == True:
        p_Y = p_Y + 0.3

p_sequence = 0
for i in range(len(p1)):
    if p1[i] == p2[i] == p3[i] == True:
       p_sequence = p_sequence + 1
    
print('point for CC-model:', round(p_CC))
print('point for D-model:', round(p_D))
print('point for Y-model:', round(p_Y))
print('point for character acc:', round(p_CC + p_D + p_Y), 'out of', len(CC_test))
print('point for correct sequence:', p_sequence, 'out of', len(CC_test))











## VISO
#%%
k_folder = 'C:/Users/45414/Desktop/Anvendt maskinelæring/DIDA/cell_image/train'
def image_files(input_directory):
    filepaths=[]
    labels=[]
    digit_folders=os.listdir(input_directory)
    #print(digit_folders)
    
    for digit in digit_folders:
        path=os.path.join(input_directory, digit)
        flist=os.listdir(path)
        for f in flist:
            fpath=os.path.join(path,f)        
            filepaths.append(fpath)
            labels.append(digit) 
    return filepaths,labels
def load_images(filepaths):
    images = []
    for i in tqdm(range(len(filepaths))):
        img = image.load_img(filepaths[i], target_size=(32,64,3), grayscale=False)
        img = image.img_to_array(img)
        img.astype('float32')
        img = img/255
        images.append(img)

    images = np.array(images)
    return images
filepaths,labels = image_files(k_folder)
images = load_images(filepaths)
#%%
train_images = images
example_image = train_images[0:1].copy()
plt.imshow(example_image[0])
#%% EXAMPLE MODEL, SHOULD BE OURS 
modelY = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 64, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(), # flatten before fully connected part
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(11, activation='softmax'), # softmax for multiple classes
    ])
#%%
#%%
modelY.summary()
#%%
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1); 
    plt.axis('off'); 
    plt.imshow(modelY.get_layer('conv2d_6')(example_image)[0, :, :, i])
#%%
def tidy_image(image):
    image = image.numpy()[0]
    image -= image.mean()
    image /= (image.std() + 0.00001)
    image *= 0.1
    image += 0.5
    image = np.clip(image, 0, 1)
        
    return image
#%%
import numpy as np

def generate_pattern(layer_name, filter_index, im_size):
    submodel = tf.keras.models.Model([modelY.inputs], [modelY.get_layer(layer_name).output])

    input_img_data = np.random.random((1, *im_size))
    input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

    # Iterate gradient ascents
    for _ in range(100):
        with tf.GradientTape() as tape:
            outputs = submodel(input_img_data)
            loss_value = tf.reduce_mean(outputs[:, :, :, filter_index])
        grads = tape.gradient(loss_value, input_img_data)
        normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
        input_img_data.assign_add(normalized_grads * 1.0)
        
    return tidy_image(input_img_data)
#%%
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1); plt.axis('off'); plt.imshow(generate_pattern('conv2d_6', i, (32, 64, 3)))
#%%
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1); plt.axis('off'); plt.imshow(generate_pattern('conv2d_7', i, (32, 64, 3)))
#%%
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1); plt.axis('off'); plt.imshow(generate_pattern('conv2d_8', i, (32, 64, 3)))
#%%
class_names=['0','1','2','3','4','5','6','7','8','9']
def transform_image(category, steps, start_image):
    submodel = tf.keras.models.Model([modelY.inputs], [modelY.get_layer('dense_5').output])

    input_img_data = start_image.copy()
    input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

    # Iterate gradient ascents
    for _ in range(steps):
        with tf.GradientTape() as tape:
            outputs = submodel(input_img_data)
            loss_value = tf.reduce_mean(outputs[:, category])
        grads = tape.gradient(loss_value, input_img_data)
        normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
        input_img_data.assign_add(normalized_grads * 1.0)
    
    yhat = np.argmax(modelY.predict(input_img_data.numpy()))
    
    return tidy_image(input_img_data), yhat
#%%

def plot(category):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        out, yhat = generate_pattern(category, i * 3, example_image)
        out = out.reshape(1, *out.shape)
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(out[0])
        plt.xlabel(class_names[yhat])
    plt.show()
#%%
plot(1)
#%%
modelY.summary()
#%%
def get_heatmap(category, start_image):
    frog_output     = modelY.get_layer('dense_5').output # output layer
    last_conv_layer = modelY.get_layer('conv2d_8').output # deep convolution 
                                                         # we could use something else
    submodel = tf.keras.models.Model([modelY.inputs], [frog_output, last_conv_layer])

    input_img_data = start_image.copy()
    input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

    with tf.GradientTape() as tape:
        outputs_class, outputs_conv = submodel(input_img_data)
        loss_value                  = tf.reduce_mean(outputs_class[:, category])

    grads = tape.gradient(loss_value, outputs_conv)

    cast_outputs_conv = tf.cast(outputs_conv > 0, "float32")
    cast_grads        = tf.cast(grads > 0, "float32")
    guided_grads      = cast_outputs_conv * cast_grads * grads
    outputs_conv      = outputs_conv[0]
    guided_grads      = guided_grads[0]
    
    weights           = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam               = tf.reduce_sum(tf.multiply(weights, outputs_conv), axis=-1)
    
    return cam
#%%
from PIL import Image
cmap = plt.get_cmap('jet')
train_labels_1=CC_train
train_labels_1=[int(g) for g in train_labels_1]
train_labels_1=np.array(train_labels_1)
def create_heatmap(idx):
    category, image = train_labels_1[idx], train_images[idx:(idx + 1)]

    heatmap = get_heatmap(category, image)
    heatmap = heatmap.numpy()
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize((64, 32), Image.ANTIALIAS) # upscale
    heatmap = np.array(heatmap) # back to numpy array
    heatmap = (heatmap / heatmap.max()) # to [0, 1]    
    heatmap = cmap(heatmap)
    heatmap = np.delete(heatmap, 3, 2)

    overlayed_heatmap = 0.6 * image[0] + 0.4 * heatmap
    
    return image[0], heatmap, overlayed_heatmap

def plot_heatmap():
    plt.figure(figsize=(10, 10))
    for i in range(3):
        images = create_heatmap(i)
        for j in range(3):
            ax = plt.subplot(3, 3, i * 3 + 1 + j); plt.axis('off'); plt.imshow(images[j])
    plt.show()
plot_heatmap()








