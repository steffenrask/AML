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
from sklearn.metrics import accuracy_score
from sklearn import ensemble # ensemble instead of tree

#set path
path = 'C:/Users/Ribert/OneDrive/Kandidat/3 semester/DS807 Anvendt Maskinlæring/Exam/'

# read in the labels as as datafram and set columns as nr and year
df = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["nr","year"])

# Set the directory for the images
input_folder = path + 'DIDA_12000_String_Digit_Images/'

# Split the data into three seperate folders whit a ratio of 70/20/10: Train, val, test
splitfolders.ratio(input_folder, output= path + 'cell_image', 
                   seed=42, ratio=(.7, .2, .1), 
                   group_prefix=None) # default values

# Add ".jpg" to the every number in the nr column
def append_ext(fn):
    return str(fn) + ".jpg"

df["nr"]=df["nr"].apply(append_ext)


# Make a new column in the df datafram called CC and give each picture a class label of 0 or 1, for 18, or not 18.  
def change_to_cat_CC(ye):
    x = str(ye)   
    if len(x) == 4:
        if x[0] == "1" and x[1] == "8":
            return str(0) # Zero is for 18
        else:
            return str(1)
    return str(1) # One is for not 18

df["CC"]=df["year"].apply(change_to_cat_CC)


# Make a new column in the df datafram called D and give each picture a class label from 0 to 10
df_2 = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["nr","year"])

def change_to_cat_D(ye): # maybe change name
    x = str(ye)
    
    if len(x) != 4:
        return str(10)
    else:
        return str(x[2])       
            
df["D"]=df_2["year"].apply(change_to_cat_D)


# Make a new column in the df datafram called Y and give each picture a class label from 0 to 10
df_3 = pd.read_csv(path + 'DIDA_12000_String_Digit_Labels.csv', names=["nr","year"])

def change_to_cat_Y(ye): # maybe change name
    x = str(ye)
    
    if len(x) != 4:
        return str(10)
    else:
        return str(x[3])       
            
df["Y"]=df_3["year"].apply(change_to_cat_Y)


#%% Function to sort the data in order like 1,2,3... etc.

def sorted_alphanumeric(data): # https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


#%% Imports the training pictures and stores them in a list.

path_CC_D_Y = path + 'cell_image/train/DIDA_1/' #path to folder containing training data. Should be adjustet to individual user user!
os.chdir(path_CC_D_Y)
y_train_CC = np.arange(0) # array to store the CC labels for the training data
y_train_D = np.arange(0) # array to store the D labels for the training data
y_train_Y = np.arange(0) # array to store the Y labels for the training data
all_files_train = [] # List to use only in the loop
loaded_images_train = [] #Liste to store the training images

for filename_train in listdir(path_CC_D_Y):
    img_data_train = cv2.imread(filename_train ,0) # the ",0" in the end is the factor that turns the pictures into gray scale 
    loaded_images_train.append(img_data_train) # append the images to the list
    all_files_train.append(filename_train)
    if len(all_files_train) == 8400: #we got 8400 training images
        sorted_alphanumeric(all_files_train) # sort the images as 1.jpg,2.jpg,3,5.. insted of 1,10,100.
        for i in range(len(df["nr"])): # Itereate through every label and add those who is pressent in training data
            if df["nr"][i] in all_files_train: # if label is present in all_files it means the picture acures in the training data
                y_train_CC = np.append(y_train_CC, df["CC"][i]) # add the CC label to the CC training labels
                y_train_D = np.append(y_train_D, df["D"][i]) # add the D label to the D training labels
                y_train_Y = np.append(y_train_Y, df["Y"][i]) # add the Y label to the Y training labels
   
     
#%% Same principle as the code above, just for validation insted

path_CC_D_Y = path + 'cell_image/val/DIDA_1/' #path to folder containing validation data. Should be adjustet to individual user user!
os.chdir(path_CC_D_Y)
y_val_CC = np.arange(0)
y_val_D = np.arange(0)
y_val_Y = np.arange(0)
all_files_val = []
loaded_images_val = []

for filename_val in listdir(path_CC_D_Y):
    img_data_val = cv2.imread(filename_val ,0)
    loaded_images_val.append(img_data_val) 
    all_files_val.append(filename_val)
    if len(all_files_val) == 2400: #we got 2400 validation images
        sorted_alphanumeric(all_files_val) 
        for i in range(len(df["nr"])): 
            if df["nr"][i] in all_files_val: 
                y_val_CC = np.append(y_val_CC, df["CC"][i]) 
                y_val_D = np.append(y_val_D, df["D"][i]) 
                y_val_Y = np.append(y_val_Y, df["Y"][i]) 


#%% Same principle again, now for the test data

path_CC_D_Y = path + 'cell_image/test/DIDA_1/' #path to folder containing test data. Should be adjustet to individual user user!
os.chdir(path_CC_D_Y)
y_test_CC = np.arange(0)
y_test_D = np.arange(0)
y_test_Y = np.arange(0)
all_files_test = []
loaded_images_test = []

for filename_test in listdir(path_CC_D_Y):
    img_data_test = cv2.imread(filename_test ,0)
    loaded_images_test.append(img_data_test) 
    all_files_test.append(filename_test)
    if len(all_files_test) == 1200: #we got 1200 test images
        sorted_alphanumeric(all_files_test) 
        for i in range(len(df["nr"])): 
            if df["nr"][i] in all_files_test: 
                y_test_CC = np.append(y_test_CC, df["CC"][i]) 
                y_test_D = np.append(y_test_D, df["D"][i]) 
                y_test_Y = np.append(y_test_Y, df["Y"][i]) 
                
#%% Reshape images, feature extraction, standardization, and insert to 2D array for the training data

resized_images_train = []
features_train = []
train_array = np.empty((0),int)

for images in loaded_images_train:
    resized_images_train.append(cv2.resize(images, (32,64))) # Reshapes every image to 32*64 pixel

for i in resized_images_train:
    features_train.append(np.reshape(i, 32*64)/255) # feature extraction and standardize each pixel-value to be between 0 and 1.

for i in range(len(features_train)):
    train_array = np.append(train_array, features_train[i]) # Make one long feature vector containing all the features
        
X_train = train_array.reshape(8400,2048) # split the feature vector into a 2D array, where each row contains one image (2048 features/pixels) 
       

#%% Reshape images, feature extraction, standardization, and insert to 2D array for the validation data (same priciples as above)

resized_images_val = []
features_val = []
val_array = np.empty((0),int)

for images in loaded_images_val:
    resized_images_val.append(cv2.resize(images, (32,64)))

for i in resized_images_val:
    features_val.append(np.reshape(i, 32*64)/255)

for i in range(len(features_val)):
    val_array = np.append(val_array, features_val[i])
    
X_val = val_array.reshape(2400,2048) 
       

#%% Reshape images, feature extraction, standardization, and insert to 2D array for the test data (same priciples as above)

resized_images_test = []
features_test = []
test_array = np.empty((0),int)

for images in loaded_images_test:
    resized_images_test.append(cv2.resize(images, (32,64)))

for i in resized_images_test:
    features_test.append(np.reshape(i, 32*64)/255)

for i in range(len(features_test)):
    test_array = np.append(test_array, features_test[i])
    
X_test = test_array.reshape(1200,2048)


#%% Useing Cross-validation-scearch to find the optimal values for each parameter for ModelCC

# Different values to combine for each parameter
n_estimators_list = [5, 10, 20] 
learning_rate_list = [0.01, 0.1, 0.3]
min_samples_split_list = [2, 5, 10]
min_samples_leaf_list = [5, 10, 20]

results_CC = []

# Loop through each value in each list
for n_estimators in n_estimators_list:
    for min_samples_split in min_samples_split_list:
        for min_samples_leaf in min_samples_leaf_list:
            for learning_rate in learning_rate_list:
                gbt_current_CC = ensemble.GradientBoostingClassifier( # Classification method and different values for the parameters
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    )
                gbt_current_CC.fit(X_train, y_train_CC) # Fit the model
                y_val_hat_CC = gbt_current_CC.predict(X_val) #Predict the model using validation data
                acc = accuracy_score(y_val_CC, y_val_hat_CC) # Evaluate the model using accuracy_score.

                results_CC.append([acc, n_estimators, min_samples_split, min_samples_leaf, learning_rate]) # Append each models result to a list

results_CC = pd.DataFrame(results_CC)
results_CC.columns = ['Accuracy', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'learning_rate']
print(results_CC)

# Find and print the optimal values for each parameter which toghter made the model whit the bedst prediction (highest accuracy score)
n_estimators_optimal_CC = results_CC.loc[results_CC['Accuracy'].idxmax()]['n_estimators'].astype(int)
min_samples_split_optimal_CC = results_CC.loc[results_CC['Accuracy'].idxmax()]['min_samples_split'].astype(int)
min_samples_leaf_optimal_CC = results_CC.loc[results_CC['Accuracy'].idxmax()]['min_samples_leaf'].astype(int)
learning_rate_opt_CC = results_CC.loc[results_CC['Accuracy'].idxmax()]['learning_rate'].astype(float)
acc_opt_CC = results_CC.loc[results_CC['Accuracy'].idxmax()]['Accuracy'].astype(float)

print(acc_opt_CC, n_estimators_optimal_CC, min_samples_split_optimal_CC, min_samples_leaf_optimal_CC, learning_rate_opt_CC)

#%% Useing Cross-validation-scearch to find the optimal values for each parameter for ModelD (WARNING! can take a long time to run)
# Same principle as above, just for the ModelD

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

#%% Useing Cross-validation-scearch to find the optimal values for each parameter for ModelY (WARNING! can take a long time to run)
# Same principle as above, just for the ModelY

n_estimators_list = [5, 10, 20]
learning_rate_list = [0.01, 0.1, 0.3]
min_samples_split_list = [2, 5, 10]
min_samples_leaf_list = [5, 10, 20]

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



#%% Initializing the final ModelCC

# Insert the optimal values for the parameters and test the ModelCC on the test data
gbt_CC = ensemble.GradientBoostingClassifier(n_estimators=n_estimators_optimal_CC,
                                             min_samples_split=min_samples_split_optimal_CC,
                                             min_samples_leaf=min_samples_leaf_optimal_CC,
                                             learning_rate=learning_rate_opt_CC)

# Fit the model using the training data
gbt_CC.fit(X_train, y_train_CC)

# Predict the model based on the test data
y_test_hat_CC = gbt_CC.predict(X_test)

# Calculate the accuracy_score
accuracy_CC = accuracy_score(y_test_CC, y_test_hat_CC)

# Prints the accuracy score for the final model
print(f'''Gradient boosted DTs with optimal settings for CC achieved {round(accuracy_CC * 100, 1)}% accuracy.''')


#%% Initializing the final ModelCC (same principles as above just for ModelD)

gbt_D = ensemble.GradientBoostingClassifier(n_estimators=n_estimators_optimal_D,
                                             min_samples_split=min_samples_split_optimal_D,
                                             min_samples_leaf=min_samples_leaf_optimal_D,
                                             learning_rate=learning_rate_opt_D)

gbt_D.fit(X_train, y_train_D)

y_test_hat_D = gbt_D.predict(X_test)

accuracy_D = accuracy_score(y_test_D, y_test_hat_D)

print(f'''Gradient boosted DTs with optimal settings for D achieved {round(accuracy_D * 100, 1)}% accuracy.''')


#%% Initializing the final ModelCC (same principles as above just for ModelD)

gbt_Y = ensemble.GradientBoostingClassifier(n_estimators=n_estimators_optimal_Y,
                                             min_samples_split=min_samples_split_optimal_Y,
                                             min_samples_leaf=min_samples_leaf_optimal_Y,
                                             learning_rate=learning_rate_opt_Y)

gbt_Y.fit(X_train, y_train_Y)

y_test_hat_Y = gbt_Y.predict(X_test)

accuracy_Y = accuracy_score(y_test_Y, y_test_hat_Y)

print(f'''Gradient boosted DTs with optimal settings for Y achieved {round(accuracy_Y * 100, 1)}% accuracy.''')


#%%

p1=(y_test_hat_CC==y_test_CC)
p2=(y_test_hat_D==y_test_D)
p3=(y_test_hat_Y==y_test_Y)

p_CC = 0
for i in range(len(y_test_hat_CC)):
    if p1[i] == True:
        p_CC = p_CC + 1
p_D = 0
for i in range(len(y_test_hat_D)):
    if p2[i] == True:
        p_D = p_D + 1
p_Y = 0
for i in range(len(y_test_hat_Y)):
    if p3[i] == True:
        p_Y = p_Y + 1

p_sequence = 0
for i in range(len(p1)):
    if p1[i] == p2[i] == p3[i] == True:
       p_sequence = p_sequence + 1
    
print('point for CC-model:', round(p_CC))
print('point for D-model:', round(p_D))
print('point for Y-model:', round(p_Y))
print('point for character acc:', round(p_CC*0.33 + p_D*0.33 + p_Y*0.33), 'out of', len(y_test_CC))
print('point for correct sequence:', p_sequence, 'out of', len(y_test_CC))


