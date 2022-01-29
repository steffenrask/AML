#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from utils import load_data_flow

def aggregated_feature_maps(model, image, same_category_as_pred):
    # find output layer
    output = model.get_layer('output').output 
    # find last convolutional layer
    last_conv_layer = None
    for layer in model.layers:
        if "conv" in layer.name:
            last_conv_layer = layer.output
    # model w. 2 outputs; original ouput and last conv layer hidden state
    submodel = tf.keras.models.Model([model.inputs], [output, last_conv_layer])
    
    # make an input variable to differentiate wrt.
    input_img = image.copy()
    input_img = tf.Variable(tf.cast(input_img, tf.float32))

    # compute gradient wrt. image input
    with tf.GradientTape() as tape:
        preds, outputs_conv = submodel(input_img)        
        category = tf.argmax(preds[0])
        # category = np.argmax(preds.numpy)
        if not same_category_as_pred:
            category = tf.argmin(preds[0])
        
        class_activation = tf.reduce_mean(preds[:, category])

    # gradients wrt. feature map activations
    grads = tape.gradient(class_activation, outputs_conv)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    
    cam = tf.squeeze(outputs_conv @ pooled_grads[...,tf.newaxis])
    cam = tf.maximum(cam,0)/tf.math.reduce_max(cam)
    return cam

def guided_aggregated_feature_maps(model, image, same_category_as_pred):
    # find output layer
    output = model.get_layer('output').output 
    # find last convolutional layer
    last_conv_layer = None
    for layer in model.layers:
        if "conv" in layer.name:
            last_conv_layer = layer.output
    
    # model w. 2 outputs; original ouput and last conv layer hidden state
    submodel = tf.keras.models.Model([model.inputs], [output, last_conv_layer])
    
    # make an input variable to differentiate wrt.
    input_img = image.copy()
    input_img = tf.Variable(tf.cast(input_img, tf.float32))

    # compute gradient wrt. image input
    with tf.GradientTape() as tape:
        preds, outputs_conv = submodel(input_img)        
        category = tf.argmax(preds[0])
        if not same_category_as_pred:
            category = tf.argmin(preds[0])
        
        class_activation = tf.reduce_mean(preds[:, category])

    # gradients wrt. feature map activations
    grads = tape.gradient(class_activation, outputs_conv)
    # cast outputs as float datatype
    outputs_conv_float = tf.cast(outputs_conv > 0, "float32")
    grads_float = tf.cast(grads > 0, "float32")
    guided_grads = outputs_conv_float * grads_float * grads
    
    outputs_conv = outputs_conv[0]
    guided_grads = guided_grads[0]
    
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, outputs_conv), axis=-1)
    
    #normalize
    cam = cam/tf.math.reduce_max(cam)
    return cam


cmap = plt.get_cmap('turbo')

def create_heatmap(model, image, pred_category):
    #category, image = train_labels[idx], train_images[idx:(idx + 1)]

    heatmap = aggregated_feature_maps(model, image, pred_category)
    heatmap = heatmap.numpy()
    heatmap = Image.fromarray(heatmap)
    
    last_conv_index = 0
    for i, layer in enumerate(model.layers):
        if "conv" in layer.name:
            last_conv_index = i

    for layer in model.layers[0:last_conv_index+1]:
        size = heatmap.size[0]
        if "pool" in layer.name:
            heatmap = heatmap.resize((size*2, size*2), Image.BILINEAR)
        elif "conv" in layer.name:
            heatmap = heatmap.resize((size + 2, size + 2), Image.BILINEAR)
    
    
    heatmap = heatmap.resize((256, 256), Image.CUBIC) # upscale
    # heatmap = heatmap.resize((256, 256), Image.CUBIC) # upscale
    heatmap = np.array(heatmap) # back to numpy array
    heatmap = (heatmap / heatmap.max()) # to [0, 1]    
    #heatmap = np.where(heatmap > 0.4, heatmap, 0)
    heatmap = cmap(heatmap)
    heatmap = np.delete(heatmap, 3, 2)

    overlayed_heatmap = 0.6 * image[0] + 0.4* heatmap
    
    return image[0], heatmap, overlayed_heatmap

def plot_heatmap(model, image, pred_category=True):
    plt.figure(figsize=(10, 10))
    images = create_heatmap(model, image, pred_category)
    orig = Image.fromarray(np.uint8(255*images[0]))
    draw = ImageDraw.Draw(orig)
    label = "cat" if model.predict(image)[0][0] > 0.5 else "dog" 
    draw.text((10,256-30), "Predicted: " + label,  fill=(255,0,0),\
              font = ImageFont.truetype("Ubuntu-M.ttf",20) )
    images = (np.array(orig), images[1], images[2])
    for j in range(3):
        
        plt.subplot(3, 3, 1 + j); plt.axis('off'); plt.imshow(images[j])
    plt.show()

if __name__ == "__main__":
    test_flow =  load_data_flow("data/test", augmentation={"rescale":1./255})
    model = load_model("val88.h5")
    batch = next(test_flow)
    image= batch[0][0].reshape((1,256,256,3))
    plot_heatmap(model, image,  pred_category=True)
    plot_heatmap(model, image,  pred_category=False)
    
