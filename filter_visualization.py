#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras import backend as K
from tqdm import tqdm
tf.compat.v1.disable_eager_execution()

OUTPUT_DIM = 100
MARGIN = 5
    
    
def max_mean_activation_input(tensor,model):
    """
       Construct input image maximizing mean activation 
    """
    input_img = model.input
    output = tensor
    # loss = K.mean(output)
    loss = K.max(output)
    grads = K.gradients(loss, input_img)[0]
    
    img = np.random.random((1,OUTPUT_DIM, OUTPUT_DIM, 3))
    
    #normalization
    grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())
    
    gradients = K.function([input_img], [grads])
    
    # gradient ascent to maximize mean activation
    ITERATIONS = 150
    LR = 1
    for _ in range(ITERATIONS):
        img += LR * gradients([img])[0]
    
    # standardize and scale
    eps = 1e-8
    img -= np.mean(img)
    img /= (np.std(img) + eps)
    img *= 0.25
    # clip to [0,1] and scale
    img += 0.5
    img = np.clip(img, 0, 1)*255
    img = img.astype('uint8')
    return img

def _plot_channel(layer, images, channel, nx, ny ):
    """
        Plot all filters wrt. to channel (should be the same for all channels)
    """
    width = nx * OUTPUT_DIM + (nx - 1) * MARGIN
    height = ny * OUTPUT_DIM + (ny - 1) * MARGIN
    stitched_filters = np.zeros((width, height, 1), dtype='uint8')

    # fill the picture with our saved filters
    for i in range(nx):
        for j in range(ny):
            img = images[i * nx + j]
            img = img.reshape((100,100,3))
            width_margin = (OUTPUT_DIM + MARGIN) * i
            height_margin = (OUTPUT_DIM + MARGIN) * j
            stitched_filters[width_margin : width_margin + OUTPUT_DIM, height_margin : height_margin + OUTPUT_DIM,:] = img[:,:,channel].reshape((100,100,1))
            
    save_img('filters_{0:}_ch{1:}.png'.format(layer.name, channel), stitched_filters)

def plot_filters(layer, model, nx, ny):
    n = layer.output_shape[-1]
    
    filters = [layer.output[:,:,:,i] for i in range(n)]
    images = []
    for f in tqdm(filters):
        images.append(max_mean_activation_input(f,model))
    
    for channel in range(3):
        _plot_channel(layer, images, channel, nx, ny)


if __name__ == "__main__":
    model = tf.keras.models.load_model("val88.h5")
    model.trainable = False
    # plot_filters(model.get_layer("conv2d_1"), model, 4, 4)
    # plot_filters(model.get_layer("conv2d_2"), model, 4, 4)
    # plot_filters(model.get_layer("conv2d_3"), model, 4, 8)
    plot_filters(model.get_layer("conv2d_4"), model, 8, 8)