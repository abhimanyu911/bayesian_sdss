import astroNN
import os
import cv2
import h5py
from livelossplot import PlotLossesKeras
from astroNN.datasets import load_galaxy10sdss
from astroNN.datasets.galaxy10sdss import galaxy10cls_lookup
import tensorflow_probability as tfp 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics 
from math import ceil, floor
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import json

tfd = tfp.distributions
tfpl = tfp.layers

total_samples = 21768
train_samples = floor(total_samples*0.7)
val_samples = ceil(total_samples*0.15)
test_samples = total_samples - train_samples - val_samples
batch_size = 64 
epochs = 50 
weights_dir = {
    'frequentist' : './checkpoints/freq_logs/best' , 
    'bayesian' : './checkpoints/baye_logs/best'
}

def remove_specific_class(images = None, labels = None, class_label = None):
    """
    Removes specific class from the dataset and returns the removed samples as well.
    """
    if images is None or labels is None or class_label is None:
        raise ValueError("Images, labels and class label must be provided.")
    else:
        # Get the indices of the class
        indices = np.where(labels == class_label)
        # Return the removed samples as well
        images_removed = np.take(images, indices, axis = 0)
        labels_removed = np.take(labels, indices, axis = 0)
        # Remove the class from the dataset
        images = np.delete(images, indices, axis = 0)
        labels = np.delete(labels, indices, axis = 0)
        return images, labels, images_removed, labels_removed
    
def convert_to_categorical(labels = None):
    """
    Converts the labels to categorical.
    """
    if labels is None:
        raise ValueError("Labels and number of classes must be provided.")
    else:
        one_hot_labels = []
        for label in labels:
            one_hot_vector = [0,0,0,0,0,0,0,0,0]
            if label>5:
                label = label - 1
            one_hot_vector[int(label)] = 1
            one_hot_labels.append(one_hot_vector)
        return np.array(one_hot_labels).astype('float32')

def create_hdf5_file(images = None, labels = None, mode = None):
    """
    Creates HDF5 file from the given images and labels.
    """
    if images is None or labels is None or mode is None:
        raise ValueError("Images, labels and mode must be provided.")
    else:
        # Create the HDF5 file
        samples = images.shape[0]
        labels_dim = labels.shape[1]
        image_size = images.shape[1]
        channels = images.shape[3]
        file = h5py.File(f'{mode}.hdf5', 'w')
        images_set, labels_set =  file.create_dataset(
            'images',
            (samples, image_size, image_size, channels),
            dtype='f4'
        ), file.create_dataset(
            'labels',
            (samples, labels_dim),
            dtype='f4'
        )
        # Write the images and labels to the HDF5 file
        images_set[...] = images
        labels_set[...] = labels
        file.close()

def HDF5ImageGenerator(hdf5_x = None, hdf5_y = None, batch_size = None, mode = 'train'):
    sample_count  = hdf5_x.shape[0]

    while True:
        batch_index = 0
        batches_list = list(range(int(ceil(float(sample_count) / batch_size))))
        if mode == 'train':
            shuffle(batches_list)

        while batch_index < len(batches_list):
            batch_number = batches_list[batch_index]
            start        = batch_number * batch_size
            end          = min(start + batch_size, sample_count)

            # Load data from disk
            x = hdf5_x[start: end]
            y = hdf5_y[start: end]

            # Augment batch

            batch_index += 1
            

            yield x,y