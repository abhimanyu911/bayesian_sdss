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
ensemble_size = 100

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


def get_ensembled_classification_report(X_test = None, y_test = None, ensemble_size = None, model = None, threshold = None):
    """
    Returns the classification report of the ensembled model.
    """
    if X_test is None or y_test is None or ensemble_size is None or model is None or threshold is None:
        raise ValueError("X_test, y_test, ensemble_size and model must be provided.")
    else:
        # Get the predictions of the ensemble
        predicted_probabilities = np.zeros((X_test.shape[0], 9))
        for i in range(ensemble_size):
            predicted_probabilities += model.predict(X_test)
        predicted_probabilities /= ensemble_size
        classified_sample_predictions = []
        classified_sample_labels = []
        for i in range(X_test.shape[0]):
            if np.max(predicted_probabilities[i]) > threshold:
                classified_sample_predictions.append(np.argmax(predicted_probabilities[i]))
                classified_sample_labels.append(np.argmax(y_test[i]))
        
        samples_classified = len(classified_sample_predictions)/X_test.shape[0]*100

        return samples_classified, metrics.classification_report(classified_sample_labels, classified_sample_predictions)


def get_ensembled_entropy_report(X_test = None, y_test = None, ensemble_size = None, model = None, threshold = None):
    """
    Returns the entropy report of the ensembled model.
    """
    if X_test is None or y_test is None or ensemble_size is None or model is None or threshold is None:
        raise ValueError("X_test, y_test, ensemble_size and model must be provided.")
    else:
        # Get the predictions of the ensemble
        mean_predicted_probabilities = np.zeros((X_test.shape[0], 9))
        entropies = np.zeros((X_test.shape[0], 1))
        for i in range(ensemble_size):
            predicted_probabilities = model.predict(X_test)
            mean_predicted_probabilities += predicted_probabilities
            entropies += np.expand_dims(-np.sum(predicted_probabilities * np.log2(predicted_probabilities), axis = 1), axis = 1)
        entropies /= ensemble_size
        mean_predicted_probabilities /= ensemble_size
        classified_sample_predictions = []
        classified_sample_labels = []
        for i in range(X_test.shape[0]):
            if entropies[i] < threshold:
                classified_sample_predictions.append(np.argmax(mean_predicted_probabilities[i]))
                classified_sample_labels.append(np.argmax(y_test[i]))
        
        samples_classified = len(classified_sample_predictions)/X_test.shape[0]*100

        return samples_classified, metrics.classification_report(classified_sample_labels, classified_sample_predictions)



def plot_sample_with_confidence(sample_index=None, X_test=None, y_test=None, ensemble_size=None, mode=None, style = None, model = None):
    # Get the sample image and true label
    sample_image = X_test[sample_index]
    if mode is None:
        true_label = np.argmax(y_test[sample_index])
        #if true_label > 5:
         #   true_label += 1
    else:
        true_label = y_test[sample_index]
    # Initialize an array to store the predicted probabilities for each class
    predicted_probabilities = np.zeros((ensemble_size, 9))
    # Make predictions using each Bayesian network
    for i in range(ensemble_size):
        predicted_probabilities[i] = model.predict(sample_image.reshape(1, 69, 69, 3))
    pct_2p5 = np.array([np.percentile(predicted_probabilities[:, i], 2.5) for i in range(9)])
    pct_97p5 = np.array([np.percentile(predicted_probabilities[:, i], 97.5) for i in range(9)])
    bar_height = pct_97p5 - pct_2p5
    # Calculate the mean probabilities for each class
    mean_probabilities = np.mean(predicted_probabilities, axis=0)
    # Create a bar chart with the mean probabilities and confidence intervals
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), gridspec_kw={'width_ratios': [3, 3]})
    ax1.imshow(sample_image)
    ax1.axis('off')
    if mode is None:
        plt.rcParams['axes.titlesize'] = 10
        ax1.set_title(f'True label: {true_label}')
    else:
        plt.rcParams['axes.titlesize'] = 10
        ax1.set_title(f'Disk, Edge-on, Boxy Bulge(Lies outside the training set)')
    x = np.arange(9)
    colours = 'green'
    if style == 'bar':
        bars = ax2.bar(x, bottom=pct_2p5, height=bar_height, width=0.8, color=colours, alpha=0.5)
        for i, bar in enumerate(bars):
            bar_x = bar.get_x()
            bar_width = bar.get_width()
            bar_height = bar.get_height()
            mean_probability = mean_probabilities[i]
            
            if mean_probability > 0.02:
                ax2.plot([bar_x, bar_x + bar_width],
                         [mean_probability, mean_probability],
                         color='black', linestyle='dashed')
                ax2.text(bar_x + bar_width, mean_probability, f'{mean_probability:.2f}', verticalalignment='center')
            ax2.legend(['Mean Probability'])
            
    ax2.set_xticks(x, ['0', '1', '2', ' 3', '4', '5', '6', '7', '8'])
    ax2.set_ylim([0, 1.05])
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Probability')
    if style == 'bar':
        plt.rcParams['axes.titlesize'] = 10
        ax2.set_title('Sample Classification with 95% CI')
    plt.show()



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