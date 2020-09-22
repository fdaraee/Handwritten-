import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pdb
from skimage import io
import sys
import pandas as pd
from scipy.misc import imread
import pickle
import matplotlib.pyplot as plt
import keras
import numpy.random as rng
import tensorflow.compat.v1 as tf
import random 
SEED = 1
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)


from tensorflow.keras import backend as K
import time
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from tensorflow.keras.models import Model

import tensorflow.keras
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.pooling import MaxPooling2D

from tensorflow.python.keras.layers.core import Lambda, Flatten, Dense
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Layer
import functools
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.vis_utils import plot_model
import pydotplus
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from sklearn.utils import shuffle
import pandas as pd

tensorflow.compat.v1.disable_eager_execution()

def scale(x):
    # normalize data
    x = (x - 127.5) / 127.5
    return x

def load_samples(csv_file):
    data = pd.read_csv(os.path.join('data_files',csv_file))
    data = data[['FileName', 'Label', 'ClassName']]
    file_names = list(data.iloc[:,0])
    # Get the labels present in the second column
    labels = list(data.iloc[:,1])
    classname = list(data.iloc[:,2])
    samples = []
    for samp,lab,clasnam  in zip(file_names,labels,classname):
        samples.append([samp,lab,clasnam])
    return samples
    
def shuffle_data(data):
	data = shuffle(data)#,random_state=2)
	return data
def generator(samples, batch_size=128,shuffle=True,image_sizeW=48, image_sizeH=128):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    if shuffle:
    	samples = shuffle_data(samples)
    while True:
    	# Loop forever so the generator never terminates  

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset+batch_size]

            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []
            
            # For each example
            for batch_sample in batch_samples:
                # Load image (X) and label (y)
                img_name = batch_sample[0]
                label = batch_sample[1]
                
                img =  cv2.imread(img_name)
                # apply any kind of preprocessing
                img = np.resize(img,(image_sizeW, image_sizeH,1))
                imge = scale(img)
                # Add example to arrays
                X_train.append(imge)
                y_train.append(label)

            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # The generator-y part: yield the next training batch            
            yield X_train, y_train

#############################################################################
def get_Monte_Carlo_model(input_shape, dropout_rate=0.15, reuse=False):

     
    x = tf.keras.Input(input_shape)
    
    with tf.variable_scope("Network_model"):
        # Convolutional Layer #1
        with tf.variable_scope("conv1", reuse=reuse):
        	conv1 = tf.keras.layers.Conv2D(32 ,(3, 3), strides = 1, padding="same", activation='relu', name='conv1')(x)
        	#conv1_1 = BatchNormalization()(conv1)
        	dropout1 = tf.keras.layers.Dropout(dropout_rate)(conv1,training=True)

        # Pooling Layer #1
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(dropout1)

        # Convolutional Layer #2 and Pooling Layer #2
        with tf.variable_scope("conv2", reuse=reuse):
            conv2 = tf.keras.layers.Conv2D(64 ,(3, 3), strides = 1, padding="same", activation='relu', name='conv2')(pool1)
            dropout2 = tf.keras.layers.Dropout(dropout_rate)(conv2,training=True)
            
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(dropout2)

        with tf.variable_scope("conv3", reuse=reuse):
            conv3 = tf.keras.layers.Conv2D(128 ,(3, 3), strides = 1, padding="same", activation='relu', name='conv3')(pool2)
            dropout3 = tf.keras.layers.Dropout(dropout_rate)(conv3,training=True)
            
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(dropout3)

        # Dense Layer
        with tf.variable_scope("conv4", reuse=reuse):
            conv4 = tf.keras.layers.Conv2D(256 ,(3, 3), strides = 1, padding="same", activation='relu', name='conv4')(pool3)
            dropout4 = tf.keras.layers.Dropout(dropout_rate)(conv4, training=True)

        pool4 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(dropout4)

        with tf.variable_scope("conv5", reuse=reuse):
            conv5 = tf.keras.layers.Conv2D(512 ,(3, 3), strides = 1, padding="same", activation='relu', name='conv5')(pool4)
            dropout5 = tf.keras.layers.Dropout(dropout_rate)(conv5, training=True)

        pool5 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(dropout5)

        # Logits Layer
        with tf.variable_scope("fc1", reuse=reuse):
            pool5_flat = tf.keras.layers.Flatten()(pool5)
            dense = tf.keras.layers.Dense(units=1024, activation='relu')(pool5_flat)
            
        model = tf.keras.Model(inputs = [x] , outputs = [dense])
        

    return model

 
 ##########################################################################3
def _center_loss_func(labels,features, alpha, num_classes, centers, feature_dim):
    assert feature_dim == features.get_shape()[1]    
    labels = K.reshape(labels, [-1])
    #labels = K.argmax(labels, axis=1)
    labels = tf.to_int32(labels)
    centers_batch = K.gather(centers, labels)
    diff = (1 - alpha) * (centers_batch - features)
    centers = tf.scatter_sub(centers, labels, diff)
    centers_batch = K.gather(centers, labels)
    loss = K.mean(K.square(features - centers_batch))
    return loss

def get_center_loss(alpha, num_classes, feature_dim):
    """Center loss based on the paper "A Discriminative 
       Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """    
    # Each output layer use one independed center: scope/centers
    centers = K.zeros([num_classes, feature_dim], dtype='float32')
    @functools.wraps(_center_loss_func)
    def center_loss(y_true, y_pred):
        return _center_loss_func(y_true, y_pred, alpha, num_classes, centers, feature_dim)
    return center_loss                

def multiple_outputs(samples, batch_size):
    gen = generator(samples, batch_size=128)
    
    while True:
        gnext = gen.__next__()
        # return image batch and 3 sets of lables
        yield gnext[0], [gnext[1], gnext[1]]
###################################################################################
if __name__=='__main__':
	
	train_folder = '/media/daraei/4bd1b8d6-a8e3-4f33-92f6-4368cab72c60/doctora/Botany_database/Konzil_augment/'
	
	batch_size = 32
	num_classes = len(os.listdir(train_folder))
	dropout_rate=0.15
	input_shape = (48,128,1)

	train_samples = load_samples('wordspotting_Konzil_train1.csv')
	validation_samples = load_samples('wordspotting_Konzil_test1.csv')
	train_generator = multiple_outputs(train_samples, batch_size=128)
	validation_generator = multiple_outputs(validation_samples, batch_size=128)
	
	num_train_samples = (len(train_samples))
	num_validation_samples = len(validation_samples)
	model = get_Monte_Carlo_model(input_shape, dropout_rate)
	model.summary()
	plot_model(model, to_file='montecarlo_Konzil_final.png', show_shapes=True, show_layer_names=True)

	last_layer = model.layers[-1].output
	#uncertainty = Dense(units=num_classes, name = 'uncertainty')(last_layer) #learns to predict aleatoric uncertainty as output
	x = Dense(128, activation = 'relu', name = 'features')(last_layer)
	features = x
	#x = Dropout(0.15)(x, training=True)
	x = Dense(num_classes, activation = 'softmax', name = 'prediction')(x)
	#threshold = -1 * tf.log(x)
	custom_model = Model(inputs = model.inputs, outputs = [x, features])
	custom_model.summary()
	plot_model(custom_model, to_file='custom_model_montecarlo_Konzil.png', show_shapes=True, show_layer_names=True)
	#pdb.set_trace()

	features = custom_model.layers[-2].output
	num_classes = custom_model.layers[-1].output_shape[1]
	feature_dim = custom_model.layers[-2].output_shape[1]
	print('num_classes:', num_classes, ', feature_dim ', feature_dim)

	center_loss = get_center_loss(0.5, num_classes, feature_dim)

	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	
	custom_model.compile(loss = {'prediction': 'sparse_categorical_crossentropy', 'features': center_loss}, optimizer= adam,
                                      metrics= {'prediction': 'accuracy'})
	pdb.set_trace()
	##############################################################################
	history = custom_model.fit_generator(train_generator, steps_per_epoch= (num_train_samples)//batch_size, validation_data=validation_generator, validation_steps = (num_validation_samples) // batch_size,epochs=25)

	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['prediction_accuracy'])
	plt.plot(history.history['val_prediction_accuracy'])
	plt.title('Model accuracy of prediction')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()
	plt.plot(history.history['features_loss'])
	plt.plot(history.history['val_features_loss'])
	plt.title('model loss of features')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['prediction_loss'])
	plt.plot(history.history['val_prediction_loss'])
	plt.title('Model loss of prediction')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()
	pdb.set_trace()

	model.save('model_montecarlo_Konzil1.h5')
	custom_model.save('custommodel_montecarlo_Konzil1.h5')
	