# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:21:04 2020

@author: Harsha Vardhan Seelam
"""
from PIL import Image
import glob
import numpy as np
import idx2numpy as inp
import tensorflow as tf
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.optimizers import SGD
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix

# =============================================================================
#                             Functions for Program
# =============================================================================

def Generate_labels(Data_set):
    print("Generating Labels....")
    Data_set = Data_set.reshape(Data_set.shape[0],78*78)
    ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(Data_set)
    return ward.labels_

def Hep_CNN_train(X_train, X_test, Y_train, Y_test, input_shape):
    print("Loading hyperparameters......")
    #Hyper Parameters
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.00005
    epochs = 35
    batch_size = 77
    verbose = 1
    
    print("Training And validating Model.....")
    #Adding pooling, dense layers to an an non-optimized empty CNN
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(7,7),activation = tf.nn.leaky_relu, input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(16, kernel_size=(4,4),activation = tf.nn.leaky_relu))
    model.add(MaxPooling2D(pool_size = (3, 3)))
    model.add(Conv2D(32, kernel_size=(3,3),activation = tf.nn.leaky_relu))
    model.add(MaxPooling2D(pool_size = (3, 3)))
    model.add(Flatten())
    model.add(Dense(150, activation = tf.nn.leaky_relu, kernel_regularizer = keras.regularizers.l2(weight_decay)))
    model.add(Dropout(0))
    model.add(Dense(6, activation = tf.nn.softmax))
    
    #setting an optimizer with a given loss function
    opt = SGD(lr = lr, momentum = momentum)
    model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    Hist = model.fit(x = X_train,y = Y_train, epochs = epochs, batch_size = batch_size, verbose = verbose, validation_data=(X_test, Y_test))
    return Hist , model

def plt_accuracy(Hist):
    train_loss = Hist.history['loss']
    val_loss = Hist.history['val_loss']
    train_acc = Hist.history['accuracy']
    val_acc = Hist.history['val_accuracy']
    xc = range(35)
    
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.style.use(['bmh']) 
    plt.show()
    
    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.style.use(['bmh'])
    plt.show()
    

# =============================================================================
#                             MAIN PROGRAM
# =============================================================================
# ===========================|| Acquiring data-set ||=========================
print("Extracting the data of HEP-2 Cell Image Classification.....")

Train_images = glob.glob('datasets/training/*.png')
train_data = np.array([np.array(Image.open(fname)) for fname in Train_images])

Valid_images = glob.glob('datasets/*.png')
valid_data = np.array([np.array(Image.open(fname)) for fname in Valid_images])

Test_images = glob.glob('datasets/*.png')
test_data = np.array([np.array(Image.open(fname)) for fname in Test_images])

# ===========================||normalizing data-set ||=========================
print("normalizing Data .....")
train_data = train_data.astype('float32')
valid_data = valid_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
valid_data /= 255
test_data /= 255

# ===========================|| Generating Labels ||=========================
train_labels = Generate_labels(train_data)
valid_labels = Generate_labels(valid_data)
test_labels = Generate_labels(test_data)

# ===========================|| Preparing dataset ||=========================
print("Data reshaping")
train_data = train_data.reshape(train_data.shape[0],78,78,1)
valid_data = valid_data.reshape(valid_data.shape[0],78,78,1)
test_data = test_data.reshape(test_data.shape[0],78,78,1)
input_shape = (78, 78, 1)

# ===========================|| Trainning MOdel ||=============================
print("CNN for Cell Image Classification")
Hist, model = Hep_CNN_train(train_data, valid_data, train_labels, valid_labels, input_shape)

plt_accuracy(Hist)

# ===========================|| Tesing the acquired model ||===================
step_4_score = model.evaluate(test_data, test_labels, verbose=0)
print('HEP-2 Test Loss:', step_4_score[0])
print('HEP-2 Test accuracy:', step_4_score[1])
# =============================================================================
#                              END
# =============================================================================
