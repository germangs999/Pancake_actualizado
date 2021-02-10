# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:26:01 2021

@author: germa
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import hdf5storage
from tensorflow import keras
import datetime
import csv

#gpu_options = tf.G
#tensorboard --logdir='D:/germa/DriveUP/TransferLearningFull/DB/logs/fit/20210201-212652/'

#Funcio para leer las imagenes de los arreglos
def LeerEstructuras(nombre, no_im):
    Arreglo = hdf5storage.loadmat(nombre)
    Imagenes = Arreglo['Arreglo'].astype('float16')
    etiquetas = Arreglo['etiqueta']
    no_img = no_im*np.ones(shape=(Imagenes.shape[0]))
    Imagenes = Imagenes.reshape((Imagenes.shape[0], Imagenes.shape[1], Imagenes.shape[2],1))
    return Imagenes, etiquetas, no_img

###Leer las imagenes y sus etiqeutas
Imagenes_9E49, etiquetas_9E49, noimg_9E49 = LeerEstructuras('D:/germa/DriveUP/TransferLearningFull/UINT8/Filtrado1/VentanasHHu8_full_traslaTodo_w224_9E49_TN_calibDB_UINT8_FIX.mat',0)
Imagenes_0309, etiquetas_0309, noimg_0309 = LeerEstructuras('D:/germa/DriveUP/TransferLearningFull/UINT8/Filtrado1/VentanasHHu8_full_traslaTodo_w224_0309_TN_calibDB_UINT8_FIX.mat',1)
Imagenes_D419, etiquetas_D419, noimg_D419 = LeerEstructuras('D:/germa/DriveUP/TransferLearningFull/UINT8/Filtrado1/VentanasHHu8_full_traslaTodo_w224_D419_TN_calibDB_UINT8_FIX.mat',2)
Imagenes_FFAF, etiquetas_FFAF, noimg_FFAF = LeerEstructuras('D:/germa/DriveUP/TransferLearningFull/UINT8/Filtrado1/VentanasHHu8_full_traslaTodo_w224_FFAF_TN_calibDB_UINT8_FIX.mat',3)

#Concatenar las matrices con las imagenes de 234x234 y generar las 3 dimensiones RGB
X_train = np.concatenate((Imagenes_9E49, Imagenes_0309, Imagenes_D419, Imagenes_FFAF), axis=0)
X_train = np.repeat(X_train, 3, -1)
print(X_train.shape) 

no_img = np.concatenate((noimg_9E49, noimg_0309, noimg_D419, noimg_FFAF), axis=0)
etiquetas = np.concatenate((etiquetas_9E49, etiquetas_0309, etiquetas_D419, etiquetas_FFAF), axis=0)
y_train = keras.utils.to_categorical(etiquetas)
#Eliminar las variables pesadas
del Imagenes_9E49, Imagenes_0309, Imagenes_D419, Imagenes_FFAF

##Cálculo del promedio y la desviación estándar
# mean_X_train = np.mean(X_train)
# var_X_train = np.mean((X_train-mean_X_train)**2)


############Modelo##################
base_model=tf.keras.applications.ResNet101(weights='imagenet',include_top=False)
#congelar el modelo base \
base_model.trainable = False
#Crear un nuevo modelo para la parte superior
inputs = keras.Input(shape=(224, 224, 3))
# norm_layer = keras.layers.experimental.preprocessing.Normalization(mean=mean_X_train, variance=var_X_train)
# x = norm_layer(inputs)
x = keras.applications.resnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x) #Para HH
x = keras.layers.Dense(1024)(x)
x = keras.layers.LeakyReLU()(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.25)(x)
x = keras.layers.Dense(1024)(x)
x = keras.layers.LeakyReLU()(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.25)(x)
x = keras.layers.Dense(512)(x)
x = keras.layers.LeakyReLU()(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.4)(x)
outputs = keras.layers.Dense(4, activation='softmax')(x)

model = keras.Model(inputs, outputs)

#compilar y entrenar
model.compile(optimizer = keras.optimizers.Nadam(0.1), loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics = [keras.metrics.Accuracy(), keras.metrics.MeanSquaredError(), keras.metrics.Recall()])

filepath="D:/germa/Pancake_modelos/ResNET_HH4_TF_2normal.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
earlystop = keras.callbacks.EarlyStopping(monitor='accuracy',  min_delta=0.001, patience=15, mode='auto',restore_best_weights=False)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', min_delta=0.001, factor=0.5, patience=10, min_lr=0.0001, mode='max')
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model_directory='logs/history/' 
class StoreModelHistory(tf.keras.callbacks.Callback):

  def on_epoch_end(self,batch,logs=None):
    if ('lr' not in logs.keys()):
      logs.setdefault('lr',0)
      logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

    if not ('model_history2.csv' in os.listdir(model_directory)):
      with open(model_directory+'model_history2.csv','a') as f:
        y=csv.DictWriter(f,logs.keys())
        y.writeheader()

    with open(model_directory+'model_history2.csv','a') as f:
      y=csv.DictWriter(f,logs.keys())
      y.writerow(logs)

callbacks_list = [checkpoint, earlystop, reduce_lr, tensorboard_callback,StoreModelHistory()]

epocas_1 = 500
model.fit(X_train, y_train, epochs=epocas_1,verbose=1, batch_size=32, callbacks = callbacks_list)


####Ajuste fino
base_model.trainable = True
model.summary()
model.compile(optimizer = keras.optimizers.Nadam(1e-4), loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics = [keras.metrics.Accuracy(), keras.metrics.MeanSquaredError(), keras.metrics.Recall()])
##Callbacks
filepath2="D:/germa/Pancake_modelos/ResNET_HH4_TF_2Fino2.hdf5"
checkpoint2 = keras.callbacks.ModelCheckpoint(filepath2, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
earlystop2 = keras.callbacks.EarlyStopping(monitor='accuracy',  min_delta=0.01, patience=15, mode='auto',restore_best_weights=False)
reduce_lr2 = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', min_delta=0.01, factor=0.1, patience=10, min_lr=1e-7, mode='max')
callbacks_list2 = [checkpoint2, earlystop2, reduce_lr2, tensorboard_callback, StoreModelHistory()]
epocas_2 = 200
model.fit(X_train, y_train, epochs=epocas_2,verbose=1, batch_size=32, callbacks = callbacks_list2)


# tf.keras.models.save_model(model, filepath2)

# model = tf.keras.models.load_model(filepath)
# model.

# import pandas as pd
# import matplotlib.pyplot as plt

# EPOCH = 10 # number of epochs the model has trained for

# history_dataframe = pd.read_csv(model_directory+'model_history.csv',sep=',')


# # Plot training & validation loss values
# plt.style.use("ggplot")
# plt.plot(range(1,EPOCH+1),
#          history_dataframe['loss'])
# plt.plot(range(1,EPOCH+1),
#          history_dataframe['val_loss'],
#          linestyle='--')
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()