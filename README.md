# Pancake_actualizado
Actualización de la detección de hielo Frazil-Pancake con transfer learning

# Entrenamiento del modelo

Las paqueterías necesarias para ejecutar el entrenamiento de la clasificación del código **Prueba.py** son:

```python
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import hdf5storage
from tensorflow import keras
import datetime
import csv
```

El objetivo de este modelo de clasificación es obtener un descripción para distinguir entre las siguientes cuatro clases:

1. Pancake. 
2. Mar Blanco (mar agitado).
3. Mar Negro (mar calmo).
4. Tierra y trozos de hielo.

Por cada imagen satelital se obtienen 2000 parches de cada clase, es decir, al final se cuenta con 8000 parches por clase para entrenamiento. Cada parche mide 224 x 224 pixeles porque es el tamaño de entrada requerido por la ResNet. Estos parches están almacenados en archivos de Matlab que se pueden leer con la siguiente función:

```python
def LeerEstructuras(nombre, no_im):
    Arreglo = hdf5storage.loadmat(nombre)
    Imagenes = Arreglo['Arreglo'].astype('float16')
    etiquetas = Arreglo['etiqueta']
    no_img = no_im*np.ones(shape=(Imagenes.shape[0]))
    Imagenes = Imagenes.reshape((Imagenes.shape[0], Imagenes.shape[1], Imagenes.shape[2],1))
    return Imagenes, etiquetas, no_img
```

El modelo base utilizado es la ResNET101 y al final se concatenan cuatro FCN: dos de 1024, una de 512 y la salida de 4 neuronas. Al principio del modelo base es necesario colocar una etapa de preprocesamiento propia de la ResNET. En la última etapa, entre cada FCN, se agregan capas de Dropout y Batch Normalization para evitar el sobreajuste del modelo. La primera etapa de ajuste del modelo se realiza con el congelamiento de los pesos del modelo base, es decir, solamente se ajustarán los pesos de la etapa de los FCN. Por lo tanto, el modelo completo es: 

```python
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
model.compile(optimizer = keras.optimizers.Nadam(0.1), loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics = [keras.metrics.Accuracy(), keras.metrics.MeanSquaredError(), keras.metrics.Recall()])

```

Además, se utilizaron 4 callbacks:
1. Almacenar el modelo que tenga el modelo con la mejor exactitud.
2. Paro anticipado para que detenga el ajuste cuando no se ha presentado una mejora en la exactitud de clasificación en cierto número de épocas.
3. Reducir la tasa de aprendizaje (learning rate) cuando no exista una mejora en la exactitud cada cierto número de épocas.
4. Almacenar los datos en cada época en el documento **model_history2.csv**

Finalmente se entrena el modelo bajo las condiciones antes mencionadas:
```python
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
model.fit(X_train, y_train, epochs=epocas_1,verbose=1, batch_size=32, callbacks = callbacks_list)
```

La segunda etapa del entrenamiento consta del ajuste de pesos del modelo completo, es decir, descongelamos el modelo base y agregamos callback semejantes a los anteriores:

```python
checkpoint2 = keras.callbacks.ModelCheckpoint(filepath2, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
earlystop2 = keras.callbacks.EarlyStopping(monitor='accuracy',  min_delta=0.01, patience=15, mode='auto',restore_best_weights=False)
reduce_lr2 = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', min_delta=0.01, factor=0.1, patience=10, min_lr=1e-7, mode='max')
callbacks_list2 = [checkpoint2, earlystop2, reduce_lr2, tensorboard_callback, StoreModelHistory()]
epocas_2 = 200
model.fit(X_train, y_train, epochs=epocas_2,verbose=1, batch_size=32, callbacks = callbacks_list2)
```
