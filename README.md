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
####Ajuste fino
base_model.trainable = True
model.compile(optimizer = keras.optimizers.Nadam(1e-4), loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics = [keras.metrics.Accuracy(), keras.metrics.MeanSquaredError(), keras.metrics.Recall()])
checkpoint2 = keras.callbacks.ModelCheckpoint(filepath2, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
earlystop2 = keras.callbacks.EarlyStopping(monitor='accuracy',  min_delta=0.01, patience=15, mode='auto',restore_best_weights=False)
reduce_lr2 = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', min_delta=0.01, factor=0.1, patience=10, min_lr=1e-7, mode='max')
callbacks_list2 = [checkpoint2, earlystop2, reduce_lr2, tensorboard_callback, StoreModelHistory()]
epocas_2 = 200
model.fit(X_train, y_train, epochs=epocas_2,verbose=1, batch_size=32, callbacks = callbacks_list2)
```

Una vez que se calculan y almacenan los pesos del mejor modelo. Las imágenes del conjunto de validación son procesadas por el programa llamado **SegmentacionImgsValidacion.py**. Cada una de esas imágenes debe ser dividida en parches de 224 x 224 pixeles y para ello se tiene la siguiente función:

```python
def dividir_conjunto_imagen(I, tam):
    tamx, tamy, tamz = I.shape
    ventx = floor(tamx/tam)
    venty = floor(tamy/tam)
    ##Arreglofinal de las imagenes divididas
    arreglo_imagen = np.empty((ventx*venty, tam, tam,3), dtype=I.dtype)
    I_rec = I[:(ventx*tam),:(venty*tam)]
    #for k in range(I_rec.shape[0]):
        #print(k)
    patches2 = view_as_blocks(I_rec, block_shape=(tam,tam,3))
    arreglo_imagen=patches2.reshape(patches2.shape[0]*patches2.shape[1],patches2.shape[3], patches2.shape[4],3)
    return I_rec,arreglo_imagen,ventx,venty
```

Se carga el modelo calculado anteriormente y cada parche es clasificado. Se almacenan las probabilidades de clase para cada parche en un archivo **.mat** para su posterior procesamiento:

```python
y_pred = model.predict(X_train)
savemat(carp_etiqueta + modelo[-2:] + modelos[mod][-7:-5] +'_y_pred_'+ name_imgs[img][:4] +'.mat', {'y_pred': y_pred})
```

Ahora, el despliegue de resultados se realiza a través del programa **DespliegueSegmentacion.m**. De los datos almacenados el archivo **.mat** se puede calcular la etiqueta de clase para cada parche de 224 x 224 pixeles, entonces hay que reconstruir una máscara de segmentación del mismo tamaño de la imagen por cada parche que la compone. La construcción de la máscara se obtiene a partir de determinar la clase del parche y reacomodar dichos parches en la posición que ocupan dentro de la imagen original de validación:

```matlab
%Leer las etiquetas
y_pred = load([dir_etiquetas model{mod} '_' fnum{fn} '_y_pred_' im{ix} '.mat']).y_pred;
%Clasificación final
[~,y_pred2] = max(y_pred,[],2);

%Imagen recortada a un tamaño adecuado
I = Imagen(1:(floor(size(Imagen,1)/N))*N, 1:(floor(size(Imagen,2)/N))*N);
%Obtener las ventanas
block_0 = mat2cell(I, repmat(N, [1, floor(size(I,1)/N)]), repmat(N, [1, floor(size(I,2)/N)]));
%Imagen que tendrá las etiquetas de clasificación
Iseg = zeros(size(I));

l = 1;
for k = 1:size(block_0,1)
    for m = 1:size(block_0,2)
        %Creación de las ventanas
        cuadro = y_pred2(l,1).*ones(N);
        Iseg(((k-1)*N)+1:k*N, ((m-1)*N)+1:m*N) = cuadro;
        l = l+1;
    end
end

%Usar solo los que tiene la etiqueta de pancake
Iseg_1 = (Iseg==1);
```

También podemos aprovechar que se tiene las máscaras de tierra para cada imagen de validación y la quitamos de la máscara obtenida anteriormente para formar la máscara final de pancake:

```matlab
%Quitar la tierra
Tierra = load([dir_tierras im{ix} '_LAND.mat']);
Tierra = Tierra.h;
Tierra = logical(Tierra(1:(floor(size(Imagen,1)/N))*N, 1:(floor(size(Imagen,2)/N))*N));

%MÁSCARA FINAL DE PANCAKE
Pancake_final = Iseg_1 & ~Tierra;
```

Finalmente, esa máscara de pancake final se compara con la máscara generada por el experto (ground truth) a través del índice Dice, una métrica muy conocida para la validación de segmentaciones:

```matlab
%Leer las mascaras objetivo
original = load([dir_original im{ix} '_MASK.mat']).BW;
original = original(1:(floor(size(original,1)/N))*N, 1:(floor(size(original,2)/N))*N);
Res_dice(mod,ix,fn) = dice(original,Pancake_final);
```


