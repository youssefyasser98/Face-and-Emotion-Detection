# -*- coding: utf-8 -*-
"""
Created on Thu May 28 23:43:17 2020

@author: moda9
"""

from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense,Dropout,
Activation,Flatten,BatchNormalization,Conv2D,MaxPooling2D,
GlobalAveragePooling2D,SeparableConv2D,Activation)
from tensorflow.keras.optimizers import RMSprop,SGD,Adam
from tensorflow.keras.callbacks import (ModelCheckpoint,EarlyStopping,
ReduceLROnPlateau,CSVLogger)
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#Some Help Functions



#Analysis Of Dataset
IMAGE_RES = 48
Batch_size = 64
base_dir = os.path.dirname("fer2013\\")

classes = ['Angry', 'Happy', 'Neutral', 'Sad']
num_classes = len(classes)
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
total_train=0
total_val=0
for cl in classes:   
  img_path = os.path.join(train_dir,cl)
  img_path_val = os.path.join(validation_dir,cl)
  images = len(os.listdir(img_path))
  images_val = len(os.listdir(img_path_val))
  total_train+= images
  total_val+= images_val
  print("{}: {} Train Images".format(cl, images))
  print("{}: {} Validation Images".format(cl,images_val))

print("Total Train Images: "+str(total_train))
print("Total Validation Images: "+str(total_val))

image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    zoom_range=.2,
                    shear_range=.2,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    fill_mode='nearest'
                    )

image_gen_val = ImageDataGenerator(rescale=1./255)

train_data_gen = image_gen_train.flow_from_directory(
                                                batch_size=Batch_size,
                                                color_mode='grayscale',
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMAGE_RES,IMAGE_RES),
                                                class_mode='categorical'
                                                )

val_data_gen = image_gen_val.flow_from_directory( directory=validation_dir,
                                                  color_mode='grayscale',
                                                  batch_size=Batch_size,
                                                  target_size=(IMAGE_RES, IMAGE_RES),
                                                  class_mode='categorical',
                                                  shuffle=True)
regularization = l2(l=0.01)
model=Sequential([#Block-1
                  Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',
                         input_shape=(IMAGE_RES,IMAGE_RES,1)),
                  Activation('elu'),
                  BatchNormalization(),
                  Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal'
                         ,input_shape=(IMAGE_RES,IMAGE_RES,1)),
                  Activation('elu'),
                  BatchNormalization(),
                  MaxPooling2D(pool_size=(2,2)),
                  Dropout(0.2),
               #Block-2
                   Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'
                          ,input_shape=(IMAGE_RES,IMAGE_RES,1)),
                   Activation('elu'),
                  BatchNormalization(),
                  Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'
                         ,input_shape=(IMAGE_RES,IMAGE_RES,1)),
                  Activation('elu'),
                  BatchNormalization(),
                  MaxPooling2D(pool_size=(2,2)),
                  Dropout(0.2),
               #Block-3
                  Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal',
                         input_shape=(IMAGE_RES,IMAGE_RES,1)),
                  Activation('elu'),
                  BatchNormalization(),
                  Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'
                         ,input_shape=(IMAGE_RES,IMAGE_RES,1)),
                  Activation('elu'),
                  BatchNormalization(),
                  MaxPooling2D(pool_size=(2,2)),
                  Dropout(0.2),
               #Block-4
                  Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal',
                         input_shape=(IMAGE_RES,IMAGE_RES,1)),
                  Activation('elu'),
                  BatchNormalization(),
                  Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'
                         ,input_shape=(IMAGE_RES,IMAGE_RES,1)),
                  Activation('elu'),
                  BatchNormalization(),
                  MaxPooling2D(pool_size=(2,2)),
                  Dropout(0.2),
               #Block-5
                  Flatten(),
                  Dense(512,kernel_initializer='he_normal'),
                  Activation('elu'),
                  BatchNormalization(),
                  Dropout(0.5),
               #Block-6
                  Dense(1024,kernel_initializer='he_normal'),
                  Activation('elu'),
                  BatchNormalization(),
                  Dropout(0.5),
                  Dense(num_classes,kernel_initializer='he_normal',activation='softmax')
               ])


checkpoint = ModelCheckpoint('Emotion.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=9,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=6,
                              verbose=1,
                              min_delta=0.0001)
CSV_logger = CSVLogger("Train.log", separator=',', append=True)

callbacks = [earlystop,checkpoint,reduce_lr,CSV_logger]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(),
              metrics=['accuracy'])

model.summary()

epochs=32

history=model.fit_generator(
                train_data_gen,
                steps_per_epoch=total_train//Batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=val_data_gen,
                validation_steps=total_val//Batch_size)


# reloaded = tf.keras.models.load_model("Emotion.h5")
# history = reloaded.fit_generator(
#                 train_data_gen,
#                 steps_per_epoch=total_train//Batch_size,
#                 epochs=1,
# #                 callbacks=callbacks,
#                 validation_data=val_data_gen,
#                 validation_steps=total_val//Batch_size)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()













